"""
Fase 2 paso 3 — Extracción end-to-end de UN video.

Pipeline por frame:
    1. YOLO11n + ByteTrack       → bbox de persona (1 id esperado)
    2. RVM mobilenetv3 (torch)   → alpha matte recurrente → silueta binaria
    3. RTMPose-m (rtmlib/onnx)   → 17 keypoints COCO usando bbox del paso 1
    4. Crop + resize 64×44       → silueta lista para OpenGait
    5. Normalización keypoints   → centrados en cadera, escalados por torso

Salida en data/processed/<subject>/<condition>/:
    - silhouettes.npy      (T, 64, 44) uint8 ∈ {0,255}
    - keypoints.npy        (T, 17, 3)  float32 (x_norm, y_norm, conf)
    - keypoints_raw.npy    (T, 17, 3)  float32 en píxeles del frame original
    - bboxes.npy           (T, 4)      float32 (x1,y1,x2,y2)
    - meta.json            metadatos + estadísticas
    - debug_grid.png       grid 4×4 de frames con overlay (silueta+pose+bbox)

Uso:
    python -m src.preprocess.extract_sequence \
        --video "C:/Proyecto3/Proyecto-Reconocedor-IA/Dataset Crudo/alexbojorquez/normal_1.mp4" \
        --subject alexbojorquez --condition normal \
        --out "C:/Proyecto3/ProyectoChino/data/processed"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

# CRÍTICO: cargar DLLs CUDA antes de importar onnxruntime
from src.preprocess._cuda_dlls import enable_torch_cuda_dlls
enable_torch_cuda_dlls()

import cv2
import numpy as np
import torch
from rtmlib import RTMPose
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extract")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIL_H, SIL_W = 64, 44   # OpenGait GaitBase standard
PAD_RATIO = 0.05        # padding del bbox antes de recortar silueta
MOTION_PX_THRESH = 6    # px desplazamiento mínimo del centro bbox para "está caminando"
MOTION_WIN = 5          # frames consecutivos con motion > thresh
RTMPOSE_ONNX_NAME = "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504"


# ─────────────────────────── modelos ───────────────────────────

def load_models():
    log.info("Cargando YOLO11n…")
    yolo = YOLO("yolo11n.pt")

    log.info("Cargando RVM mobilenetv3…")
    rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True)
    rvm = rvm.to(DEVICE).eval()

    log.info("Cargando RTMPose-m…")
    cache = Path.home() / ".cache" / "rtmlib" / "hub" / "checkpoints"
    onnx_path = cache / f"{RTMPOSE_ONNX_NAME}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"No se encontró {onnx_path}. Corre primero un smoke con `from rtmlib import Body; "
            f"Body(mode='balanced', backend='onnxruntime', device='cuda')` para descargar."
        )
    rtmpose = RTMPose(onnx_model=str(onnx_path), model_input_size=(192, 256),
                      backend="onnxruntime", device="cuda")
    return yolo, rvm, rtmpose


# ─────────────────────────── utilidades ───────────────────────────

@dataclass
class FrameOut:
    bbox: np.ndarray | None         # (4,) float32 en píxeles del frame original
    silhouette: np.ndarray | None   # (SIL_H, SIL_W) uint8
    keypoints_raw: np.ndarray | None  # (17, 3) float32 (x_px, y_px, conf)
    keypoints_norm: np.ndarray | None  # (17, 3) float32 (x, y, conf), x ∈ ~[-1,1]


def detect_person(yolo: YOLO, frame_bgr: np.ndarray) -> np.ndarray | None:
    """Devuelve bbox (x1,y1,x2,y2) de la persona más grande, o None."""
    res = yolo.predict(frame_bgr, classes=[0], verbose=False, device=DEVICE,
                       conf=0.35, iou=0.5)
    if not res or len(res[0].boxes) == 0:
        return None
    boxes = res[0].boxes.xyxy.cpu().numpy()  # (N,4)
    confs = res[0].boxes.conf.cpu().numpy()  # (N,)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = np.argmax(areas * confs)
    return boxes[best].astype(np.float32)


def to_rvm_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 → RGB float [0,1] tensor (1,3,H,W)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(DEVICE).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t


def crop_and_normalize_silhouette(alpha: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """alpha (H,W) float ∈[0,1], bbox (4,) → silueta (SIL_H, SIL_W) uint8."""
    h, w = alpha.shape
    x1, y1, x2, y2 = bbox
    pad_w = (x2 - x1) * PAD_RATIO
    pad_h = (y2 - y1) * PAD_RATIO
    x1 = max(0, int(x1 - pad_w));  y1 = max(0, int(y1 - pad_h))
    x2 = min(w, int(x2 + pad_w));  y2 = min(h, int(y2 + pad_h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((SIL_H, SIL_W), dtype=np.uint8)
    crop = alpha[y1:y2, x1:x2]
    binm = (crop > 0.5).astype(np.uint8) * 255

    # Centro horizontal por centroide del blanco; rellenar para obtener proporción 64×44.
    ys, xs = np.where(binm > 0)
    if len(xs) < 50:
        return np.zeros((SIL_H, SIL_W), dtype=np.uint8)
    cx = int(xs.mean())
    bh, bw = binm.shape
    target_aspect = SIL_W / SIL_H            # 0.6875
    cur_aspect = bw / bh
    if cur_aspect < target_aspect:
        # Demasiado angosto → ensanchar lienzo
        new_bw = int(bh * target_aspect)
        canvas = np.zeros((bh, new_bw), dtype=np.uint8)
        offset = max(0, new_bw // 2 - cx)
        end = min(new_bw, offset + bw)
        canvas[:, offset:end] = binm[:, :end - offset]
        binm = canvas
    else:
        # Demasiado ancho → recortar simétrico al centroide
        new_bw = int(bh * target_aspect)
        half = new_bw // 2
        x_lo = max(0, cx - half);  x_hi = min(bw, x_lo + new_bw)
        x_lo = max(0, x_hi - new_bw)
        binm = binm[:, x_lo:x_hi]

    return cv2.resize(binm, (SIL_W, SIL_H), interpolation=cv2.INTER_AREA)


def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    """COCO-17. Centrar en cadera (mid 11-12), escalar por (cadera→hombro).
    Devuelve (17,3) con x,y normalizados y conf intacto."""
    out = kp.copy().astype(np.float32)
    if out.shape[0] < 17:
        return out
    hip = (out[11, :2] + out[12, :2]) / 2.0
    shoulder = (out[5, :2] + out[6, :2]) / 2.0
    torso = np.linalg.norm(shoulder - hip)
    if torso < 1e-3:
        return out
    out[:, :2] = (out[:, :2] - hip) / torso
    return out


# ─────────────────────────── lógica principal ───────────────────────────

def process_video(video_path: Path, out_dir: Path, subject: str, condition: str,
                  models: tuple | None = None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {video_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    log.info("Video %s | %d frames @ %.1f fps", video_path.name, n_total, fps)

    yolo, rvm, rtmpose = models if models is not None else load_models()
    rec = [None, None, None, None]   # estado recurrente RVM (reset por video)
    downsample_ratio = 0.25          # 1920x1080 → ~480x270 internamente

    frames_out: list[FrameOut] = []
    t0 = time.time()

    with torch.inference_mode():
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            bbox = detect_person(yolo, frame)
            t = to_rvm_tensor(frame)
            fgr, pha, *rec = rvm(t, *rec, downsample_ratio=downsample_ratio)
            alpha = pha[0, 0].clamp(0, 1).cpu().numpy()  # (H, W) float

            sil = None
            kp_raw = None
            kp_norm = None
            if bbox is not None:
                sil = crop_and_normalize_silhouette(alpha, bbox)
                kp, scores = rtmpose(frame, bboxes=np.asarray([bbox]))
                if kp is not None and len(kp) > 0:
                    kp_raw = np.concatenate([kp[0], scores[0][:, None]], axis=1).astype(np.float32)
                    kp_norm = normalize_keypoints(kp_raw)

            frames_out.append(FrameOut(bbox, sil, kp_raw, kp_norm))
            idx += 1
            if idx % 30 == 0:
                log.info("  frame %d/%d  (%.1f fps)", idx, n_total, idx / (time.time() - t0))

    cap.release()
    dt = time.time() - t0
    log.info("Procesado %d frames en %.1fs (%.1f fps)", len(frames_out), dt, len(frames_out) / dt)

    # ─────── recorte de tramo útil por motion del bbox ───────
    centers = np.array(
        [[(f.bbox[0] + f.bbox[2]) / 2 if f.bbox is not None else np.nan,
          (f.bbox[1] + f.bbox[3]) / 2 if f.bbox is not None else np.nan]
         for f in frames_out]
    )
    motion = np.full(len(frames_out), np.nan)
    for i in range(1, len(frames_out)):
        d = np.linalg.norm(centers[i] - centers[i - 1])
        motion[i] = d
    moving = (motion > MOTION_PX_THRESH).astype(int)
    # primer i tal que MOTION_WIN consecutivos ≥ thresh
    start, end = 0, len(frames_out) - 1
    for i in range(len(moving) - MOTION_WIN):
        if moving[i:i + MOTION_WIN].sum() == MOTION_WIN:
            start = i
            break
    for i in range(len(moving) - 1, MOTION_WIN, -1):
        if moving[i - MOTION_WIN:i].sum() == MOTION_WIN:
            end = i
            break
    log.info("Tramo útil: frames %d→%d (%d frames)", start, end, end - start + 1)

    # ─────── arrays alineados (descartando frames sin bbox) ───────
    sils, kps_raw, kps_norm, bboxes, used_idx = [], [], [], [], []
    for i in range(start, end + 1):
        f = frames_out[i]
        if f.bbox is None or f.silhouette is None or f.keypoints_norm is None:
            continue
        sils.append(f.silhouette)
        kps_raw.append(f.keypoints_raw)
        kps_norm.append(f.keypoints_norm)
        bboxes.append(f.bbox)
        used_idx.append(i)

    sils_a    = np.stack(sils, axis=0) if sils else np.zeros((0, SIL_H, SIL_W), dtype=np.uint8)
    kps_raw_a = np.stack(kps_raw, axis=0) if kps_raw else np.zeros((0, 17, 3), dtype=np.float32)
    kps_norm_a = np.stack(kps_norm, axis=0) if kps_norm else np.zeros((0, 17, 3), dtype=np.float32)
    bboxes_a  = np.stack(bboxes, axis=0) if bboxes else np.zeros((0, 4), dtype=np.float32)

    # ─────── persistencia ───────
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "silhouettes.npy",    sils_a)
    np.save(out_dir / "keypoints.npy",      kps_norm_a)
    np.save(out_dir / "keypoints_raw.npy",  kps_raw_a)
    np.save(out_dir / "bboxes.npy",         bboxes_a)

    meta = {
        "subject": subject,
        "condition": condition,
        "source_video": str(video_path),
        "fps_source": fps,
        "n_frames_total": n_total,
        "n_frames_with_bbox": int(sum(1 for f in frames_out if f.bbox is not None)),
        "n_frames_useful": int(sils_a.shape[0]),
        "useful_window_idx": [int(start), int(end)],
        "useful_window_idx_kept": [int(used_idx[0]) if used_idx else -1,
                                   int(used_idx[-1]) if used_idx else -1],
        "silhouette_shape": list(sils_a.shape),
        "keypoints_shape": list(kps_norm_a.shape),
        "preprocess": {
            "sil_resize": [SIL_H, SIL_W],
            "pad_ratio": PAD_RATIO,
            "motion_px_thresh": MOTION_PX_THRESH,
            "motion_window": MOTION_WIN,
            "rvm_downsample_ratio": downsample_ratio,
            "rtmpose_model": RTMPOSE_ONNX_NAME,
        },
        "elapsed_s": round(dt, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    # ─────── grid debug 4×4 ───────
    if sils_a.shape[0] >= 16:
        idxs = np.linspace(0, sils_a.shape[0] - 1, 16).astype(int)
        cells = []
        for j in idxs:
            cell = cv2.cvtColor(sils_a[j], cv2.COLOR_GRAY2BGR)
            cells.append(cell)
        rows = []
        for r in range(4):
            rows.append(np.hstack(cells[r * 4:(r + 1) * 4]))
        grid = np.vstack(rows)
        cv2.imwrite(str(out_dir / "debug_silhouettes_grid.png"), grid)

    log.info("→ %s", out_dir)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--out", required=True, help="raíz, p.ej. data/processed")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.subject / args.condition
    process_video(Path(args.video), out_dir, args.subject, args.condition)


if __name__ == "__main__":
    main()
