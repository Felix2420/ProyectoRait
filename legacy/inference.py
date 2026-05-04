"""
inference.py v2 — Inferencia en tiempo real (webcam)
Mejoras respecto a v1:
  • Galería cargada desde gallery_embeddings.npy (no recalcula)
  • Umbral calibrado automáticamente cargado del checkpoint
  • Barra de confianza + historia de predicciones (votación por mayoría)
  • Soporte MediaPipe o YOLOv8-pose (--detector)
  • Teclas: q=salir  r=reset  s=guardar frame de la persona identificada
"""

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from pathlib import Path
from collections import deque, Counter
import argparse
import time
import mediapipe as mp

from model import HybridGaitNet, GalleryManager, EMBED_DIM, SEQ_LEN
from preprocess import (LEFT_HIP_IDX, RIGHT_HIP_IDX, LEFT_SHOULDER,
                        compute_gei, GEI_H, GEI_W)

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

COLOR_KNOWN   = (0, 210, 90)
COLOR_UNKNOWN = (30, 60, 220)
COLOR_INFO    = (210, 210, 210)
FONT          = cv2.FONT_HERSHEY_DUPLEX
VOTE_WINDOW   = 5   # últimas N predicciones para votación por mayoría


# ── Carga del modelo ──────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device):
    ckpt      = torch.load(ckpt_path, map_location=device)
    kp_dim    = ckpt.get("kp_dim", 66)
    label_map = ckpt["label_map"]
    threshold = ckpt.get("threshold", 1.0)
    model     = HybridGaitNet(kp_input_dim=kp_dim, embed_dim=EMBED_DIM).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Modelo cargado  | Val Acc: {ckpt.get('val_acc',0)*100:.1f}%")
    print(f"[INFO] Threshold base  : {threshold:.4f}")
    return model, label_map, threshold


# ── Extracción keypoints (MediaPipe) ──────────────────────────────────────────
def get_kp_mediapipe(frame, pose):
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return None, None
    lms  = result.pose_landmarks.landmark
    kp   = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
    root = (kp[LEFT_HIP_IDX] + kp[RIGHT_HIP_IDX]) / 2.0
    kp  -= root
    kp  /= (np.linalg.norm(kp[LEFT_SHOULDER] - kp[LEFT_HIP_IDX]) + 1e-6)
    return kp.flatten(), result.pose_landmarks


# ── Extracción keypoints (YOLO) ────────────────────────────────────────────────
def get_kp_yolo(frame, yolo_model):
    L_HIP, R_HIP, L_SHLDR = 11, 12, 5
    results = yolo_model(frame, verbose=False)
    if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        h, w   = frame.shape[:2]
        kp_raw = results[0].keypoints.xy[0].cpu().numpy()
        kp     = kp_raw / np.array([w, h], dtype=np.float32)
        root   = (kp[L_HIP] + kp[R_HIP]) / 2.0
        kp    -= root
        kp    /= (np.linalg.norm(kp[L_SHLDR] - kp[L_HIP]) + 1e-6)
        return kp.flatten(), None
    return None, None


# ── Silueta ───────────────────────────────────────────────────────────────────
def get_silhouette(frame, bg_sub):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask   = bg_sub.apply(gray)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.resize(mask, (GEI_W, GEI_H))
    _, bw  = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return bw


# ── Inferencia sobre buffer ───────────────────────────────────────────────────
@torch.no_grad()
def infer(kp_buf, sil_buf, model, gm: GalleryManager, device):
    T  = SEQ_LEN
    if len(kp_buf) < T:
        pad    = T - len(kp_buf)
        kp_seq = np.stack(kp_buf + [kp_buf[-1]] * pad, axis=0)
        si_seq = np.stack(sil_buf + [sil_buf[-1]] * pad, axis=0)
    else:
        kp_seq = np.stack(kp_buf[-T:], axis=0)
        si_seq = np.stack(sil_buf[-T:], axis=0)

    gei   = compute_gei(si_seq)
    kp_t  = torch.from_numpy(kp_seq).float().unsqueeze(0).to(device)
    gei_t = torch.from_numpy(gei).float().unsqueeze(0).unsqueeze(0).to(device)

    with autocast(enabled=(device.type == "cuda")):
        emb = model(kp_t, gei_t).squeeze(0).float().cpu().numpy()

    return gm.match(emb, gm._threshold)


# ── HUD overlay ───────────────────────────────────────────────────────────────
def draw_hud(frame, identity, dist, threshold, fps, n_frames, vote_hist):
    h, w    = frame.shape[:2]
    known   = identity != "DESCONOCIDO"
    color   = COLOR_KNOWN if known else COLOR_UNKNOWN
    conf    = max(0.0, min(1.0, 1.0 - dist / (threshold * 1.8)))

    # Fondo semitransparente
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 100), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

    # ID principal
    cv2.putText(frame, f"ID: {identity}", (14, 40), FONT, 1.0, color, 2, cv2.LINE_AA)

    # Distancia y umbral
    cv2.putText(frame, f"dist={dist:.3f}  thr={threshold:.3f}",
                (14, 72), FONT, 0.55, COLOR_INFO, 1, cv2.LINE_AA)

    # Barra de confianza
    bw = int(conf * 180)
    cv2.rectangle(frame, (w-200, 12), (w-20, 38), (55, 55, 55), -1)
    cv2.rectangle(frame, (w-200, 12), (w-200+bw, 38), color, -1)
    cv2.putText(frame, f"{conf*100:.0f}%", (w-195, 58), FONT, 0.5, COLOR_INFO, 1, cv2.LINE_AA)

    # Historial de votos
    hist_str = " | ".join(str(v) for v in list(vote_hist)[-VOTE_WINDOW:])
    cv2.putText(frame, f"Votos: [{hist_str}]", (14, h-36), FONT, 0.45, COLOR_INFO, 1, cv2.LINE_AA)

    # FPS y frames
    cv2.putText(frame, f"FPS={fps:.1f}  buf={n_frames}/{SEQ_LEN}",
                (14, h-12), FONT, 0.45, COLOR_INFO, 1, cv2.LINE_AA)
    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_map, threshold = load_model(args.ckpt, device)

    # Galería
    gm = GalleryManager(args.gallery)
    gm.load()
    gm._threshold = args.threshold if args.threshold else threshold
    if args.recalibrate:
        gm._threshold = gm.calibrate_threshold(args.threshold_percentile)
    print(f"[INFO] Umbral activo   : {gm._threshold:.4f}")
    print(f"[INFO] Sujetos galería : {[label_map.get(k,k) for k in gm.gallery]}")

    # Detector
    yolo_model = None
    if args.detector == "yolo":
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n-pose.pt")
        print("[INFO] Detector: YOLOv8-pose")
    else:
        print("[INFO] Detector: MediaPipe Pose")

    # Webcam
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    bg_sub    = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
    kp_buf    = deque(maxlen=SEQ_LEN)
    sil_buf   = deque(maxlen=SEQ_LEN)
    vote_hist = deque(maxlen=VOTE_WINDOW)

    identity  = "Acumulando..."
    dist      = float("inf")
    fps       = 0.0
    t0        = time.time()
    step      = max(1, SEQ_LEN // 4)
    frame_cnt = 0
    save_dir  = Path("captures"); save_dir.mkdir(exist_ok=True)

    print("\n[INFO] Iniciado. Teclas: q=salir  r=reset  s=guardar frame")

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      smooth_landmarks=True, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose_ctx:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.detector == "yolo":
                kp_vec, landmarks = get_kp_yolo(frame, yolo_model)
            else:
                kp_vec, landmarks = get_kp_mediapipe(frame, pose_ctx)

            sil = get_silhouette(frame, bg_sub)

            if kp_vec is not None:
                kp_buf.append(kp_vec)
                sil_buf.append(sil)
                frame_cnt += 1

            min_buf = max(10, SEQ_LEN // 3)
            if len(kp_buf) >= min_buf and frame_cnt % step == 0:
                lbl, dist = infer(list(kp_buf), list(sil_buf), model, gm, device)
                name      = label_map.get(lbl, "DESCONOCIDO") if lbl != -1 else "DESCONOCIDO"

                # Votación por mayoría
                vote_hist.append(name)
                most_common = Counter(vote_hist).most_common(1)[0][0]
                identity    = most_common

            # Dibujar esqueleto MediaPipe
            if landmarks and args.detector == "mediapipe":
                mp_drawing.draw_landmarks(
                    frame, landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,170,255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,255,180), thickness=2)
                )

            fps = 1.0 / max(time.time() - t0, 1e-6)
            t0  = time.time()
            frame = draw_hud(frame, identity, dist, gm._threshold,
                             fps, len(kp_buf), vote_hist)
            cv2.imshow("Gait Recognition v2", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                kp_buf.clear(); sil_buf.clear(); vote_hist.clear()
                identity = "Acumulando..."; dist = float("inf"); frame_cnt = 0
            elif key == ord("s"):
                fname = save_dir / f"{identity}_{int(time.time())}.jpg"
                cv2.imwrite(str(fname), frame)
                print(f"[INFO] Frame guardado → {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Inferencia finalizada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia v2 — Gait Recognition")
    parser.add_argument("--ckpt",                 required=True,
                        help="Ruta al best_model.pth")
    parser.add_argument("--gallery",              default="checkpoints/gallery_embeddings.npy",
                        help="Ruta a gallery_embeddings.npy")
    parser.add_argument("--cam",                  type=int,   default=0)
    parser.add_argument("--detector",             choices=["mediapipe","yolo"], default="mediapipe")
    parser.add_argument("--threshold",            type=float, default=None,
                        help="Sobrescribir umbral (None = usa el del checkpoint)")
    parser.add_argument("--recalibrate",          action="store_true",
                        help="Recalibrar umbral con los datos de la galería")
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    args = parser.parse_args()
    run(args)
