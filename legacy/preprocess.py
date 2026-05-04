"""
preprocess.py v2 — Extracción de características para Gait Recognition Híbrido
Soporta: MediaPipe Pose (default) | YOLOv8-pose (--detector yolo)
Flujo 1: Keypoints 2D con root-normalization + escala de torso
Flujo 2: Siluetas binarias (MOG2) → GEI (Gait Energy Image)
Salida : archivos .npy por ventana deslizante
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm

# ── Constantes ────────────────────────────────────────────────────────────────
KEYPOINT_DIM  = 33 * 2   # MediaPipe: 33 joints × (x,y)
YOLO_KP_DIM   = 17 * 2   # COCO-17 joints × (x,y)
SEQ_LEN       = 60
GEI_H, GEI_W  = 128, 88
LEFT_HIP_IDX  = 23        # MediaPipe
RIGHT_HIP_IDX = 24
LEFT_SHOULDER = 11


# ── MediaPipe extractor ───────────────────────────────────────────────────────
def _extract_kp_mediapipe(video_path: str) -> np.ndarray | None:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      smooth_landmarks=True, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark
                kp  = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
                root       = (kp[LEFT_HIP_IDX] + kp[RIGHT_HIP_IDX]) / 2.0
                kp        -= root
                torso      = np.linalg.norm(kp[LEFT_SHOULDER] - kp[LEFT_HIP_IDX]) + 1e-6
                kp        /= torso
                frames.append(kp.flatten())
            else:
                frames.append(frames[-1].copy() if frames else np.zeros(KEYPOINT_DIM, dtype=np.float32))
    cap.release()
    return np.stack(frames).astype(np.float32) if frames else None


# ── YOLOv8-pose extractor ─────────────────────────────────────────────────────
def _extract_kp_yolo(video_path: str) -> np.ndarray | None:
    """
    Requiere: pip install ultralytics
    Usa COCO-17 keypoints (índices: 11=left_hip, 12=right_hip, 5=left_shoulder)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Instala ultralytics: pip install ultralytics")

    L_HIP, R_HIP, L_SHLDR = 11, 12, 5
    model = YOLO("yolov8n-pose.pt")
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kp_raw = results[0].keypoints.xy[0].cpu().numpy()  # (17,2)
            # Normalizar al espacio [0,1]
            h, w   = frame.shape[:2]
            kp     = kp_raw / np.array([w, h], dtype=np.float32)
            root   = (kp[L_HIP] + kp[R_HIP]) / 2.0
            kp    -= root
            torso  = np.linalg.norm(kp[L_SHLDR] - kp[L_HIP]) + 1e-6
            kp    /= torso
            frames.append(kp.flatten())
        else:
            frames.append(frames[-1].copy() if frames else np.zeros(YOLO_KP_DIM, dtype=np.float32))
    cap.release()
    return np.stack(frames).astype(np.float32) if frames else None


def extract_skeleton_sequence(video_path: str, detector: str = "mediapipe") -> np.ndarray | None:
    if detector == "yolo":
        return _extract_kp_yolo(video_path)
    return _extract_kp_mediapipe(video_path)


# ── Siluetas → GEI ───────────────────────────────────────────────────────────
def extract_silhouette_sequence(video_path: str) -> np.ndarray | None:
    cap    = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sils   = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask  = bg_sub.apply(gray)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask  = cv2.resize(mask, (GEI_W, GEI_H))
        _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        sils.append(bw)
    cap.release()
    return np.stack(sils).astype(np.uint8) if sils else None


def compute_gei(sil_seq: np.ndarray) -> np.ndarray:
    return (sil_seq.astype(np.float32).mean(axis=0) / 255.0).astype(np.float32)


def sliding_windows(seq: np.ndarray, win_len: int, stride: int):
    T = seq.shape[0]
    wins = [seq[s:s+win_len] for s in range(0, T - win_len + 1, stride)]
    if not wins:
        pad  = np.zeros((win_len - T, seq.shape[1]), dtype=np.float32)
        wins = [np.concatenate([seq, pad], axis=0)]
    return wins


# ── Pipeline principal ────────────────────────────────────────────────────────
def process_dataset(data_root: str, out_dir: str, stride: int = 30, detector: str = "mediapipe"):
    """
    Estructura del dataset:
        data_root/
          subject_01/ normal.mp4  fast.mp4
          subject_02/ ...
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    subjects  = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    label_map = {}

    for idx, subject in enumerate(tqdm(subjects, desc="Subjects")):
        subj_path  = os.path.join(data_root, subject)
        label_map[idx] = subject
        videos     = [f for f in os.listdir(subj_path) if f.lower().endswith((".mp4",".avi",".mov"))]

        for vid in videos:
            vpath    = os.path.join(subj_path, vid)
            vstem    = Path(vid).stem
            kp_seq   = extract_skeleton_sequence(vpath, detector)
            sil_seq  = extract_silhouette_sequence(vpath)
            if kp_seq is None or sil_seq is None:
                print(f"  [SKIP] {subject}/{vid}")
                continue
            gei     = compute_gei(sil_seq)
            windows = sliding_windows(kp_seq, SEQ_LEN, stride)

            for wi, win in enumerate(windows):
                base = f"{subject}_{vstem}_win{wi}"
                np.save(f"{out_dir}/{base}_kp.npy",    win)
                np.save(f"{out_dir}/{base}_gei.npy",   gei)
                np.save(f"{out_dir}/{base}_label.npy", np.array(idx, dtype=np.int64))

    np.save(f"{out_dir}/label_map.npy", label_map)
    print(f"\n✓ Procesamiento completo → {out_dir}  ({len(label_map)} sujetos)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     required=True,  help="Carpeta raíz del dataset")
    parser.add_argument("--out",      default="features")
    parser.add_argument("--stride",   type=int, default=30)
    parser.add_argument("--detector", choices=["mediapipe","yolo"], default="mediapipe")
    args = parser.parse_args()
    process_dataset(args.data, args.out, args.stride, args.detector)
