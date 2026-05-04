"""
Fase 4, Paso C — Generar heatmaps de pose (bone + joint) para SkeletonGait++.

Lee  data/poses_raw/<subject>/<cond>_<session>/090/seq00.pkl  (T, 17, 3) float32
y escribe:
  data/heatmaps/<subject>/<cond>_<session>/090/seq00.pkl  (T, 2, 64, 64) uint8

Canal 0 = limb/bone heatmap (conexiones del esqueleto COCO-17).
Canal 1 = joint heatmap (puntos clave individuales).

Pipeline por secuencia:
  1. CenterAndScaleNormalizer  → keypoints en espacio de píxel del heatmap (64×64)
  2. GeneratePoseTarget(limb)  → (T, 17, 64, 64) gaussianas por conexión
  3. HeatmapToImage            → (T, 1, 64, 64) imagen grayscale (colormap max)
  4. HeatmapAlignment          → (T, 1, 64, 64) crop vertical centrado
  5. idem para joints
  6. concatenar → (T, 2, 64, 64) uint8

Uso:
    python scripts/gen_heatmaps.py
    python scripts/gen_heatmaps.py --force   # reprocesar aunque existan
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gen_heatmaps")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "OpenGait"))

# Importar clases de OpenGait (pretreatment_heatmap.py)
from datasets.pretreatment_heatmap import (
    CenterAndScaleNormalizer,
    GeneratePoseTarget,
    HeatmapToImage,
    HeatmapAlignment,
    PadKeypoints,
)

POSES_ROOT = REPO / "data" / "poses_raw"
HEATMAPS_ROOT = REPO / "data" / "heatmaps"

# Parámetros del heatmap (deben coincidir con lo que espera SkeletonGait++)
IMG_SIZE = 64        # cuadrado; se recortará a 64×44 en inputs_pretreament
SIGMA = 0.6          # ancho de gaussiana (estándar SkeletonGait)
HEATMAP_HEIGHT = 64  # altura del espacio de normalización

# Esqueleto COCO-17 (mismas conexiones que GeneratePoseTarget default)
COCO17_SKELETONS = (
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
    (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
    (13, 15), (6, 12), (12, 14), (14, 16), (11, 12),
)

VIEW = "090"
SEQ = "seq00"


def build_transform():
    """Devuelve la función que convierte (T, 17, 3) → (T, 2, 64, 64) uint8."""
    normalizer = CenterAndScaleNormalizer(
        pose_format="coco",
        use_conf=True,
        heatmap_image_height=HEATMAP_HEIGHT,
    )
    padder = PadKeypoints(pad_method="knn", use_conf=True)

    gen_limb = GeneratePoseTarget(
        sigma=SIGMA, use_score=True,
        with_kp=False, with_limb=True,
        skeletons=COCO17_SKELETONS,
        img_h=IMG_SIZE, img_w=IMG_SIZE,
    )
    gen_joint = GeneratePoseTarget(
        sigma=SIGMA, use_score=True,
        with_kp=True, with_limb=False,
        skeletons=COCO17_SKELETONS,
        img_h=IMG_SIZE, img_w=IMG_SIZE,
    )

    to_image = HeatmapToImage()
    aligner = HeatmapAlignment(
        align=True,
        final_img_size=IMG_SIZE,
        offset=0,
        heatmap_image_size=IMG_SIZE,
    )

    def transform(kps: np.ndarray) -> np.ndarray:
        """kps: (T, 17, 3) float32 pixel coords → (T, 2, 64, 64) uint8."""
        # Rellenar keypoints con baja confianza (KNN imputer)
        kps = padder(kps)
        # Centrar en cadera y escalar al espacio del heatmap
        kps = normalizer(kps)
        # Generar heatmaps por canal
        hm_limb = gen_limb(kps)      # (T, 17, 64, 64)
        hm_joint = gen_joint(kps)    # (T, 17, 64, 64)
        # Colapsar por max + colormap → imagen grayscale
        img_limb = to_image(hm_limb)   # (T, 1, 64, 64) uint8
        img_joint = to_image(hm_joint)  # (T, 1, 64, 64) uint8
        # Alinear verticalmente (crop centrado en persona)
        img_limb = aligner(img_limb)   # (T, 1, 64, 64) uint8
        img_joint = aligner(img_joint)  # (T, 1, 64, 64) uint8
        # Concatenar: canal 0 = limb, canal 1 = joint
        return np.concatenate([img_limb, img_joint], axis=1)  # (T, 2, 64, 64)

    return transform


def process_all(force: bool) -> None:
    pose_pkls = sorted(POSES_ROOT.glob(f"*/*/{VIEW}/{SEQ}.pkl"))
    if not pose_pkls:
        log.error("No hay pkls en %s — corre pack_pose_to_pkl.py primero", POSES_ROOT)
        return

    log.info("Secuencias de pose: %d", len(pose_pkls))
    transform = build_transform()

    n_ok = n_skip = n_err = 0
    all_conf: list[float] = []
    all_T: list[int] = []

    for pose_path in pose_pkls:
        # pose_path = data/poses_raw/<subject>/<cond_session>/090/seq00.pkl
        cond_session = pose_path.parent.parent.name   # ej: normal_s1
        subject = pose_path.parent.parent.parent.name

        out_dir = HEATMAPS_ROOT / subject / cond_session / VIEW
        out_path = out_dir / f"{SEQ}.pkl"

        if out_path.exists() and not force:
            log.debug("[skip] %s/%s", subject, cond_session)
            n_skip += 1
            continue

        try:
            with pose_path.open("rb") as f:
                kps = pickle.load(f)   # (T, 17, 3)

            T = kps.shape[0]
            conf_mean = float(kps[:, :, 2].mean())
            all_conf.append(conf_mean)
            all_T.append(T)

            heatmap = transform(kps)   # (T, 2, 64, 64) uint8

            out_dir.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                pickle.dump(heatmap, f, protocol=pickle.HIGHEST_PROTOCOL)

            log.info("[ok] %s/%s  T=%d  conf=%.3f  shape=%s",
                     subject, cond_session, T, conf_mean, heatmap.shape)
            n_ok += 1

        except Exception:
            log.exception("[err] %s/%s", subject, cond_session)
            n_err += 1

    log.info("=" * 60)
    log.info("OK=%d  skip=%d  err=%d", n_ok, n_skip, n_err)
    if all_conf:
        log.info("conf_mean_global=%.3f  T_mean=%.0f  T_min=%d  T_max=%d",
                 np.mean(all_conf), np.mean(all_T), min(all_T), max(all_T))
    log.info("Siguiente: python scripts/build_multimodal_dataset.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    HEATMAPS_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("Salida → %s", HEATMAPS_ROOT)
    process_all(args.force)


if __name__ == "__main__":
    main()
