"""
Fase 2 paso 4b — Empaquetado al formato OpenGait.

Lee los .npy bajo data/processed/<subject>/<condition>/silhouettes.npy
y produce data/pkl/<subject>/<condition>/090/seq00.pkl con un único
np.ndarray (T, 64, 44) uint8 — compatible con el DataLoader de OpenGait
para CASIA-B.

(El ángulo es 090 fijo: única vista lateral del dataset.)

Uso:
    python scripts/pack_to_pkl.py \
        --processed "C:/Proyecto3/ProyectoChino/data/processed" \
        --out       "C:/Proyecto3/ProyectoChino/data/pkl"
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pack")

VIEW = "090"
SEQ = "seq00"
MIN_FRAMES = 30  # OpenGait suele requerir ≥30 frames por secuencia


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src_root = Path(args.processed)
    dst_root = Path(args.out)
    dst_root.mkdir(parents=True, exist_ok=True)

    sil_files = sorted(src_root.glob("*/*/silhouettes.npy"))
    log.info("Secuencias encontradas: %d", len(sil_files))

    n_ok = 0
    n_short = 0
    n_total_frames = 0
    short_list = []

    for sil_path in sil_files:
        condition = sil_path.parent.name
        subject = sil_path.parent.parent.name
        sil = np.load(sil_path)  # (T, 64, 44) uint8

        if sil.ndim != 3 or sil.shape[1:] != (64, 44):
            log.error("Shape inesperado en %s: %s", sil_path, sil.shape)
            continue

        if sil.shape[0] < MIN_FRAMES:
            short_list.append((subject, condition, sil.shape[0]))
            n_short += 1
            log.warning("Secuencia corta: %s/%s = %d frames (<%d) — se empaqueta igual",
                        subject, condition, sil.shape[0], MIN_FRAMES)

        out_dir = dst_root / subject / condition / VIEW
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{SEQ}.pkl"
        with out_path.open("wb") as fh:
            pickle.dump(sil, fh, protocol=pickle.HIGHEST_PROTOCOL)
        n_ok += 1
        n_total_frames += sil.shape[0]

    log.info("=" * 60)
    log.info("OK: %d secuencias  |  cortas (<%d): %d  |  total frames: %d",
             n_ok, MIN_FRAMES, n_short, n_total_frames)
    if short_list:
        log.info("Cortas:")
        for s, c, n in short_list:
            log.info("  %s/%s = %d", s, c, n)
    log.info("→ %s", dst_root)


if __name__ == "__main__":
    main()
