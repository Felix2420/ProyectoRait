"""
Fase 4, Paso B — Empaquetar keypoints_raw.npy en pkls para heatmap generation.

Lee data/processed/<subject>/<cond>/keypoints_raw.npy  (sesión 1)
  y data/processed_s2/<subject>/<cond>/keypoints_raw.npy (sesión 2)
y escribe:
  data/poses_raw/<subject>/<cond>_s1/090/seq00.pkl  (T, 17, 3) float32
  data/poses_raw/<subject>/<cond>_s2/090/seq00.pkl  (T, 17, 3) float32

Formato: coordenadas pixel del frame original (x, y, confidence).
El CenterAndScaleNormalizer de gen_heatmaps.py las convierte al espacio del heatmap.

Uso:
    python scripts/pack_pose_to_pkl.py
    python scripts/pack_pose_to_pkl.py --force   # reprocesar aunque existan
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pack_pose")

REPO = Path(__file__).resolve().parents[1]

VIEW = "090"
SEQ = "seq00"
MIN_FRAMES = 30


class Source(NamedTuple):
    processed_root: Path
    session_suffix: str   # "_s1" o "_s2"


SOURCES: list[Source] = [
    Source(REPO / "data" / "processed",    "_s1"),
    Source(REPO / "data" / "processed_s2", "_s2"),
]
POSES_ROOT = REPO / "data" / "poses_raw"


def pack_source(src: Source, force: bool) -> dict:
    processed_root = src.processed_root
    suffix = src.session_suffix

    if not processed_root.exists():
        log.warning("No existe %s — saltar sesión", processed_root)
        return {"skipped": 0, "ok": 0, "missing": 0}

    kp_files = sorted(processed_root.glob("*/*/keypoints_raw.npy"))
    if not kp_files:
        log.warning("No hay keypoints_raw.npy en %s", processed_root)
        return {"skipped": 0, "ok": 0, "missing": 0}

    log.info("Procesando %s: %d secuencias", src.session_suffix, len(kp_files))

    n_ok = n_skip = n_missing = 0
    stats: list[dict] = []

    for kp_path in kp_files:
        cond = kp_path.parent.name      # normal | rapido
        subject = kp_path.parent.parent.name

        out_dir = POSES_ROOT / subject / f"{cond}{suffix}" / VIEW
        out_path = out_dir / f"{SEQ}.pkl"

        if out_path.exists() and not force:
            log.debug("[skip] %s/%s%s", subject, cond, suffix)
            n_skip += 1
            continue

        kp = np.load(kp_path)  # (T, 17, 3)
        if kp.ndim != 3 or kp.shape[1:] != (17, 3):
            log.error("Shape inesperado en %s: %s", kp_path, kp.shape)
            n_missing += 1
            continue

        if kp.shape[0] < MIN_FRAMES:
            log.warning("Secuencia corta: %s/%s%s = %d frames", subject, cond, suffix, kp.shape[0])

        out_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            pickle.dump(kp, fh, protocol=pickle.HIGHEST_PROTOCOL)

        conf_mean = float(kp[:, :, 2].mean())
        conf_low = float((kp[:, :, 2] < 0.3).mean())
        n_ok += 1
        stats.append({
            "subject": subject, "cond": f"{cond}{suffix}",
            "T": kp.shape[0], "conf_mean": round(conf_mean, 3),
            "frac_low_conf": round(conf_low, 3),
        })

    if stats:
        conf_means = [s["conf_mean"] for s in stats]
        low_confs = [s["frac_low_conf"] for s in stats]
        log.info(
            "[%s] OK=%d skip=%d err=%d | conf_mean=%.3f | frac_conf<0.3=%.3f",
            suffix, n_ok, n_skip, n_missing,
            np.mean(conf_means), np.mean(low_confs),
        )
        # Alertar si hay secuencias con mucho porcentaje de baja confianza
        bad = [s for s in stats if s["frac_low_conf"] > 0.2]
        if bad:
            log.warning("Secuencias con >20%% frames de baja confianza:")
            for s in bad:
                log.warning("  %s/%s  T=%d  low_conf=%.1f%%",
                            s["subject"], s["cond"], s["T"], s["frac_low_conf"] * 100)

    return {"ok": n_ok, "skipped": n_skip, "missing": n_missing, "detail": stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reprocesar aunque existan pkls")
    args = ap.parse_args()

    POSES_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("Salida → %s", POSES_ROOT)

    totals = {"ok": 0, "skipped": 0, "missing": 0}
    for src in SOURCES:
        r = pack_source(src, args.force)
        for k in ("ok", "skipped", "missing"):
            totals[k] += r.get(k, 0)

    log.info("=" * 60)
    log.info("TOTAL  ok=%d  skip=%d  err=%d", totals["ok"], totals["skipped"], totals["missing"])
    log.info("Siguiente: python scripts/gen_heatmaps.py")


if __name__ == "__main__":
    main()
