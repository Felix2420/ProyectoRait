"""
Fase 4, Paso D — Construir data/pkl_multimodal/ (heatmap + silueta por secuencia).

Para cada secuencia produce DOS pkls en el mismo directorio:
  data/pkl_multimodal/<subject>/<cond>_<session>/090/
    0_heatmap.pkl   (T, 2, 64, 64) uint8  — canal 0: limb, canal 1: joint
    1_sil.pkl       (T, 64, 44)    uint8  — silueta binaria

El DataLoader de OpenGait los lee en orden alfabético → data_list[0]=heatmap, data_list[1]=sil.

Fuentes:
  heatmaps:   data/heatmaps/<subject>/<cond>_<session>/090/seq00.pkl
  siluetas:   data/processed/<subject>/<cond>/silhouettes.npy  (s1)
              data/processed_s2/<subject>/<cond>/silhouettes.npy (s2)

Las siluetas se toman de los .npy de la extracción ACTUAL (no de los pkls antiguos)
para garantizar T_sil == T_heatmap.

Uso:
    python scripts/build_multimodal_dataset.py
    python scripts/build_multimodal_dataset.py --force
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_multimodal")

REPO = Path(__file__).resolve().parents[1]

HEATMAPS_ROOT = REPO / "data" / "heatmaps"
PROCESSED_S1 = REPO / "data" / "processed"
PROCESSED_S2 = REPO / "data" / "processed_s2"
MULTIMODAL_ROOT = REPO / "data" / "pkl_multimodal"

VIEW = "090"
SEQ = "seq00"


def sil_path(subject: str, cond: str, session: str) -> Path | None:
    """Busca silhouettes.npy en el directorio de extracción correcto."""
    if session == "s1":
        p = PROCESSED_S1 / subject / cond / "silhouettes.npy"
    else:
        p = PROCESSED_S2 / subject / cond / "silhouettes.npy"
    return p if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # Descubrir todos los heatmaps disponibles
    heatmap_pkls = sorted(HEATMAPS_ROOT.glob(f"*/*/{VIEW}/{SEQ}.pkl"))
    if not heatmap_pkls:
        log.error("No hay heatmaps en %s — corre gen_heatmaps.py primero", HEATMAPS_ROOT)
        return

    log.info("Heatmaps encontrados: %d", len(heatmap_pkls))
    MULTIMODAL_ROOT.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_err = n_tmismatch = 0

    for hm_path in heatmap_pkls:
        # hm_path = data/heatmaps/<subject>/<cond>_<session>/090/seq00.pkl
        cond_session = hm_path.parent.parent.name   # ej: normal_s1
        subject = hm_path.parent.parent.parent.name

        out_dir = MULTIMODAL_ROOT / subject / cond_session / VIEW
        out_hm = out_dir / "0_heatmap.pkl"
        out_sil = out_dir / "1_sil.pkl"

        if out_hm.exists() and out_sil.exists() and not args.force:
            log.debug("[skip] %s/%s", subject, cond_session)
            n_skip += 1
            continue

        # Parsear cond y session del nombre de directorio
        if "_s1" in cond_session:
            cond = cond_session.replace("_s1", "")
            session = "s1"
        elif "_s2" in cond_session:
            cond = cond_session.replace("_s2", "")
            session = "s2"
        else:
            log.error("Nombre inesperado: %s", cond_session)
            n_err += 1
            continue

        sp = sil_path(subject, cond, session)
        if sp is None:
            log.warning("[falta sil] %s/%s — silhouettes.npy no encontrado", subject, cond_session)
            n_err += 1
            continue

        try:
            with hm_path.open("rb") as f:
                heatmap = pickle.load(f)   # (T, 2, 64, 64)

            sil = np.load(sp)              # (T, 64, 44)

            T_hm = heatmap.shape[0]
            T_sil = sil.shape[0]

            if T_hm != T_sil:
                log.error(
                    "[T-mismatch] %s/%s  T_heatmap=%d != T_sil=%d — SKIP",
                    subject, cond_session, T_hm, T_sil,
                )
                n_tmismatch += 1
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            with out_hm.open("wb") as f:
                pickle.dump(heatmap, f, protocol=pickle.HIGHEST_PROTOCOL)
            with out_sil.open("wb") as f:
                pickle.dump(sil, f, protocol=pickle.HIGHEST_PROTOCOL)

            log.info("[ok] %s/%s  T=%d", subject, cond_session, T_hm)
            n_ok += 1

        except Exception:
            log.exception("[err] %s/%s", subject, cond_session)
            n_err += 1

    log.info("=" * 60)
    log.info("OK=%d  skip=%d  T-mismatch=%d  err=%d", n_ok, n_skip, n_tmismatch, n_err)

    if n_tmismatch > 0:
        log.warning(
            "%d secuencias con T-mismatch. Esto indica que la extracción actual "
            "y los heatmaps no coinciden. Revisa si gen_heatmaps.py usó los datos "
            "correctos.", n_tmismatch,
        )

    total = n_ok + n_skip
    log.info("Total secuencias en pkl_multimodal: %d", total)
    log.info("Siguiente: python scripts/smoke_test_skeletongaitpp.py")


if __name__ == "__main__":
    main()
