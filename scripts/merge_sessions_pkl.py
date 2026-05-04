"""Fusiona data/pkl (sesión 1) y data/pkl_s2 (sesión 2) en data/pkl_multisession,
usando el campo `type` de OpenGait para distinguir sesión.

Estructura resultante:
    data/pkl_multisession/<subject>/<cond>_<sesion>/090/seq00.pkl

Donde <cond>_<sesion> ∈ {normal_s1, normal_s2, rapido_s1, rapido_s2}.
OpenGait DataSet trata cada (subject, type, view) como secuencia independiente,
lo que permite que TripletSampler pueda muestrear mezclas cross-session dentro
del mismo batch (triplets cross-clothing).

El label sigue siendo <subject>, por lo que identidad es consistente entre
sesiones.

Uso:
    python scripts/merge_sessions_pkl.py \\
        --pkl-s1 data/pkl \\
        --pkl-s2 data/pkl_s2 \\
        --out data/pkl_multisession
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("merge")


def copy_session(src_root: Path, dst_root: Path, session_tag: str) -> tuple[int, int]:
    """Copia src_root/<subject>/<cond>/090/seq00.pkl →
    dst_root/<subject>/<cond>_<session_tag>/090/seq00.pkl.

    Devuelve (n_copiados, n_saltados_missing).
    """
    n_ok, n_missing = 0, 0
    if not src_root.exists():
        raise FileNotFoundError(src_root)

    for subj_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for cond_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir()):
            view_dir = cond_dir / "090"
            if not view_dir.is_dir():
                continue
            pkls = sorted(view_dir.glob("*.pkl"))
            if not pkls:
                n_missing += 1
                log.warning("sin .pkl en %s", view_dir)
                continue
            if len(pkls) > 1:
                log.warning("múltiples .pkl en %s, solo se copia el primero", view_dir)
            src_pkl = pkls[0]
            new_cond = f"{cond_dir.name}_{session_tag}"
            dst = dst_root / subj_dir.name / new_cond / "090" / "seq00.pkl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_pkl, dst)
            n_ok += 1
    return n_ok, n_missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-s1", default="data/pkl")
    ap.add_argument("--pkl-s2", default="data/pkl_s2")
    ap.add_argument("--out", default="data/pkl_multisession")
    ap.add_argument("--force", action="store_true", help="borra --out si existe")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        if args.force:
            log.info("borrando %s existente (--force)", out)
            shutil.rmtree(out)
        else:
            log.warning("%s ya existe (usa --force para sobreescribir)", out)
    out.mkdir(parents=True, exist_ok=True)

    n1_ok, n1_miss = copy_session(Path(args.pkl_s1), out, "s1")
    log.info("sesión 1: %d copiadas, %d faltantes", n1_ok, n1_miss)

    n2_ok, n2_miss = copy_session(Path(args.pkl_s2), out, "s2")
    log.info("sesión 2: %d copiadas, %d faltantes", n2_ok, n2_miss)

    # Inventario final por sujeto
    log.info("=" * 60)
    log.info("Inventario final por sujeto (se espera 4 conds por sujeto):")
    for subj_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        conds = sorted(p.name for p in subj_dir.iterdir() if p.is_dir())
        marker = "" if len(conds) == 4 else f"  <-- INCOMPLETO ({len(conds)}/4)"
        log.info("  %-30s  %s%s", subj_dir.name, ", ".join(conds), marker)

    total = n1_ok + n2_ok
    log.info("=" * 60)
    log.info("Total secuencias en %s: %d", out, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
