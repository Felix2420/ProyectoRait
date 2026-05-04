"""
Fase 2 paso 4a — Extracción batch sobre los 38 videos del Dataset Crudo.

Carga modelos UNA vez y procesa todos los `<sujeto>/{normal,rapido}_*.mp4`.
Idempotente: si ya existe `meta.json` para el (sujeto, condición), salta —
salvo que se pase --force.

Uso:
    python scripts/extract_all.py \
        --raw "C:/Proyecto3/Proyecto-Reconocedor-IA/Dataset Crudo" \
        --out "C:/Proyecto3/ProyectoChino/data/processed"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

# Permitir `from src.preprocess...` cuando se invoca como script (no -m)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess._cuda_dlls import enable_torch_cuda_dlls
enable_torch_cuda_dlls()

from src.preprocess.extract_sequence import load_models, process_video  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extract_all")

COND_RE = re.compile(r"^(normal|rapido)_\d+\.mp4$", re.IGNORECASE)


def discover(raw_root: Path) -> list[tuple[str, str, Path]]:
    items = []
    for sub_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for vid in sorted(sub_dir.glob("*.mp4")):
            m = COND_RE.match(vid.name)
            if not m:
                continue
            items.append((sub_dir.name, m.group(1).lower(), vid))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true",
                    help="reprocesar aunque exista meta.json")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    items = discover(raw_root)
    log.info("Videos a procesar: %d", len(items))

    log.info("Cargando modelos (1 sola vez)…")
    models = load_models()

    summary = []
    t_global = time.time()
    for i, (subject, condition, video) in enumerate(items, 1):
        out_dir = out_root / subject / condition
        meta_path = out_dir / "meta.json"
        if meta_path.exists() and not args.force:
            log.info("[%d/%d] %s/%s → SKIP (ya procesado)", i, len(items), subject, condition)
            with meta_path.open(encoding="utf-8") as fh:
                summary.append(json.load(fh))
            continue
        log.info("[%d/%d] %s/%s", i, len(items), subject, condition)
        try:
            meta = process_video(video, out_dir, subject, condition, models=models)
            summary.append(meta)
        except Exception:
            log.exception("Error procesando %s", video)

    dt = time.time() - t_global
    log.info("=" * 60)
    log.info("Total: %d videos en %.1fs (%.2fs por video promedio)",
             len(summary), dt, dt / max(1, len(summary)))

    # Resumen agregado
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    log.info("Resumen agregado → %s", summary_path)

    # Estadísticas rápidas
    if summary:
        useful = [s["n_frames_useful"] for s in summary]
        log.info("Frames útiles: min=%d  max=%d  mean=%.1f  total=%d",
                 min(useful), max(useful), sum(useful) / len(useful), sum(useful))


if __name__ == "__main__":
    main()
