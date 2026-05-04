"""Empaqueta el proyecto en zips separados para transferir a otra PC.

Genera 4 zips en --out-dir (por defecto C:/Proyecto3/export/):
    1. ProyectoChino_code.zip        (~40 MB)  - codigo, configs, reports, docs, src, third_party, scripts, yolo11n.pt, requirements.lock.txt, CLAUDE.md, README.md, SETUP_NEW_PC.md
    2. ProyectoChino_checkpoints.zip (~175 MB) - pretrained Gait3D + finetune multisession (solo los que importan)
    3. ProyectoChino_pkls.zip        (~45 MB)  - pkls ya procesados (atajo para no regenerar)
    4. ProyectoChino_data_raw.zip    (~200 MB) - videos crudos s1 (fuera del repo) + s2 (dataset2/)

Ajustes:
    --skip-raw        no empaqueta data_raw (los videos pueden ir por separado en USB)
    --skip-pkls       no empaqueta pkls (si se van a regenerar en la PC nueva)
    --s1-raw PATH     ruta al Dataset Crudo de sesion 1 (default:
                      C:/Proyecto3/Proyecto-Reconocedor-IA/Dataset Crudo)

Uso:
    python scripts/export_project.py
    python scripts/export_project.py --out-dir D:/export --skip-raw
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
DEFAULT_S1_RAW = Path("C:/Proyecto3/Proyecto-Reconocedor-IA/Dataset Crudo")

EXCLUDE_DIRS = {
    "__pycache__", ".git", ".vscode", ".idea", "venv", ".venv",
    "node_modules", ".ipynb_checkpoints", ".pytest_cache",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".DS_Store"}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("export")


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if should_skip(p):
                continue
            yield p


def make_zip(zip_path: Path, items: list[tuple[Path, Path]]) -> None:
    """items = [(src_abs, arcname_relative)]."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    n = 0
    log.info("creando %s ...", zip_path.name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arc in items:
            if not src.is_file():
                continue
            zf.write(src, arcname=str(arc).replace("\\", "/"))
            total_bytes += src.stat().st_size
            n += 1
            if n % 500 == 0:
                log.info("  %d archivos, %.1f MB sin comprimir...", n, total_bytes / 1e6)
    size = zip_path.stat().st_size / 1e6
    log.info("  listo %s: %d archivos, %.1f MB comprimidos (%.1f MB sin comprimir)",
             zip_path.name, n, size, total_bytes / 1e6)


def collect_code(repo: Path) -> list[tuple[Path, Path]]:
    """code zip: todo el repo menos venv, legacy, data/, checkpoints/, dataset2/, y zips grandes."""
    items: list[tuple[Path, Path]] = []
    top_level_exclude = {"data", "checkpoints", "dataset2", "legacy", "venv", ".venv", "export"}
    for child in repo.iterdir():
        if child.name in top_level_exclude:
            continue
        if child.is_file():
            if should_skip(child):
                continue
            items.append((child, Path(repo.name) / child.name))
            continue
        for f in iter_files(child):
            rel = f.relative_to(repo)
            items.append((f, Path(repo.name) / rel))
    return items


def collect_checkpoints(repo: Path) -> list[tuple[Path, Path]]:
    ck = repo / "checkpoints"
    items: list[tuple[Path, Path]] = []
    # Solo el pretrained base y el multisession mejor; saltamos iter50 intermedios.
    wanted = [
        ck / "pretrained" / "GaitBase_Gait3D_120000.pt",
        ck / "finetune" / "gaitbase_ft_multisession_best_iter1200.pt",
        ck / "finetune" / "gaitbase_ft_best_iter400.pt",  # Fase 3 por si regresamos
    ]
    for f in wanted:
        if f.is_file():
            items.append((f, Path(repo.name) / f.relative_to(repo)))
        else:
            log.warning("checkpoint no encontrado: %s", f)
    return items


def collect_pkls(repo: Path) -> list[tuple[Path, Path]]:
    items: list[tuple[Path, Path]] = []
    # pkl_multimodal/ es REQUERIDO para Fase 4 (SkeletonGait++ Ruta A)
    for subdir in ("pkl", "pkl_s2", "pkl_multisession", "pkl_multimodal"):
        src = repo / "data" / subdir
        if not src.is_dir():
            continue
        for f in iter_files(src):
            items.append((f, Path(repo.name) / f.relative_to(repo)))
    return items


def collect_data_raw(repo: Path, s1_raw: Path) -> list[tuple[Path, Path]]:
    items: list[tuple[Path, Path]] = []
    # Sesion 1 viene de fuera del repo; adentro se mete como data_raw/s1/<subject>/...
    if s1_raw.is_dir():
        for f in iter_files(s1_raw):
            rel = f.relative_to(s1_raw)
            items.append((f, Path(repo.name) / "data_raw" / "s1" / rel))
    else:
        log.warning("--s1-raw no existe: %s (sesion 1 no se empaqueta)", s1_raw)
    # Sesion 2 esta dentro del repo en dataset2/
    ds2 = repo / "dataset2"
    if ds2.is_dir():
        for f in iter_files(ds2):
            rel = f.relative_to(ds2)
            items.append((f, Path(repo.name) / "data_raw" / "s2" / rel))
    else:
        log.warning("dataset2/ no existe en el repo")
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="C:/Proyecto3/export",
                    help="donde guardar los zips (default C:/Proyecto3/export)")
    ap.add_argument("--s1-raw", default=str(DEFAULT_S1_RAW),
                    help="ruta al Dataset Crudo de sesion 1")
    ap.add_argument("--skip-raw", action="store_true",
                    help="omitir zip de videos crudos")
    ap.add_argument("--skip-pkls", action="store_true",
                    help="omitir zip de pkls")
    ap.add_argument("--skip-checkpoints", action="store_true",
                    help="omitir zip de checkpoints")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("repo    : %s", REPO)
    log.info("out_dir : %s", out_dir)

    log.info("=" * 60)
    items = collect_code(REPO)
    make_zip(out_dir / "ProyectoChino_code.zip", items)

    if not args.skip_checkpoints:
        log.info("=" * 60)
        items = collect_checkpoints(REPO)
        make_zip(out_dir / "ProyectoChino_checkpoints.zip", items)

    if not args.skip_pkls:
        log.info("=" * 60)
        items = collect_pkls(REPO)
        make_zip(out_dir / "ProyectoChino_pkls.zip", items)

    if not args.skip_raw:
        log.info("=" * 60)
        items = collect_data_raw(REPO, Path(args.s1_raw))
        make_zip(out_dir / "ProyectoChino_data_raw.zip", items)

    log.info("=" * 60)
    log.info("EXPORT LISTO. Archivos en %s:", out_dir)
    for zp in sorted(out_dir.glob("ProyectoChino_*.zip")):
        size = zp.stat().st_size / 1e6
        log.info("  %-35s  %7.1f MB", zp.name, size)
    log.info("")
    log.info("Siguiente: copia los zips a la PC nueva y sigue SETUP_NEW_PC.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
