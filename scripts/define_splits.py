"""
Fase 2 paso 2 — Definición de splits.

Genera configs/splits.yaml con dos esquemas:

(A) subject-disjoint para evaluar el MODELO (Fase 3+):
    train(13) / val(3) / test(3), asignación determinística vía sha256
    del nombre del sujeto + SEED.  Sin bias alfabético, reproducible.

(B) video-disjoint para evaluar el SISTEMA de pase de lista (Fase 6):
    los 19 sujetos en gallery (condición "normal") y en probe (condición
    "rapido").  Es exactamente el escenario de uso real.

Uso:
    python scripts/define_splits.py \
        --raw "C:/Proyecto3/Proyecto-Reconocedor-IA/Dataset Crudo" \
        --out "C:/Proyecto3/ProyectoChino/configs/splits.yaml" \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("splits")


def hash_rank(name: str, seed: int) -> int:
    """Devuelve un entero determinístico para ordenar sujetos."""
    h = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def list_subjects(raw_root: Path) -> list[str]:
    subs = sorted(d.name for d in raw_root.iterdir() if d.is_dir())
    if not subs:
        raise SystemExit(f"No se hallaron subdirectorios en {raw_root}")
    return subs


def list_videos(raw_root: Path, subject: str) -> dict[str, str]:
    """Mapea condición → ruta relativa del video.

    Reconoce nombres `normal_*.mp4` y `rapido_*.mp4`.  Si hay >1 toma de
    una misma condición, conserva la última (alfabéticamente).
    """
    out: dict[str, str] = {}
    sub_dir = raw_root / subject
    for f in sorted(sub_dir.glob("*.mp4")):
        low = f.name.lower()
        if low.startswith("normal_"):
            out["normal"] = f.name
        elif low.startswith("rapido_"):
            out["rapido"] = f.name
    return out


def build_subject_disjoint(subjects: list[str], seed: int) -> dict[str, list[str]]:
    ranked = sorted(subjects, key=lambda s: hash_rank(s, seed))
    if len(ranked) != 19:
        log.warning("Esperaba 19 sujetos, encontré %d. Ajustando proporciones.", len(ranked))
    n = len(ranked)
    n_test = max(1, round(n * 3 / 19))
    n_val = max(1, round(n * 3 / 19))
    n_train = n - n_val - n_test
    return {
        "train": ranked[:n_train],
        "val":   ranked[n_train:n_train + n_val],
        "test":  ranked[n_train + n_val:],
    }


def build_video_disjoint(subjects: list[str], video_map: dict[str, dict[str, str]]):
    gallery, probe, missing = [], [], []
    for s in subjects:
        v = video_map.get(s, {})
        if "normal" in v:
            gallery.append({"subject": s, "video": v["normal"]})
        else:
            missing.append({"subject": s, "missing": "normal"})
        if "rapido" in v:
            probe.append({"subject": s, "video": v["rapido"]})
        else:
            missing.append({"subject": s, "missing": "rapido"})
    return {
        "gallery_condition": "normal",
        "probe_condition":   "rapido",
        "gallery": gallery,
        "probe":   probe,
        "missing": missing,
    }


def to_yaml(data, indent: int = 0) -> str:
    """Mini-emitter YAML para evitar dependencia adicional. Soporta dict/list/str/int/bool."""
    lines: list[str] = []
    pad = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.append(to_yaml(v, indent + 1))
            elif isinstance(v, list):
                lines.append(f"{pad}{k}: []")
            elif isinstance(v, dict):
                lines.append(f"{pad}{k}: {{}}")
            else:
                lines.append(f"{pad}{k}: {_scalar(v)}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                inner = to_yaml(item, indent + 1).splitlines()
                if inner:
                    lines.append(f"{pad}- {inner[0].lstrip()}")
                    lines.extend(inner[1:])
            else:
                lines.append(f"{pad}- {_scalar(item)}")
    else:
        lines.append(f"{pad}{_scalar(data)}")
    return "\n".join(lines)


def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(c in s for c in ":#[]{}&*!|>'\"%@`,") or s != s.strip():
        return json.dumps(s, ensure_ascii=False)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subjects = list_subjects(raw_root)
    log.info("Sujetos detectados: %d", len(subjects))

    video_map = {s: list_videos(raw_root, s) for s in subjects}

    subj = build_subject_disjoint(subjects, args.seed)
    vid = build_video_disjoint(subjects, video_map)

    out = {
        "meta": {
            "n_subjects":   len(subjects),
            "seed":         args.seed,
            "raw_root":     str(raw_root),
            "phase":        "Fase 2 paso 2 — splits definidos",
            "ranking_rule": "sha256(f'{seed}:{subject}')[:16] como int → orden creciente",
        },
        "subject_disjoint": {
            "purpose": "evaluar capacidad de generalización del modelo a IDs no vistas",
            "n": {k: len(v) for k, v in subj.items()},
            **subj,
        },
        "video_disjoint": {
            "purpose": "evaluar funcionalidad del sistema de pase de lista (N=19)",
            **vid,
        },
    }

    yaml_text = to_yaml(out)
    out_path.write_text(yaml_text + "\n", encoding="utf-8")

    log.info("Splits escritos en %s", out_path)
    log.info("subject-disjoint  → train=%d  val=%d  test=%d",
             len(subj["train"]), len(subj["val"]), len(subj["test"]))
    log.info("  train: %s", subj["train"])
    log.info("  val:   %s", subj["val"])
    log.info("  test:  %s", subj["test"])
    log.info("video-disjoint    → gallery=%d  probe=%d  faltantes=%d",
             len(vid["gallery"]), len(vid["probe"]), len(vid["missing"]))
    if vid["missing"]:
        log.warning("Sujetos con condición faltante: %s", vid["missing"])


if __name__ == "__main__":
    main()
