"""Genera partition JSON en el formato que espera opengait/data/dataset.py.

Lee configs/splits.yaml (Fase 2) y emite dos JSONs:

1. configs/partition_finetune.json
   - TRAIN_SET: 13 sujetos de train.
   - TEST_SET: 6 sujetos (val + test) — para evaluacion subject-disjoint en fine-tuning.

2. configs/partition_zeroshot.json
   - TRAIN_SET: [] (no se usa en --phase test).
   - TEST_SET: los 19 sujetos completos — para extraer embeddings de todos
     y luego hacer matching gallery(normal) vs probe(rapido).
"""

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="configs/splits.yaml")
    ap.add_argument("--out_dir", default="configs")
    args = ap.parse_args()

    splits_path = Path(args.splits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with splits_path.open("r", encoding="utf-8") as f:
        splits = yaml.safe_load(f)

    sd = splits["subject_disjoint"]
    train = sorted(sd["train"])
    val = sorted(sd["val"])
    test = sorted(sd["test"])
    all_19 = sorted(train + val + test)

    finetune = {"TRAIN_SET": train, "TEST_SET": sorted(val + test)}
    zeroshot = {"TRAIN_SET": [], "TEST_SET": all_19}

    (out_dir / "partition_finetune.json").write_text(
        json.dumps(finetune, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "partition_zeroshot.json").write_text(
        json.dumps(zeroshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"finetune: {len(finetune['TRAIN_SET'])} train / {len(finetune['TEST_SET'])} test")
    print(f"zeroshot: {len(zeroshot['TEST_SET'])} test (todos los sujetos)")
    print(f"escrito en {out_dir}/")


if __name__ == "__main__":
    main()
