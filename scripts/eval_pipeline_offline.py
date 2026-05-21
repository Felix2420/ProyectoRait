"""Validación offline del pipeline de Fase 6 sobre dataset2/.

Procesa cada video .mp4 bajo `dataset2/<subject>/{normal_1,rapido_1}.mp4`,
ejecutando el pipeline completo (capture → detect → track → segment →
seq_buffer → embed → match), y agrega:
  - Top-1 sujeto y similitud por video.
  - Acierto rank-1 (top1 == GT).
  - Aceptación con τ (sim_top1 >= τ AND top1 == GT).

No usa la política N=2 (los videos son de 5 s y solo dan 1 secuencia). Esto
valida pipeline puro: paridad del preproceso runtime con el de entrenamiento,
y consistencia del matcher contra la gallery multi-template.

Salidas:
  reports/16_pipeline_offline_eval.json
  reports/16_pipeline_offline_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.pipeline.capture import Capture          # noqa: E402
from src.pipeline.detect import PersonDetector    # noqa: E402
from src.pipeline.track import TrackerFSM         # noqa: E402
from src.pipeline.segment import Segmenter        # noqa: E402
from src.pipeline.seq_buffer import SequenceBuffer  # noqa: E402
from src.pipeline.embed import GaitEmbedder       # noqa: E402
from src.pipeline.match import GalleryMatcher     # noqa: E402

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("eval_pipeline_offline")
log.setLevel(logging.INFO)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO / pp)


def process_one(video: Path, cap_fps: int, det: PersonDetector,
                seg_factory, buf_factory, emb: GaitEmbedder,
                mch: GalleryMatcher, fsm_kwargs: Dict[str, Any]) -> Dict[str, Any] | None:
    """Procesa un video. Devuelve dict con resultado o None si no se pudo."""
    cap = Capture(str(video), target_fps=cap_fps)
    fsm = TrackerFSM(**fsm_kwargs)
    seg = seg_factory()
    buf = buf_factory()

    n_frames = 0
    sequences = []
    t0 = time.time()
    while True:
        item = cap.read()
        if item is None:
            break
        frame, _ts = item
        n_frames += 1
        bbox = det.detect(frame)
        state = fsm.step(bbox)
        if state == "RESET":
            buf.reset()
            seg.reset_state()
            continue
        if state != "ACTIVE":
            _ = seg.segment(frame, fsm.last_bbox)
            continue
        sil = seg.segment(frame, fsm.last_bbox)
        if sil is None:
            continue
        seq = buf.push(sil)
        if seq is not None:
            sequences.append(seq)
    cap.release()
    dt = time.time() - t0

    if not sequences:
        return {
            "video": str(video.relative_to(REPO)),
            "n_frames": n_frames,
            "n_sequences": 0,
            "elapsed_s": round(dt, 2),
            "fps": round(n_frames / dt if dt > 0 else 0.0, 2),
            "top1": None,
            "sim_top1": None,
            "top5": [],
        }

    # Tomar la primera secuencia (con 5 s de video solo hay 1)
    seq = sequences[0]
    e = emb.embed(seq)
    m = mch.match(e)
    return {
        "video": str(video.relative_to(REPO)).replace("\\", "/"),
        "n_frames": n_frames,
        "n_sequences": len(sequences),
        "elapsed_s": round(dt, 2),
        "fps": round(n_frames / dt if dt > 0 else 0.0, 2),
        "top1": m.subject if not m.unknown else None,
        "top1_raw": m.top5[0][0] if m.top5 else None,
        "sim_top1": round(m.sim, 4),
        "unknown": bool(m.unknown),
        "top5": [(s, round(v, 4)) for s, v in m.top5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(REPO / "dataset2"),
                    help="Raíz dataset2/<subj>/<cond>_1.mp4")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    dataset_root = Path(args.dataset)
    if not dataset_root.is_dir():
        log.error("dataset no existe: %s", dataset_root)
        return 1

    # Inicializar modelos UNA vez (compartidos entre videos)
    det = PersonDetector(weights=resolve(cfg["detector"]["weights"]),
                         conf=cfg["detector"]["conf"],
                         iou=cfg["detector"]["iou"],
                         imgsz=cfg["detector"].get("imgsz", 640))
    emb = GaitEmbedder(ckpt=resolve(cfg["embedder"]["ckpt"]),
                       class_num=cfg["embedder"]["class_num"],
                       compile=cfg["embedder"].get("compile", False))
    mch = GalleryMatcher(gallery_npy=resolve(cfg["matcher"]["gallery_npy"]),
                         valid_mask_npy=resolve(cfg["matcher"]["valid_mask_npy"]),
                         index_json=resolve(cfg["matcher"]["index_json"]),
                         tau=cfg["matcher"]["tau"])

    fsm_kwargs = dict(
        warmup_frames=cfg["tracker"]["warmup_frames"],
        reset_frames=cfg["tracker"]["reset_frames"],
        iou_min=cfg["tracker"]["iou_min"],
    )

    def seg_factory():
        return Segmenter(downsample_ratio=cfg["segmenter"]["downsample_ratio"])

    def buf_factory():
        return SequenceBuffer(window=cfg["sequence"]["window"],
                              stride=cfg["sequence"]["stride"])

    # Listar videos
    videos: List[tuple[Path, str, str]] = []
    for subj_dir in sorted(dataset_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        for v in sorted(subj_dir.glob("*.mp4")):
            cond = v.stem  # normal_1 / rapido_1
            videos.append((v, subj_dir.name, cond))
    log.info("encontrados %d videos en %s", len(videos), dataset_root)

    rows: List[Dict[str, Any]] = []
    n_top1_correct = 0
    n_accepted_correct = 0
    n_with_seq = 0
    tau = float(cfg["matcher"]["tau"])

    for v, subj, cond in videos:
        res = process_one(v, cfg["camera"]["target_fps"], det, seg_factory,
                          buf_factory, emb, mch, fsm_kwargs)
        if res is None:
            log.warning("no se pudo procesar %s", v)
            continue
        res["expected_subject"] = subj
        res["expected_cond"] = cond
        if res["n_sequences"] > 0:
            n_with_seq += 1
            top1_raw = res.get("top1_raw")
            res["top1_correct"] = (top1_raw == subj)
            res["accepted_correct"] = (res["top1_correct"] and res["sim_top1"] >= tau)
            if res["top1_correct"]:
                n_top1_correct += 1
            if res["accepted_correct"]:
                n_accepted_correct += 1
        else:
            res["top1_correct"] = False
            res["accepted_correct"] = False
        rows.append(res)
        flag_t1 = "OK" if res.get("top1_correct") else "ERR"
        flag_acc = "ACC" if res.get("accepted_correct") else ("REJ" if res["n_sequences"] > 0 else "NO_SEQ")
        log.info("%s/%s top1=%s sim=%s [%s/%s] (%.1f fps, %d frames)",
                 subj, cond,
                 res.get("top1_raw"), res.get("sim_top1"),
                 flag_t1, flag_acc, res["fps"], res["n_frames"])

    n = len(rows)
    summary = {
        "n_videos": n,
        "n_with_sequence": n_with_seq,
        "n_top1_correct": n_top1_correct,
        "n_accepted_correct": n_accepted_correct,
        "rank1_acc": round(n_top1_correct / n_with_seq, 4) if n_with_seq else 0.0,
        "accepted_rate_at_tau": round(n_accepted_correct / n_with_seq, 4) if n_with_seq else 0.0,
        "tau": tau,
    }
    out = {
        "summary": summary,
        "config_used": {
            "tau": tau,
            "window": cfg["sequence"]["window"],
            "stride": cfg["sequence"]["stride"],
            "target_fps": cfg["camera"]["target_fps"],
        },
        "rows": rows,
    }

    out_json = REPO / "reports" / "16_pipeline_offline_eval.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    out_csv = REPO / "reports" / "16_pipeline_offline_eval.csv"
    if rows:
        cols = ["expected_subject", "expected_cond", "top1_raw", "sim_top1",
                "top1_correct", "accepted_correct", "n_sequences", "n_frames", "fps"]
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])

    log.info("=== RESUMEN ===")
    for k, v in summary.items():
        log.info("  %s = %s", k, v)
    log.info("-> %s", out_json.relative_to(REPO))
    log.info("-> %s", out_csv.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
