"""Sensitivity Analysis v2: carga modelos UNA VEZ y muta sus atributos.

Soluciona el bug de v1 donde cada eval re-cargaba YOLO+RVM+GaitBase
(overhead de ~4 minutos por eval via subprocess).

Estrategia:
- YOLO weights se cargan 1 vez. Para cambiar conf/iou/imgsz, mutamos atributos.
- RVM (torch.hub) se carga 1 vez. Para cambiar downsample_ratio, mutamos atributo.
- GaitBase + Matcher: no cambian en este barrido.
- SequenceBuffer + TrackerFSM: livianos, se recrean por video.

Speedup esperado: 2-3x vs v1 (de ~9min/eval a ~3-4min/eval).

Salidas:
    logs/sensitivity/sensitivity_v2_<tag>.{json,csv}

Uso:
    python scripts/sensitivity_analysis_v2.py --tag fps_focus_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.pipeline.capture import Capture           # noqa: E402
from src.pipeline.detect import PersonDetector     # noqa: E402
from src.pipeline.track import TrackerFSM          # noqa: E402
from src.pipeline.segment import Segmenter         # noqa: E402
from src.pipeline.seq_buffer import SequenceBuffer  # noqa: E402
from src.pipeline.embed import GaitEmbedder        # noqa: E402
from src.pipeline.match import GalleryMatcher      # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("sensitivity_v2")
log.setLevel(logging.INFO)

BASELINE_CONFIG = REPO / "configs" / "pipeline.yaml"

PARAM_GRID = {
    "detector.imgsz":               [480, 832],          # 640 baseline (no probamos 320 que cuelga)
    "detector.conf":                [0.25, 0.50, 0.70],   # 0.35 baseline
    "detector.iou":                 [0.30, 0.70],         # 0.5 baseline
    "segmenter.downsample_ratio":   [0.125, 0.50],        # 0.25 baseline
    "sequence.window":              [30, 90, 120],        # 60 baseline
    "sequence.stride":              [15, 50],             # 30 baseline
}


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO / pp)


def process_one(video: Path, cap_fps: int, det: PersonDetector,
                seg: Segmenter, buf: SequenceBuffer, emb: GaitEmbedder,
                mch: GalleryMatcher, fsm_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Procesa UN video reusando los modelos cargados. Reset state al inicio."""
    cap = Capture(str(video), target_fps=cap_fps)
    fsm = TrackerFSM(**fsm_kwargs)
    seg.reset_state()
    buf.reset()

    n_frames = 0
    sequences: List = []
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
        return {"n_frames": n_frames, "n_sequences": 0, "elapsed_s": dt,
                "fps": n_frames / dt if dt > 0 else 0,
                "top1": None, "sim_top1": None}

    seq = sequences[0]
    e = emb.embed(seq)
    m = mch.match(e)
    return {
        "n_frames": n_frames, "n_sequences": len(sequences), "elapsed_s": dt,
        "fps": n_frames / dt if dt > 0 else 0,
        "top1": m.top5[0][0] if m.top5 else None,
        "sim_top1": float(m.sim),
        "unknown": bool(m.unknown),
    }


def evaluate_pipeline(cfg: Dict[str, Any], det: PersonDetector, seg: Segmenter,
                      emb: GaitEmbedder, mch: GalleryMatcher,
                      videos: List) -> Dict[str, Any]:
    """Procesa todos los videos con la config dada. Devuelve metricas agregadas."""
    # Reinstanciar buffer (window/stride pueden cambiar)
    buf = SequenceBuffer(window=cfg["sequence"]["window"],
                         stride=cfg["sequence"]["stride"])
    fsm_kwargs = dict(
        warmup_frames=cfg["tracker"]["warmup_frames"],
        reset_frames=cfg["tracker"]["reset_frames"],
        iou_min=cfg["tracker"]["iou_min"],
    )
    # Mutar matcher.tau si cambio
    mch.tau = float(cfg["matcher"]["tau"])

    tau = float(cfg["matcher"]["tau"])
    n_top1 = 0
    n_acc = 0
    n_with_seq = 0
    fps_list: List[float] = []

    for v, subj, _cond in videos:
        r = process_one(v, cfg["camera"]["target_fps"], det, seg, buf, emb, mch, fsm_kwargs)
        if r["n_sequences"] > 0:
            n_with_seq += 1
            fps_list.append(r["fps"])
            top1_raw = r["top1"]
            if top1_raw == subj:
                n_top1 += 1
                if r["sim_top1"] >= tau:
                    n_acc += 1

    n = len(videos)
    return {
        "n_videos": n,
        "n_with_sequence": n_with_seq,
        "n_top1_correct": n_top1,
        "n_accepted_correct": n_acc,
        "rank1_acc": round(n_top1 / n_with_seq, 4) if n_with_seq else 0.0,
        "accepted_rate_at_tau": round(n_acc / n_with_seq, 4) if n_with_seq else 0.0,
        "fps_mean": round(sum(fps_list) / len(fps_list), 2) if fps_list else 0,
        "fps_min": round(min(fps_list), 2) if fps_list else 0,
        "fps_max": round(max(fps_list), 2) if fps_list else 0,
    }


def list_videos(dataset_root: Path):
    out = []
    for subj_dir in sorted(dataset_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        for v in sorted(subj_dir.glob("*.mp4")):
            cond = v.stem
            out.append((v, subj_dir.name, cond))
    return out


def apply_config_change(param_path: str, value, det: PersonDetector, seg: Segmenter):
    """Muta el atributo del modelo correspondiente sin recargar."""
    parts = param_path.split(".")
    section, key = parts[0], parts[1]
    if section == "detector":
        setattr(det, key, value)
        log.info("  det.%s = %s", key, value)
    elif section == "segmenter":
        setattr(seg, key, value)
        log.info("  seg.%s = %s", key, value)
    # tracker, sequence, matcher se aplican en evaluate_pipeline directamente


def _set_nested(d: Dict, path: str, value) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    ap.add_argument("--dataset", default=str(REPO / "dataset2"))
    args = ap.parse_args()

    out_dir = REPO / "logs" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    with BASELINE_CONFIG.open("r", encoding="utf-8") as f:
        baseline_cfg = yaml.safe_load(f)

    videos = list_videos(Path(args.dataset))
    print(f"Videos encontrados: {len(videos)}")

    print("=" * 70)
    print("SENSITIVITY ANALYSIS v2 - modelos cargados UNA VEZ")
    print("=" * 70)

    # === Carga UNA VEZ los modelos pesados ===
    t0 = time.time()
    print("\n[init] cargando YOLO + RVM + GaitBase + Matcher...")
    det = PersonDetector(
        weights=_resolve(baseline_cfg["detector"]["weights"]),
        conf=baseline_cfg["detector"]["conf"],
        iou=baseline_cfg["detector"]["iou"],
        imgsz=baseline_cfg["detector"].get("imgsz", 640),
    )
    seg = Segmenter(downsample_ratio=baseline_cfg["segmenter"]["downsample_ratio"])
    emb = GaitEmbedder(
        ckpt=_resolve(baseline_cfg["embedder"]["ckpt"]),
        class_num=baseline_cfg["embedder"]["class_num"],
        compile=baseline_cfg["embedder"].get("compile", False),
    )
    mch = GalleryMatcher(
        gallery_npy=_resolve(baseline_cfg["matcher"]["gallery_npy"]),
        valid_mask_npy=_resolve(baseline_cfg["matcher"]["valid_mask_npy"]),
        index_json=_resolve(baseline_cfg["matcher"]["index_json"]),
        tau=baseline_cfg["matcher"]["tau"],
    )
    print(f"[init] modelos listos en {time.time()-t0:.1f}s")

    # === Plan de runs ===
    total_runs = 1 + sum(
        len([v for v in vals]) for p, vals in PARAM_GRID.items()
    )
    print(f"\nTotal runs: 1 baseline + {total_runs-1} variaciones = {total_runs}")
    print()

    results: List[Dict[str, Any]] = []
    t_start = time.time()
    run_idx = 0

    # 1. Baseline
    run_idx += 1
    print(f"[{run_idx}/{total_runs}] BASELINE")
    t0 = time.time()
    r = evaluate_pipeline(baseline_cfg, det, seg, emb, mch, videos)
    r["param"] = "baseline"
    r["value"] = None
    r["elapsed_s"] = round(time.time() - t0, 1)
    r["is_baseline"] = True
    print(f"  rank1={r['rank1_acc']}  fps_mean={r['fps_mean']}  elapsed={r['elapsed_s']}s")
    results.append(r)

    fps_baseline = r["fps_mean"]
    acc_baseline = r["rank1_acc"]

    # Snapshot baseline atributos
    baseline_attrs = {
        "detector.imgsz": det.imgsz,
        "detector.conf": det.conf,
        "detector.iou": det.iou,
        "segmenter.downsample_ratio": seg.downsample_ratio,
    }

    # 2. Variaciones
    for param_path, values in PARAM_GRID.items():
        for v in values:
            run_idx += 1
            cfg_var = json.loads(json.dumps(baseline_cfg))  # deep copy
            _set_nested(cfg_var, param_path, v)

            # Aplicar al modelo (mutacion) si es detector/segmenter
            section = param_path.split(".")[0]
            if section in ("detector", "segmenter"):
                apply_config_change(param_path, v, det, seg)

            print(f"\n[{run_idx}/{total_runs}] {param_path} = {v}")
            t0 = time.time()
            r = evaluate_pipeline(cfg_var, det, seg, emb, mch, videos)
            r["param"] = param_path
            r["value"] = v
            r["elapsed_s"] = round(time.time() - t0, 1)
            r["is_baseline"] = False
            r["delta_fps"] = round(r["fps_mean"] - fps_baseline, 2)
            r["delta_acc"] = round(r["rank1_acc"] - acc_baseline, 4)
            print(f"  rank1={r['rank1_acc']}  fps_mean={r['fps_mean']}  delta_fps={r['delta_fps']:+}  delta_acc={r['delta_acc']:+}  elapsed={r['elapsed_s']}s")
            results.append(r)

            # Guardado incremental
            with (out_dir / f"sensitivity_v2_{args.tag}_partial.json").open("w") as f:
                json.dump(results, f, indent=2)

            # Restaurar atributo del modelo (volver a baseline) si modificamos
            if param_path in baseline_attrs:
                setattr(det if section == "detector" else seg,
                        param_path.split(".")[1], baseline_attrs[param_path])

    total_dt = time.time() - t_start

    # Guardar resultados finales
    out_json = out_dir / f"sensitivity_v2_{args.tag}.json"
    out_csv = out_dir / f"sensitivity_v2_{args.tag}.csv"

    final = {
        "tag": args.tag,
        "total_runs": len(results),
        "total_elapsed_s": round(total_dt, 1),
        "baseline_fps": fps_baseline,
        "baseline_acc": acc_baseline,
        "results": results,
    }
    with out_json.open("w") as f:
        json.dump(final, f, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "value", "rank1_acc", "fps_mean", "delta_fps", "delta_acc", "elapsed_s"])
        for r in results:
            w.writerow([
                r.get("param"), r.get("value"), r.get("rank1_acc"),
                r.get("fps_mean"), r.get("delta_fps"), r.get("delta_acc"),
                r.get("elapsed_s"),
            ])

    # Ranking
    print("\n" + "=" * 70)
    print("RANKING POR IMPACTO EN FPS (absoluto)")
    print("=" * 70)
    variations = [r for r in results if not r.get("is_baseline")]
    variations.sort(key=lambda x: abs(x.get("delta_fps", 0)), reverse=True)
    for r in variations:
        delta_fps = r.get("delta_fps", 0)
        delta_acc = r.get("delta_acc", 0)
        print(f"  {r['param']:35s} = {str(r['value']):8s}  "
              f"fps={r['fps_mean']:6.2f} (delta={delta_fps:+6.2f})  "
              f"acc={r['rank1_acc']:.4f} (delta={delta_acc:+.4f})")

    print(f"\nTotal: {total_dt/60:.1f} min ({total_dt:.0f}s)")
    print(f"-> {out_json}")
    print(f"-> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
