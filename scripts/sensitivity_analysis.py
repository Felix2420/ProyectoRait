"""Sensitivity Analysis para el pipeline completo de gait recognition.

Prueba cada parametro AISLADAMENTE (uno a la vez, los demas en baseline)
para identificar cuales mas impactan FPS y accuracy.

Estrategia: para cada (param, value), genera un config temporal, corre
eval_pipeline_offline (37 videos) y captura FPS promedio + accuracy.

Salidas:
    logs/sensitivity/sensitivity_<tag>.json
    logs/sensitivity/sensitivity_<tag>.csv

Uso:
    python scripts/sensitivity_analysis.py --tag fps_focus
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPO / "configs" / "pipeline.yaml"
EVAL_REPORT = REPO / "reports" / "16_pipeline_offline_eval.json"


# Definicion de parametros a evaluar
# Cada entrada: (path_en_config, valores_a_probar)
PARAM_GRID = {
    "detector.imgsz":           [320, 480, 640, 832],
    "detector.conf":            [0.25, 0.35, 0.50, 0.70],
    "detector.iou":             [0.30, 0.50, 0.70],
    "segmenter.downsample_ratio": [0.125, 0.25, 0.50],
    "sequence.window":          [30, 60, 90, 120],
    "sequence.stride":          [15, 30, 50],
}


def _set_nested(d: Dict, path: str, value) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def _get_nested(d: Dict, path: str):
    cur = d
    for k in path.split("."):
        cur = cur[k]
    return cur


def run_one_eval(config_path: Path) -> Dict[str, Any]:
    """Ejecuta eval_pipeline_offline con un config y retorna metricas."""
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "eval_pipeline_offline.py"),
         "--config", str(config_path)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=900,
    )
    dt = time.time() - t0

    if result.returncode != 0:
        return {
            "elapsed_s": dt,
            "error": f"returncode={result.returncode}",
            "stderr_tail": (result.stderr or "")[-500:],
        }

    # Leer el report JSON
    if not EVAL_REPORT.exists():
        return {"elapsed_s": dt, "error": "no se encontro reports/16_pipeline_offline_eval.json"}

    with EVAL_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    summary = report.get("summary", {})
    rows = report.get("rows", [])
    fps_values = [r["fps"] for r in rows if r.get("fps") is not None and r["n_sequences"] > 0]
    fps_mean = sum(fps_values) / len(fps_values) if fps_values else 0.0

    return {
        "elapsed_s": round(dt, 1),
        "rank1_acc": summary.get("rank1_acc"),
        "accepted_rate": summary.get("accepted_rate_at_tau"),
        "n_top1_correct": summary.get("n_top1_correct"),
        "n_videos": summary.get("n_videos"),
        "n_with_sequence": summary.get("n_with_sequence"),
        "fps_mean": round(fps_mean, 2),
        "fps_min": round(min(fps_values), 2) if fps_values else 0,
        "fps_max": round(max(fps_values), 2) if fps_values else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    ap.add_argument("--baseline-only", action="store_true",
                    help="solo corre baseline (para verificar antes de barrido completo)")
    args = ap.parse_args()

    out_dir = REPO / "logs" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = REPO / "logs" / "sensitivity" / "_tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with BASELINE_CONFIG.open("r", encoding="utf-8") as f:
        baseline_cfg = yaml.safe_load(f)

    print("=" * 70)
    print("SENSITIVITY ANALYSIS - Pipeline completo")
    print("=" * 70)
    print(f"baseline config: {BASELINE_CONFIG.name}")
    for path in PARAM_GRID:
        print(f"  {path:35s} = {_get_nested(baseline_cfg, path)}")
    print()

    # Conteo total
    total_runs = 1 + sum(
        len([v for v in vals if v != _get_nested(baseline_cfg, p)])
        for p, vals in PARAM_GRID.items()
    )
    print(f"Total runs: 1 baseline + {total_runs-1} variaciones = {total_runs}")
    print(f"Tiempo estimado: ~{total_runs * 5:.0f}-{total_runs * 6:.0f} min")
    print("=" * 70)

    results: List[Dict[str, Any]] = []
    run_idx = 0
    t_start = time.time()

    # 1. Baseline
    run_idx += 1
    print(f"\n[{run_idx}/{total_runs}] BASELINE (config original)")
    baseline_result = run_one_eval(BASELINE_CONFIG)
    baseline_result["param"] = "baseline"
    baseline_result["value"] = None
    baseline_result["is_baseline"] = True
    print(f"  rank1={baseline_result.get('rank1_acc')}  fps_mean={baseline_result.get('fps_mean')}  elapsed={baseline_result.get('elapsed_s')}s")
    results.append(baseline_result)

    if args.baseline_only:
        print("\n[baseline-only] terminando aqui.")
        with (out_dir / f"sensitivity_{args.tag}_baseline.json").open("w") as f:
            json.dump(results, f, indent=2)
        return 0

    fps_baseline = baseline_result.get("fps_mean", 0)
    acc_baseline = baseline_result.get("rank1_acc", 0)

    # 2. Variar cada parametro
    for param_path, values in PARAM_GRID.items():
        baseline_value = _get_nested(baseline_cfg, param_path)
        for v in values:
            if v == baseline_value:
                continue  # ya cubierto por baseline
            run_idx += 1
            cfg_copy = json.loads(json.dumps(baseline_cfg))  # deep copy
            _set_nested(cfg_copy, param_path, v)

            tmp_cfg = tmp_dir / f"cfg_{param_path.replace('.', '_')}_{v}.yaml"
            with tmp_cfg.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_copy, f)

            print(f"\n[{run_idx}/{total_runs}] {param_path} = {v}  (baseline {baseline_value})")
            r = run_one_eval(tmp_cfg)
            r["param"] = param_path
            r["value"] = v
            r["is_baseline"] = False
            r["delta_fps"] = round(r.get("fps_mean", 0) - fps_baseline, 2) if r.get("fps_mean") else None
            r["delta_acc"] = round(r.get("rank1_acc", 0) - acc_baseline, 4) if r.get("rank1_acc") is not None else None
            print(f"  rank1={r.get('rank1_acc')}  fps_mean={r.get('fps_mean')}  delta_fps={r.get('delta_fps'):+}  delta_acc={r.get('delta_acc'):+}  elapsed={r.get('elapsed_s')}s")
            results.append(r)

            # Guardado incremental por si se interrumpe
            with (out_dir / f"sensitivity_{args.tag}_partial.json").open("w") as f:
                json.dump(results, f, indent=2)

    total_dt = time.time() - t_start

    # Guardar resultados finales
    out_json = out_dir / f"sensitivity_{args.tag}.json"
    out_csv = out_dir / f"sensitivity_{args.tag}.csv"

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

    # CSV con resumen
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "value", "rank1_acc", "fps_mean", "delta_fps", "delta_acc", "elapsed_s"])
        for r in results:
            w.writerow([
                r.get("param"), r.get("value"),
                r.get("rank1_acc"), r.get("fps_mean"),
                r.get("delta_fps"), r.get("delta_acc"),
                r.get("elapsed_s"),
            ])

    # Resumen ranking por impacto en FPS
    print("\n" + "=" * 70)
    print("RANKING POR IMPACTO EN FPS")
    print("=" * 70)
    variations = [r for r in results if not r.get("is_baseline") and r.get("delta_fps") is not None]
    variations.sort(key=lambda r: abs(r.get("delta_fps", 0)), reverse=True)
    for r in variations[:10]:
        print(f"  {r['param']:35s} = {str(r['value']):8s}  delta_fps={r['delta_fps']:+6.2f}  delta_acc={r['delta_acc']:+.4f}")

    print(f"\nTotal: {total_dt/60:.1f} min")
    print(f"-> {out_json}")
    print(f"-> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
