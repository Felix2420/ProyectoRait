"""Recalibración de τ para Fase 6 con gallery multi-template + matching max-sim.

Protocolo Leave-Condition-Out (LCO) sobre 19 sujetos × 4 condiciones (75 secs):

Para cada probe (subj_p, cond_p):
  - Gallery efectiva = gallery completa SIN la pkl (subj_p, cond_p).
    Cada sujeto contribuye con sus K-1 (o K) templates restantes.
  - Genuino: top-1 sobre la gallery efectiva. Si top1_subj == subj_p, correcto.
            sim_top1 = max sim contra los templates del propio sujeto.
  - Impostor: misma probe contra gallery efectiva SIN ningún template de subj_p.
              sim_top1_imp = max sim contra cualquier template de otros sujetos.

Para cada τ:
  TAR = (top1_correct AND sim_top1 >= τ) / n_genuine
  FAR = (sim_top1_imp >= τ) / n_impostor

Operating points: EER, τ@FAR=0%, τ@FAR≤1%, τ@FAR≤5%.

Salidas:
  reports/15_openset_multitemplate.json
  reports/15_openset_multitemplate_pairs.csv
  reports/15_openset_multitemplate_roc.png (si matplotlib)

Uso:
    python scripts/openset_eval_multitemplate.py \\
        --checkpoint checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt \\
        --class-num 13
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
OPENGAIT = REPO / "third_party" / "OpenGait"
REPORTS = REPO / "reports"

sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))

from modeling.backbones.resnet import ResNet9  # noqa: E402
from modeling.modules import (  # noqa: E402
    SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
    SeparateFCs, SeparateBNNecks,
)


class GaitBaseInfer(nn.Module):
    def __init__(self, class_num: int = 13) -> None:
        super().__init__()
        backbone = ResNet9(
            block="BasicBlock",
            channels=[64, 128, 256, 512],
            in_channel=1,
            layers=[1, 1, 1, 1],
            strides=[1, 2, 2, 1],
            maxpool=False,
        )
        self.Backbone = SetBlockWrapper(backbone)
        self.FCs = SeparateFCs(in_channels=512, out_channels=256, parts_num=16)
        self.BNNecks = SeparateBNNecks(class_num=class_num, in_channels=256, parts_num=16)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def forward(self, sils: torch.Tensor, seqL: List[torch.Tensor]) -> torch.Tensor:
        if sils.dim() == 4:
            sils = sils.unsqueeze(1)
        outs = self.Backbone(sils)
        outs = self.TP(outs, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs)
        return self.FCs(feat)


def load_silhouettes(pkl_path: Path) -> np.ndarray:
    with pkl_path.open("rb") as f:
        arr = pickle.load(f)
    if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[1:] != (64, 44):
        raise ValueError(f"{pkl_path}: shape inesperada")
    return arr


@torch.no_grad()
def extract_emb(model: GaitBaseInfer, sils_np: np.ndarray, device: torch.device) -> np.ndarray:
    sils = torch.from_numpy(sils_np.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    seqL = [torch.tensor([sils.shape[1]], device=device)]
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        emb = model(sils, seqL)
    e = emb.float().squeeze(0).cpu().numpy().reshape(-1)
    n = float(np.linalg.norm(e))
    return (e / n).astype(np.float32) if n > 1e-12 else e.astype(np.float32)


def collect_pkls(ms_root: Path) -> Dict[Tuple[str, str], Path]:
    """Devuelve {(subj, cond): pkl_path} para todas las pkls de pkl_multisession."""
    out: Dict[Tuple[str, str], Path] = {}
    for subj_dir in sorted(ms_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        for cond_dir in sorted(subj_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            seq_dir = cond_dir / "090"
            if not seq_dir.is_dir():
                continue
            pkls = sorted(seq_dir.glob("*.pkl"))
            if pkls:
                out[(subj_dir.name, cond_dir.name)] = pkls[0]
    return out


def sweep_tau(genuine: List[Dict], impostor: List[Dict], n_taus: int = 4001) -> Dict:
    g_sim = np.array([p["sim_top1"] for p in genuine])
    g_ok = np.array([p["correct_top1"] for p in genuine], dtype=bool)
    i_sim = np.array([p["sim_top1"] for p in impostor])
    if len(g_sim) == 0 or len(i_sim) == 0:
        return {"n_genuine": int(len(g_sim)), "n_impostor": int(len(i_sim))}

    lo = float(min(g_sim.min(), i_sim.min())) - 1e-4
    hi = float(max(g_sim.max(), i_sim.max())) + 1e-4
    taus = np.linspace(lo, hi, n_taus)

    tar = np.array([(g_ok & (g_sim >= t)).sum() / len(g_sim) for t in taus])
    far = np.array([(i_sim >= t).sum() / len(i_sim) for t in taus])
    frr = 1.0 - tar
    eer_idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[eer_idx] + frr[eer_idx]) / 2.0)

    def tau_at_far(target: float) -> Tuple[float, float, float]:
        valid = np.where(far <= target)[0]
        if len(valid) == 0:
            return float("nan"), float("nan"), float("nan")
        idx = int(valid[0])
        return float(taus[idx]), float(far[idx]), float(tar[idx])

    t0, far0, tar0 = tau_at_far(0.0)
    t1, far1, tar1 = tau_at_far(0.01)
    t5, far5, tar5 = tau_at_far(0.05)

    return {
        "n_genuine": int(len(g_sim)),
        "n_impostor": int(len(i_sim)),
        "rank1_top1_correct": round(float(g_ok.mean()), 4),
        "EER": round(eer, 4),
        "tau_EER": round(float(taus[eer_idx]), 4),
        "genuine_stats": {
            "mean": round(float(g_sim.mean()), 4),
            "min":  round(float(g_sim.min()), 4),
            "max":  round(float(g_sim.max()), 4),
            "std":  round(float(g_sim.std()), 4),
        },
        "impostor_stats": {
            "mean": round(float(i_sim.mean()), 4),
            "min":  round(float(i_sim.min()), 4),
            "max":  round(float(i_sim.max()), 4),
            "std":  round(float(i_sim.std()), 4),
        },
        "gap_mean": round(float(g_sim.mean() - i_sim.mean()), 4),
        "op_points": {
            "FAR=0%":  {"tau": round(t0, 4), "FAR": round(far0, 4), "TAR": round(tar0, 4)},
            "FAR<=1%": {"tau": round(t1, 4), "FAR": round(far1, 4), "TAR": round(tar1, 4)},
            "FAR<=5%": {"tau": round(t5, 4), "FAR": round(far5, 4), "TAR": round(tar5, 4)},
        },
        "curve": {
            "tau": [round(float(t), 4) for t in taus],
            "TAR": [round(float(v), 4) for v in tar],
            "FAR": [round(float(v), 4) for v in far],
            "FRR": [round(float(v), 4) for v in frr],
        },
    }


def try_plot_roc(curves: Dict[str, Dict], out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib no disponible ({e}); se omite PNG")
        return False
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for name, r in curves.items():
        if "curve" not in r:
            continue
        ax.plot(r["curve"]["FAR"], r["curve"]["TAR"], label=f"{name} (EER={r['EER']*100:.1f}%)")
    ax.set_xlabel("FAR"); ax.set_ylabel("TAR")
    ax.set_title("Open-set ROC — multi-template + max-sim (LCO, 19 sujetos)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints" / "finetune" /
                                                 "gaitbase_ft_multisession_best_iter1200.pt"))
    ap.add_argument("--class-num", type=int, default=13)
    ap.add_argument("--multisession-root", default=str(REPO / "data" / "pkl_multisession"))
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device} | checkpoint={Path(args.checkpoint).name}")

    t0 = time.time()
    model = GaitBaseInfer(class_num=args.class_num).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[info] pesos cargados en {time.time()-t0:.1f}s | "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    # Cargar todas las pkls e indexar por (subj, cond)
    ms_root = Path(args.multisession_root)
    pkls = collect_pkls(ms_root)
    subjects = sorted({s for s, _ in pkls.keys()})
    print(f"[info] {len(pkls)} pkls totales, {len(subjects)} sujetos")

    # Embeddings normalizados (256*16,) por (subj, cond)
    t0 = time.time()
    emb_db: Dict[Tuple[str, str], np.ndarray] = {}
    for key, p in pkls.items():
        sils = load_silhouettes(p)
        emb_db[key] = extract_emb(model, sils, device)
    print(f"[info] embeddings listos en {time.time()-t0:.1f}s")

    # --- LCO eval ---
    genuine_all: List[Dict] = []
    impostor_all: List[Dict] = []
    by_protocol: Dict[str, Tuple[List[Dict], List[Dict]]] = {}

    for (subj_p, cond_p), probe in emb_db.items():
        # Gallery efectiva genuino: TODOS los demás (subj, cond) en emb_db
        # Impostor: TODOS los (subj!=subj_p, cond) en emb_db
        # Compute max sim per gallery subject under both regimes.

        # Genuino: incluye subj_p (con sus K-1 templates) y los demás sujetos completos
        sims_per_subj_gen: Dict[str, float] = {}
        for s in subjects:
            sims = []
            for (sj, cj), e in emb_db.items():
                if sj != s:
                    continue
                if sj == subj_p and cj == cond_p:
                    continue  # excluir la propia pkl del probe
                sims.append(float(e @ probe))
            if sims:
                sims_per_subj_gen[s] = max(sims)
        if not sims_per_subj_gen:
            continue
        top1_gen = max(sims_per_subj_gen, key=sims_per_subj_gen.get)
        genuine_all.append({
            "protocol": cond_p,
            "kind": "genuine",
            "probe": subj_p,
            "probe_cond": cond_p,
            "top1": top1_gen,
            "sim_top1": round(sims_per_subj_gen[top1_gen], 6),
            "correct_top1": (top1_gen == subj_p),
        })

        # Impostor: gallery sin ningún template de subj_p
        sims_per_subj_imp: Dict[str, float] = {}
        for s in subjects:
            if s == subj_p:
                continue
            sims = [float(e @ probe) for (sj, _), e in emb_db.items() if sj == s]
            if sims:
                sims_per_subj_imp[s] = max(sims)
        if not sims_per_subj_imp:
            continue
        top1_imp = max(sims_per_subj_imp, key=sims_per_subj_imp.get)
        impostor_all.append({
            "protocol": cond_p,
            "kind": "impostor",
            "probe": subj_p,
            "probe_cond": cond_p,
            "top1": top1_imp,
            "sim_top1": round(sims_per_subj_imp[top1_imp], 6),
            "correct_top1": False,
        })

    # Agrupar por protocolo (cond del probe)
    for cond in sorted({p["probe_cond"] for p in genuine_all}):
        gs = [p for p in genuine_all if p["probe_cond"] == cond]
        ims = [p for p in impostor_all if p["probe_cond"] == cond]
        by_protocol[cond] = (gs, ims)

    # --- Resultados ---
    results = {
        "model": "GaitBase",
        "checkpoint": Path(args.checkpoint).name,
        "matching": "multi-template + max-sim",
        "protocol": "Leave-Condition-Out (LCO)",
        "n_subjects": len(subjects),
        "n_pkls": len(pkls),
        "protocols": {},
    }

    print("\n=== Open-set LCO multi-template ===")
    for cond, (gs, ims) in by_protocol.items():
        r = sweep_tau(gs, ims)
        results["protocols"][cond] = r
        op = r["op_points"]
        print(f"  {cond:10s} n_g={r['n_genuine']:3d} n_i={r['n_impostor']:3d}  "
              f"EER={r['EER']*100:5.2f}%  "
              f"tau@FAR=0%={op['FAR=0%']['tau']:.4f} (TAR={op['FAR=0%']['TAR']*100:5.1f}%)  "
              f"rank1={r['rank1_top1_correct']*100:5.1f}%  gap={r['gap_mean']:+.4f}")

    r_all = sweep_tau(genuine_all, impostor_all)
    results["protocols"]["ALL"] = r_all
    op = r_all["op_points"]
    print(f"  {'ALL':10s} n_g={r_all['n_genuine']:3d} n_i={r_all['n_impostor']:3d}  "
          f"EER={r_all['EER']*100:5.2f}%  "
          f"tau@FAR=0%={op['FAR=0%']['tau']:.4f} (TAR={op['FAR=0%']['TAR']*100:5.1f}%)  "
          f"rank1={r_all['rank1_top1_correct']*100:5.1f}%  gap={r_all['gap_mean']:+.4f}")

    # Recomendación de τ producción
    tau_prod = op["FAR=0%"]["tau"]
    print(f"\n[reco] tau produccion = {tau_prod:.4f}  (FAR=0%, TAR={op['FAR=0%']['TAR']*100:.1f}%, "
          f"protocolo ALL)")

    # JSON sin curve (mucha data) en versión compacta
    compact = {**results}
    compact["protocols"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "curve"}
        for k, v in results["protocols"].items()
    }
    out_json = REPORTS / "15_openset_multitemplate.json"
    out_json.write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out_json.relative_to(REPO)}")

    # CSV pares
    out_csv = REPORTS / "15_openset_multitemplate_pairs.csv"
    cols = ["protocol", "kind", "probe", "probe_cond", "top1", "sim_top1", "correct_top1"]
    lines = [",".join(cols)]
    for row in genuine_all + impostor_all:
        lines.append(",".join(str(row[c]) for c in cols))
    out_csv.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_csv.relative_to(REPO)}")

    # ROC con todas las curvas
    out_png = REPORTS / "15_openset_multitemplate_roc.png"
    if try_plot_roc(results["protocols"], out_png):
        print(f"-> {out_png.relative_to(REPO)}")

    # Top-5 errores
    print("\n[detalle] impostores con sim_top1 más alta (top 5):")
    for row in sorted(impostor_all, key=lambda r: -r["sim_top1"])[:5]:
        print(f"  {row['probe_cond']:10s} probe={row['probe']:30s} -> top1={row['top1']:30s} "
              f"sim={row['sim_top1']:.4f}")

    print("\n[detalle] genuinos con sim_top1 más baja (top 5):")
    for row in sorted(genuine_all, key=lambda r: r["sim_top1"])[:5]:
        ok = "OK" if row["correct_top1"] else "ERR"
        print(f"  {row['probe_cond']:10s} probe={row['probe']:30s} -> top1={row['top1']:30s} "
              f"sim={row['sim_top1']:.4f} [{ok}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
