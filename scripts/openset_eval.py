"""Evaluación open-set (Fase 5).

Protocolo leave-one-subject-out (LOSO) sobre 19 sujetos cross-session:
  - Gallery: s1_normal (1 template por sujeto, 18 válidos; falta
    jesusantonioaguilarfelix/normal_s1).
  - Probes: s2_normal (NN) y s2_rapido (NR).
  - Para cada probe con identidad i:
      genuine:  query contra gallery completa -> match correcto esperado
                y sim_top1 >= τ (aceptar).
      impostor: query contra gallery sin i    -> cualquiera sea top1,
                sim_top1 debería ser < τ (rechazar).

Métricas por τ:
  - TAR = (genuine_correct_top1 AND sim>=τ) / n_genuine
  - FAR =  sim_top1_impostor >= τ           / n_impostor
  - FRR = 1 - TAR

Operating points:
  - EER: FAR == FRR
  - τ @ FAR=1% (si alcanzable), τ @ FAR=5%, τ @ FAR=0%

Uso:
    python scripts/openset_eval.py \\
        --checkpoint checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt \\
        --tag ft_multisession_iter1200 --class-num 13

Salidas:
    reports/06_openset_<tag>.json
    reports/06_openset_<tag>_pairs.csv
    reports/06_openset_<tag>_roc.png (si matplotlib)
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
    def __init__(self, class_num: int = 3000) -> None:
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
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{pkl_path}: esperaba ndarray, obtuve {type(arr)}")
    if arr.ndim != 3 or arr.shape[1:] != (64, 44):
        raise ValueError(f"{pkl_path}: shape inesperada {arr.shape}")
    return arr


@torch.no_grad()
def extract_embedding(model: GaitBaseInfer, sils_np: np.ndarray, device: torch.device) -> np.ndarray:
    sils = torch.from_numpy(sils_np.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    seqL = [torch.tensor([sils.shape[1]], device=device)]
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        emb = model(sils, seqL)
    return emb.float().squeeze(0).cpu().numpy().reshape(-1)


def load_pkls(root: Path, conditions: Tuple[str, ...]) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue
        for cond in conditions:
            seq_dir = subj_dir / cond / "090"
            if not seq_dir.is_dir():
                continue
            pkls = sorted(seq_dir.glob("*.pkl"))
            if not pkls:
                continue
            out[(subj_dir.name, cond)] = pkls[0]
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_pairs(
    gallery: Dict[str, np.ndarray],
    probes: Dict[str, np.ndarray],
    protocol: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Construye listas de genuinos e impostores LOSO.

    gallery: {subject -> emb_s1_normal}
    probes:  {subject -> emb_s2_<cond>}
    """
    gallery_subjects = sorted(gallery.keys())
    probe_subjects = sorted(probes.keys())
    G = np.stack([gallery[s] for s in gallery_subjects], axis=0)
    G_norm = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)

    genuine: List[Dict] = []
    impostor: List[Dict] = []

    for pname in probe_subjects:
        q = probes[pname]
        qn = q / (np.linalg.norm(q) + 1e-12)
        sims = G_norm @ qn  # (n_gallery,)

        # --- GENUINE: gallery completa ---
        if pname in gallery:
            order = np.argsort(-sims)
            top1_idx = int(order[0])
            top1_name = gallery_subjects[top1_idx]
            top1_sim = float(sims[top1_idx])
            correct_top1 = (top1_name == pname)
            genuine.append({
                "protocol": protocol,
                "kind": "genuine",
                "probe": pname,
                "top1": top1_name,
                "top1_sim": round(top1_sim, 6),
                "correct_top1": correct_top1,
            })

        # --- IMPOSTOR: gallery - self ---
        if pname in gallery:
            mask = np.array([g != pname for g in gallery_subjects])
        else:
            # probe no estaba en gallery; es impostor natural
            mask = np.ones(len(gallery_subjects), dtype=bool)
        sims_imp = sims.copy()
        sims_imp[~mask] = -np.inf
        top1_idx = int(np.argmax(sims_imp))
        top1_sim = float(sims_imp[top1_idx])
        impostor.append({
            "protocol": protocol,
            "kind": "impostor",
            "probe": pname,
            "top1": gallery_subjects[top1_idx],
            "top1_sim": round(top1_sim, 6),
            "correct_top1": False,  # por construcción
        })

    return genuine, impostor


def sweep_tau(
    genuine: List[Dict],
    impostor: List[Dict],
    n_taus: int = 2001,
) -> Dict:
    """Barrido de τ y cálculo TAR/FAR/FRR.

    TAR = (correct_top1 AND sim>=τ) / n_genuine
    FAR = (sim>=τ) / n_impostor
    """
    g_sim = np.array([p["top1_sim"] for p in genuine])
    g_ok = np.array([p["correct_top1"] for p in genuine], dtype=bool)
    i_sim = np.array([p["top1_sim"] for p in impostor])

    lo = float(min(g_sim.min(), i_sim.min())) - 1e-4
    hi = float(max(g_sim.max(), i_sim.max())) + 1e-4
    taus = np.linspace(lo, hi, n_taus)

    tar = np.array([(g_ok & (g_sim >= t)).sum() / max(1, len(g_sim)) for t in taus])
    far = np.array([(i_sim >= t).sum() / max(1, len(i_sim)) for t in taus])
    frr = 1.0 - tar

    # EER: punto donde |FAR - FRR| es mínimo
    eer_idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[eer_idx] + frr[eer_idx]) / 2.0)
    tau_eer = float(taus[eer_idx])

    def tau_at_far(target_far: float) -> Tuple[float, float, float]:
        """Menor τ con FAR <= target_far. Devuelve (τ, FAR, TAR)."""
        valid = np.where(far <= target_far)[0]
        if len(valid) == 0:
            return float("nan"), float("nan"), float("nan")
        idx = int(valid[0])  # menor τ que cumple (el array está en τ crecientes)
        return float(taus[idx]), float(far[idx]), float(tar[idx])

    tau_far0, far0, tar_far0 = tau_at_far(0.0)
    tau_far1, far1, tar_far1 = tau_at_far(0.01)
    tau_far5, far5, tar_far5 = tau_at_far(0.05)

    return {
        "n_genuine": int(len(g_sim)),
        "n_impostor": int(len(i_sim)),
        "genuine_sim_stats": {
            "min": round(float(g_sim.min()), 4),
            "max": round(float(g_sim.max()), 4),
            "mean": round(float(g_sim.mean()), 4),
            "std": round(float(g_sim.std()), 4),
        },
        "impostor_sim_stats": {
            "min": round(float(i_sim.min()), 4),
            "max": round(float(i_sim.max()), 4),
            "mean": round(float(i_sim.mean()), 4),
            "std": round(float(i_sim.std()), 4),
        },
        "gap_mean": round(float(g_sim.mean() - i_sim.mean()), 4),
        "EER": round(eer, 4),
        "tau_EER": round(tau_eer, 4),
        "rank1_top1_correct": round(float(g_ok.mean()), 4),
        "op_points": {
            "FAR=0%":  {"tau": round(tau_far0, 4), "FAR": round(far0, 4), "TAR": round(tar_far0, 4)},
            "FAR<=1%": {"tau": round(tau_far1, 4), "FAR": round(far1, 4), "TAR": round(tar_far1, 4)},
            "FAR<=5%": {"tau": round(tau_far5, 4), "FAR": round(far5, 4), "TAR": round(tar_far5, 4)},
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
        ax.plot(r["curve"]["FAR"], r["curve"]["TAR"], label=f"{name} (EER={r['EER']*100:.1f}%)")
    ax.set_xlabel("FAR (False Accept Rate)")
    ax.set_ylabel("TAR (True Accept Rate)")
    ax.set_title("Open-set ROC (LOSO, 19 sujetos cross-session)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gallery-root", default=str(REPO / "data" / "pkl"))
    ap.add_argument("--probe-root", default=str(REPO / "data" / "pkl_s2"))
    ap.add_argument("--class-num", type=int, default=3000)
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device} | checkpoint={Path(args.checkpoint).name} | tag={args.tag}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model = GaitBaseInfer(class_num=args.class_num).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[info] pesos cargados en {time.time()-t0:.1f}s | missing={len(missing)} unexpected={len(unexpected)}")

    gallery_root = Path(args.gallery_root)
    probe_root = Path(args.probe_root)
    g_pkls = load_pkls(gallery_root, ("normal",))
    p_pkls = load_pkls(probe_root, ("normal", "rapido"))
    print(f"[info] gallery normal (s1): {len(g_pkls)} | probes s2 (n+r): {len(p_pkls)}")

    t0 = time.time()
    gallery_emb: Dict[str, np.ndarray] = {}
    probes_n: Dict[str, np.ndarray] = {}
    probes_r: Dict[str, np.ndarray] = {}
    for (subj, _cond), path in g_pkls.items():
        gallery_emb[subj] = extract_embedding(model, load_silhouettes(path), device)
    for (subj, cond), path in p_pkls.items():
        emb = extract_embedding(model, load_silhouettes(path), device)
        if cond == "normal":
            probes_n[subj] = emb
        elif cond == "rapido":
            probes_r[subj] = emb
    dt = time.time() - t0
    print(f"[info] embeddings listos en {dt:.1f}s "
          f"(gallery={len(gallery_emb)}, probes_n={len(probes_n)}, probes_r={len(probes_r)})")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[info] VRAM pico: {peak:.0f} MB")

    # --- Pares por protocolo ---
    g_nn, i_nn = build_pairs(gallery_emb, probes_n, "NN")
    g_nr, i_nr = build_pairs(gallery_emb, probes_r, "NR")

    # --- Por protocolo individual ---
    res_nn = sweep_tau(g_nn, i_nn)
    res_nr = sweep_tau(g_nr, i_nr)

    # --- Combinado (NN + NR) ---
    res_all = sweep_tau(g_nn + g_nr, i_nn + i_nr)

    def summary_line(name: str, r: Dict) -> str:
        op = r["op_points"]
        return (f"  {name:5s}  EER={r['EER']*100:5.2f}%  "
                f"TAR@FAR<=1%={op['FAR<=1%']['TAR']*100:5.2f}% (tau={op['FAR<=1%']['tau']:.4f})  "
                f"TAR@FAR<=5%={op['FAR<=5%']['TAR']*100:5.2f}% (tau={op['FAR<=5%']['tau']:.4f})  "
                f"gap_means={r['gap_mean']:+.4f}")

    print("\n=== Open-set LOSO ===")
    print(summary_line("NN", res_nn))
    print(summary_line("NR", res_nr))
    print(summary_line("ALL", res_all))

    results = {
        "model": "GaitBase",
        "checkpoint": Path(args.checkpoint).name,
        "tag": args.tag,
        "gallery_root": str(gallery_root),
        "probe_root": str(probe_root),
        "n_gallery": len(gallery_emb),
        "n_probes_NN": len(probes_n),
        "n_probes_NR": len(probes_r),
        "protocols": {
            "NN": res_nn,
            "NR": res_nr,
            "ALL": res_all,
        },
    }

    out_json = REPORTS / f"06_openset_{args.tag}.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out_json.relative_to(REPO)}")

    # CSV de pares
    out_csv = REPORTS / f"06_openset_{args.tag}_pairs.csv"
    cols = ["protocol", "kind", "probe", "top1", "top1_sim", "correct_top1"]
    lines = [",".join(cols)]
    for row in g_nn + i_nn + g_nr + i_nr:
        lines.append(",".join(str(row[c]) for c in cols))
    out_csv.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_csv.relative_to(REPO)}")

    # ROC PNG
    out_png = REPORTS / f"06_openset_{args.tag}_roc.png"
    if try_plot_roc({"NN": res_nn, "NR": res_nr, "ALL": res_all}, out_png):
        print(f"-> {out_png.relative_to(REPO)}")

    # Detalle pares con riesgo (impostores con sim alta, genuinos con sim baja)
    print("\n[detalle] impostores con sim_top1 más alta (top 5, protocolo ALL):")
    for row in sorted(i_nn + i_nr, key=lambda r: -r["top1_sim"])[:5]:
        print(f"  {row['protocol']} probe={row['probe']:30s} -> top1={row['top1']:30s} sim={row['top1_sim']:.4f}")

    print("\n[detalle] genuinos con sim_top1 más baja (top 5, protocolo ALL):")
    for row in sorted(g_nn + g_nr, key=lambda r: r["top1_sim"])[:5]:
        ok = "OK" if row["correct_top1"] else "ERR"
        print(f"  {row['protocol']} probe={row['probe']:30s} -> top1={row['top1']:30s} sim={row['top1_sim']:.4f} [{ok}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
