"""Evaluación open-set con SkeletonGait++ (Fase 4 → Fase 5).

Protocolo LOSO cross-session idéntico al de openset_eval.py pero usando
embeddings multimodales del modelo SkeletonGait++.

Gallery:  pkl_multimodal/<subj>/normal_s1/090/  (19 sujetos − 1 faltante)
Probes NN: pkl_multimodal/<subj>/normal_s2/090/
Probes NR: pkl_multimodal/<subj>/rapido_s2/090/

Uso:
    python scripts/openset_eval_skeletongaitpp.py \\
        --checkpoint checkpoints/finetune/skeletongaitpp_best_iter800.pt \\
        --tag skeletongaitpp_iter800
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
OPENGAIT = REPO / "third_party" / "OpenGait"
MULTIMODAL_ROOT = REPO / "data" / "pkl_multimodal"
REPORTS = REPO / "reports"

sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))

SIL_H, SIL_W = 64, 44
CUTTING = (SIL_H - SIL_W) // 2  # = 10

MODEL_CFG = {
    "Backbone": {"in_channels": 3, "blocks": [1, 4, 4, 1], "C": 2},
    "SeparateBNNecks": {"class_num": 13},
    "use_emb2": False,
}


# ───────────────────────── modelo ─────────────────────────────────

def _load_skeletongait_pp_class() -> type:
    import importlib, importlib.util
    importlib.import_module("modeling")
    importlib.import_module("modeling.models")
    mod_name = "modeling.models.skeletongaitpp"
    if mod_name not in sys.modules:
        model_file = OPENGAIT / "opengait" / "modeling" / "models" / "skeletongait++.py"
        spec = importlib.util.spec_from_file_location(mod_name, str(model_file))
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "modeling.models"
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name].SkeletonGaitPP


def build_model(checkpoint: Path, device: torch.device) -> nn.Module:
    SkeletonGaitPP = _load_skeletongait_pp_class()
    model = SkeletonGaitPP.__new__(SkeletonGaitPP)
    nn.Module.__init__(model)
    model.training = True
    model.build_network(MODEL_CFG)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"[ckpt] cargado {checkpoint.name}  missing={len(missing)}  unexpected={len(unexpected)}")
    return model.to(device).eval()


# ───────────────────────── embedding ──────────────────────────────

@torch.no_grad()
def _embed(model: nn.Module, seq_dir: Path, device: torch.device) -> Optional[np.ndarray]:
    hm_path = seq_dir / "0_heatmap.pkl"
    sil_path = seq_dir / "1_sil.pkl"
    if not hm_path.exists() or not sil_path.exists():
        return None
    with hm_path.open("rb") as f:
        hm = pickle.load(f)
    with sil_path.open("rb") as f:
        sil = pickle.load(f)

    h_crop = hm[:, :, :, CUTTING:-CUTTING].astype(np.float32) / 255.0
    s_exp = sil[:, np.newaxis, :, :].astype(np.float32) / 255.0
    combined = np.concatenate([h_crop, s_exp], axis=1)
    x = torch.from_numpy(combined).unsqueeze(0).to(device)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        retval = model(([x], None, None, None, None))
    embed = retval["inference_feat"]["embeddings"]
    return embed.float().squeeze(0).cpu().numpy().reshape(-1)


def collect_embeddings(model: nn.Module, device: torch.device) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]
]:
    """Devuelve gallery (normal_s1), probes_nn (normal_s2), probes_nr (rapido_s2)."""
    gallery: Dict[str, np.ndarray] = {}
    probes_nn: Dict[str, np.ndarray] = {}
    probes_nr: Dict[str, np.ndarray] = {}

    for subj_dir in sorted(MULTIMODAL_ROOT.iterdir()):
        if not subj_dir.is_dir():
            continue
        s = subj_dir.name

        e = _embed(model, subj_dir / "normal_s1" / "090", device)
        if e is not None:
            gallery[s] = e

        e = _embed(model, subj_dir / "normal_s2" / "090", device)
        if e is not None:
            probes_nn[s] = e

        e = _embed(model, subj_dir / "rapido_s2" / "090", device)
        if e is not None:
            probes_nr[s] = e

    return gallery, probes_nn, probes_nr


# ───────────────────────── pares y métricas ────────────────────────

def build_pairs(
    gallery: Dict[str, np.ndarray],
    probes: Dict[str, np.ndarray],
    protocol: str,
) -> Tuple[List[Dict], List[Dict]]:
    gallery_subjects = sorted(gallery.keys())
    G = np.stack([gallery[s] for s in gallery_subjects], axis=0)
    G_norm = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)

    genuine: List[Dict] = []
    impostor: List[Dict] = []

    for pname in sorted(probes.keys()):
        q = probes[pname]
        qn = q / (np.linalg.norm(q) + 1e-12)
        sims = G_norm @ qn

        if pname in gallery:
            order = np.argsort(-sims)
            top1_idx = int(order[0])
            genuine.append({
                "protocol": protocol, "kind": "genuine",
                "probe": pname, "top1": gallery_subjects[top1_idx],
                "top1_sim": round(float(sims[top1_idx]), 6),
                "correct_top1": gallery_subjects[top1_idx] == pname,
            })

        mask = np.array([g != pname for g in gallery_subjects]) if pname in gallery \
               else np.ones(len(gallery_subjects), dtype=bool)
        sims_imp = sims.copy()
        sims_imp[~mask] = -np.inf
        top1_idx = int(np.argmax(sims_imp))
        impostor.append({
            "protocol": protocol, "kind": "impostor",
            "probe": pname, "top1": gallery_subjects[top1_idx],
            "top1_sim": round(float(sims_imp[top1_idx]), 6),
            "correct_top1": False,
        })

    return genuine, impostor


def sweep_tau(genuine: List[Dict], impostor: List[Dict], n_taus: int = 2001) -> Dict:
    g_sim = np.array([p["top1_sim"] for p in genuine])
    g_ok = np.array([p["correct_top1"] for p in genuine], dtype=bool)
    i_sim = np.array([p["top1_sim"] for p in impostor])

    lo = float(min(g_sim.min(), i_sim.min())) - 1e-4
    hi = float(max(g_sim.max(), i_sim.max())) + 1e-4
    taus = np.linspace(lo, hi, n_taus)

    tar = np.array([(g_ok & (g_sim >= t)).sum() / max(1, len(g_sim)) for t in taus])
    far = np.array([(i_sim >= t).sum() / max(1, len(i_sim)) for t in taus])
    frr = 1.0 - tar

    eer_idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[eer_idx] + frr[eer_idx]) / 2.0)

    def tau_at_far(target: float) -> Tuple[float, float, float]:
        valid = np.where(far <= target)[0]
        if len(valid) == 0:
            return float("nan"), float("nan"), float("nan")
        idx = int(valid[0])
        return float(taus[idx]), float(far[idx]), float(tar[idx])

    tau0, far0, tar0 = tau_at_far(0.0)
    tau1, far1, tar1 = tau_at_far(0.01)
    tau5, far5, tar5 = tau_at_far(0.05)

    return {
        "n_genuine": int(len(g_sim)),
        "n_impostor": int(len(i_sim)),
        "genuine_sim_stats": {k: round(float(v), 4) for k, v in
                              zip(("min","max","mean","std"),
                                  (g_sim.min(), g_sim.max(), g_sim.mean(), g_sim.std()))},
        "impostor_sim_stats": {k: round(float(v), 4) for k, v in
                               zip(("min","max","mean","std"),
                                   (i_sim.min(), i_sim.max(), i_sim.mean(), i_sim.std()))},
        "gap_mean": round(float(g_sim.mean() - i_sim.mean()), 4),
        "EER": round(eer, 4),
        "tau_EER": round(float(taus[eer_idx]), 4),
        "rank1_top1_correct": round(float(g_ok.mean()), 4),
        "op_points": {
            "FAR=0%":  {"tau": round(tau0,4), "FAR": round(far0,4), "TAR": round(tar0,4)},
            "FAR<=1%": {"tau": round(tau1,4), "FAR": round(far1,4), "TAR": round(tar1,4)},
            "FAR<=5%": {"tau": round(tau5,4), "FAR": round(far5,4), "TAR": round(tar5,4)},
        },
    }


def try_plot_roc(curves: Dict[str, Dict], out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in curves.items():
        # rebuild curve arrays from op_points for a clean plot
        pass
    # Load full results for curve data
    ax.set_xlabel("FAR")
    ax.set_ylabel("TAR")
    ax.set_title("Open-set ROC — SkeletonGait++ (LOSO, cross-session)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


# ───────────────────────── main ───────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ckpt_path = Path(args.checkpoint)
    model = build_model(ckpt_path, device)

    print("[info] extrayendo embeddings de pkl_multimodal...")
    t0 = time.time()
    gallery, probes_nn, probes_nr = collect_embeddings(model, device)
    print(f"[info] gallery={len(gallery)}  probes_NN={len(probes_nn)}  probes_NR={len(probes_nr)}  "
          f"({time.time()-t0:.1f}s)")

    if device.type == "cuda":
        print(f"[info] VRAM pico: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")

    g_nn, i_nn = build_pairs(gallery, probes_nn, "NN")
    g_nr, i_nr = build_pairs(gallery, probes_nr, "NR")

    res_nn  = sweep_tau(g_nn, i_nn)
    res_nr  = sweep_tau(g_nr, i_nr)
    res_all = sweep_tau(g_nn + g_nr, i_nn + i_nr)

    def line(name: str, r: Dict) -> str:
        op = r["op_points"]
        return (f"  {name:5s}  EER={r['EER']*100:5.2f}%  "
                f"TAR@FAR=0%={op['FAR=0%']['TAR']*100:5.1f}% (tau={op['FAR=0%']['tau']:.4f})  "
                f"TAR@FAR<=1%={op['FAR<=1%']['TAR']*100:5.1f}%  "
                f"gap={r['gap_mean']:+.4f}")

    print("\n=== Open-set LOSO — SkeletonGait++ ===")
    print(line("NN", res_nn))
    print(line("NR", res_nr))
    print(line("ALL", res_all))

    print("\n[detalle] impostores con sim_top1 más alta (top 5):")
    for row in sorted(i_nn + i_nr, key=lambda r: -r["top1_sim"])[:5]:
        print(f"  {row['protocol']} {row['probe']:30s} -> {row['top1']:30s} sim={row['top1_sim']:.4f}")

    print("\n[detalle] genuinos con sim_top1 más baja (top 5):")
    for row in sorted(g_nn + g_nr, key=lambda r: r["top1_sim"])[:5]:
        flag = "OK" if row["correct_top1"] else "ERR"
        print(f"  {row['protocol']} {row['probe']:30s} -> {row['top1']:30s} sim={row['top1_sim']:.4f} [{flag}]")

    results = {
        "model": "SkeletonGaitPP",
        "checkpoint": ckpt_path.name,
        "tag": args.tag,
        "multimodal_root": str(MULTIMODAL_ROOT),
        "n_gallery": len(gallery),
        "n_probes_NN": len(probes_nn),
        "n_probes_NR": len(probes_nr),
        "protocols": {"NN": res_nn, "NR": res_nr, "ALL": res_all},
    }

    out_json = REPORTS / f"10_openset_{args.tag}.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out_json.relative_to(REPO)}")

    out_csv = REPORTS / f"10_openset_{args.tag}_pairs.csv"
    cols = ["protocol","kind","probe","top1","top1_sim","correct_top1"]
    lines_csv = [",".join(cols)]
    for row in g_nn + i_nn + g_nr + i_nr:
        lines_csv.append(",".join(str(row[c]) for c in cols))
    out_csv.write_text("\n".join(lines_csv), encoding="utf-8")
    print(f"-> {out_csv.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
