"""Evaluación open-set fusión tardía GaitBase + SkeletonGait++ (Fase 4).

Combina las similitudes coseno de ambos modelos:
  score(q,g) = α * sim_gaitbase(q,g) + (1-α) * sim_sgpp(q,g)

Se evalúan α ∈ {0.0, 0.1, …, 1.0} y se reporta el mejor EER y TAR@FAR=0%.

GaitBase: lee data/pkl/ (gallery) y data/pkl_s2/ (probes)  — seq00.pkl single-channel
SkeletonGait++: lee data/pkl_multimodal/ — 0_heatmap.pkl + 1_sil.pkl

Uso:
    python scripts/openset_eval_fusion.py
"""

from __future__ import annotations

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
REPORTS = REPO / "reports"

sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))

PKL_S1 = REPO / "data" / "pkl"
PKL_S2 = REPO / "data" / "pkl_s2"
MULTIMODAL = REPO / "data" / "pkl_multimodal"

CKPT_GB = REPO / "checkpoints" / "finetune" / "gaitbase_ft_multisession_best_iter1200.pt"
CKPT_SGPP = REPO / "checkpoints" / "finetune" / "skeletongaitpp_best_iter800.pt"

SIL_H, SIL_W = 64, 44
CUTTING = (SIL_H - SIL_W) // 2

SGPP_CFG = {
    "Backbone": {"in_channels": 3, "blocks": [1, 4, 4, 1], "C": 2},
    "SeparateBNNecks": {"class_num": 13},
    "use_emb2": False,
}


# ─────────────── GaitBase ────────────────────────────────────────

def _build_gaitbase(device: torch.device) -> nn.Module:
    from modeling.backbones.resnet import ResNet9
    from modeling.modules import (
        SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
        SeparateFCs, SeparateBNNecks,
    )

    class GB(nn.Module):
        def __init__(self):
            super().__init__()
            self.Backbone = SetBlockWrapper(ResNet9(
                block="BasicBlock", channels=[64, 128, 256, 512],
                in_channel=1, layers=[1, 1, 1, 1],
                strides=[1, 2, 2, 1], maxpool=False,
            ))
            self.FCs = SeparateFCs(in_channels=512, out_channels=256, parts_num=16)
            self.BNNecks = SeparateBNNecks(class_num=13, in_channels=256, parts_num=16)
            self.TP = PackSequenceWrapper(torch.max)
            self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        def forward(self, sils, seqL):
            if sils.dim() == 4:
                sils = sils.unsqueeze(1)
            outs = self.Backbone(sils)
            outs = self.TP(outs, seqL, options={"dim": 2})[0]
            return self.FCs(self.HPP(outs))

    model = GB()
    ckpt = torch.load(CKPT_GB, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def _embed_gb(model: nn.Module, pkl_path: Path, device: torch.device) -> Optional[np.ndarray]:
    if not pkl_path.exists():
        return None
    with pkl_path.open("rb") as f:
        sils = pickle.load(f)
    x = torch.from_numpy(sils.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    seqL = [torch.tensor([x.shape[1]], device=device)]
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        emb = model(x, seqL)
    return emb.float().squeeze(0).cpu().numpy().reshape(-1)


# ─────────────── SkeletonGait++ ──────────────────────────────────

def _load_sgpp_class() -> type:
    import importlib, importlib.util
    importlib.import_module("modeling")
    importlib.import_module("modeling.models")
    mod_name = "modeling.models.skeletongaitpp"
    if mod_name not in sys.modules:
        f = OPENGAIT / "opengait" / "modeling" / "models" / "skeletongait++.py"
        spec = importlib.util.spec_from_file_location(mod_name, str(f))
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "modeling.models"
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name].SkeletonGaitPP


def _build_sgpp(device: torch.device) -> nn.Module:
    cls = _load_sgpp_class()
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.training = True
    model.build_network(SGPP_CFG)
    ckpt = torch.load(CKPT_SGPP, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    return model.to(device).eval()


@torch.no_grad()
def _embed_sgpp(model: nn.Module, seq_dir: Path, device: torch.device) -> Optional[np.ndarray]:
    hm_p = seq_dir / "0_heatmap.pkl"
    sil_p = seq_dir / "1_sil.pkl"
    if not hm_p.exists() or not sil_p.exists():
        return None
    with hm_p.open("rb") as f:
        hm = pickle.load(f)
    with sil_p.open("rb") as f:
        sil = pickle.load(f)
    h = hm[:, :, :, CUTTING:-CUTTING].astype(np.float32) / 255.0
    s = sil[:, np.newaxis, :, :].astype(np.float32) / 255.0
    x = torch.from_numpy(np.concatenate([h, s], axis=1)).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        retval = model(([x], None, None, None, None))
    e = retval["inference_feat"]["embeddings"]
    return e.float().squeeze(0).cpu().numpy().reshape(-1)


# ─────────────── open-set mechanics ──────────────────────────────

def _norm(embs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v / (np.linalg.norm(v) + 1e-12) for k, v in embs.items()}


def _sim_matrix(probes: Dict[str, np.ndarray],
                gallery: Dict[str, np.ndarray]) -> Tuple[List[str], List[str], np.ndarray]:
    pnames = sorted(probes.keys())
    gnames = sorted(gallery.keys())
    P = np.stack([probes[p] for p in pnames])
    G = np.stack([gallery[g] for g in gnames])
    return pnames, gnames, P @ G.T


def sweep_tau(genuine: List[Dict], impostor: List[Dict]) -> Dict:
    g_sim = np.array([p["top1_sim"] for p in genuine])
    g_ok = np.array([p["correct_top1"] for p in genuine], dtype=bool)
    i_sim = np.array([p["top1_sim"] for p in impostor])
    lo = float(min(g_sim.min(), i_sim.min())) - 1e-4
    hi = float(max(g_sim.max(), i_sim.max())) + 1e-4
    taus = np.linspace(lo, hi, 2001)
    tar = np.array([(g_ok & (g_sim >= t)).sum() / max(1, len(g_sim)) for t in taus])
    far = np.array([(i_sim >= t).sum() / max(1, len(i_sim)) for t in taus])
    frr = 1.0 - tar
    eer_idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[eer_idx] + frr[eer_idx]) / 2.0)

    def at(target):
        valid = np.where(far <= target)[0]
        if len(valid) == 0:
            return float("nan"), float("nan"), float("nan")
        idx = int(valid[0])
        return float(taus[idx]), float(far[idx]), float(tar[idx])

    tau0, far0, tar0 = at(0.0)
    tau1, far1, tar1 = at(0.01)
    return {
        "EER": round(eer, 4),
        "tau_EER": round(float(taus[eer_idx]), 4),
        "TAR_FAR0": round(tar0, 4), "tau_FAR0": round(tau0, 4),
        "TAR_FAR1": round(tar1, 4), "tau_FAR1": round(tau1, 4),
        "gap_mean": round(float(g_sim.mean() - i_sim.mean()), 4),
        "rank1": round(float(g_ok.mean()), 4),
        "n_genuine": len(genuine), "n_impostor": len(impostor),
    }


def eval_protocol(
    sim_gb: np.ndarray, sim_sgpp: np.ndarray,
    pnames: List[str], gnames: List[str],
    protocol: str, alpha: float,
) -> Tuple[List[Dict], List[Dict]]:
    """Construye pares genuino/impostor con score fusionado."""
    sim = alpha * sim_gb + (1 - alpha) * sim_sgpp
    genuine, impostor = [], []
    for pi, pname in enumerate(pnames):
        sims_row = sim[pi]
        order = np.argsort(-sims_row)

        if pname in gnames:
            gi = gnames.index(pname)
            top1_idx = int(order[0])
            genuine.append({
                "protocol": protocol, "kind": "genuine", "probe": pname,
                "top1": gnames[top1_idx], "top1_sim": float(sims_row[top1_idx]),
                "correct_top1": gnames[top1_idx] == pname,
            })
            # impostor: gallery sin self
            mask = np.ones(len(gnames), dtype=bool)
            mask[gi] = False
            sims_imp = sims_row.copy()
            sims_imp[~mask] = -np.inf
            ti = int(np.argmax(sims_imp))
            impostor.append({
                "protocol": protocol, "kind": "impostor", "probe": pname,
                "top1": gnames[ti], "top1_sim": float(sims_imp[ti]),
                "correct_top1": False,
            })
        else:
            ti = int(order[0])
            impostor.append({
                "protocol": protocol, "kind": "impostor", "probe": pname,
                "top1": gnames[ti], "top1_sim": float(sims_row[ti]),
                "correct_top1": False,
            })
    return genuine, impostor


# ─────────────── main ────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("[info] cargando GaitBase...")
    gb = _build_gaitbase(device)

    print("[info] cargando SkeletonGait++...")
    sgpp = _build_sgpp(device)

    # ── Embeddings GaitBase (de pkl/ y pkl_s2/) ──
    print("[info] embeddings GaitBase...")
    t0 = time.time()
    gal_gb: Dict[str, np.ndarray] = {}
    pnn_gb: Dict[str, np.ndarray] = {}
    pnr_gb: Dict[str, np.ndarray] = {}

    for subj_dir in sorted(PKL_S1.iterdir()):
        if not subj_dir.is_dir():
            continue
        s = subj_dir.name
        p = subj_dir / "normal" / "090" / "seq00.pkl"
        e = _embed_gb(gb, p, device)
        if e is not None:
            gal_gb[s] = e

    for subj_dir in sorted(PKL_S2.iterdir()):
        if not subj_dir.is_dir():
            continue
        s = subj_dir.name
        for cond, store in (("normal", pnn_gb), ("rapido", pnr_gb)):
            p = subj_dir / cond / "090" / "seq00.pkl"
            e = _embed_gb(gb, p, device)
            if e is not None:
                store[s] = e

    print(f"[info] GB: gallery={len(gal_gb)} probes_NN={len(pnn_gb)} probes_NR={len(pnr_gb)}  ({time.time()-t0:.1f}s)")

    # ── Embeddings SkeletonGait++ (de pkl_multimodal/) ──
    print("[info] embeddings SkeletonGait++...")
    t0 = time.time()
    gal_sg: Dict[str, np.ndarray] = {}
    pnn_sg: Dict[str, np.ndarray] = {}
    pnr_sg: Dict[str, np.ndarray] = {}

    for subj_dir in sorted(MULTIMODAL.iterdir()):
        if not subj_dir.is_dir():
            continue
        s = subj_dir.name
        e = _embed_sgpp(sgpp, subj_dir / "normal_s1" / "090", device)
        if e is not None:
            gal_sg[s] = e
        e = _embed_sgpp(sgpp, subj_dir / "normal_s2" / "090", device)
        if e is not None:
            pnn_sg[s] = e
        e = _embed_sgpp(sgpp, subj_dir / "rapido_s2" / "090", device)
        if e is not None:
            pnr_sg[s] = e

    print(f"[info] SGPP: gallery={len(gal_sg)} probes_NN={len(pnn_sg)} probes_NR={len(pnr_sg)}  ({time.time()-t0:.1f}s)")

    # ── Alinear sujetos disponibles en ambos modelos ──
    all_subjects = sorted(set(gal_gb) & set(gal_sg))
    nn_subjects  = sorted(set(pnn_gb) & set(pnn_sg))
    nr_subjects  = sorted(set(pnr_gb) & set(pnr_sg))
    print(f"[info] sujetos coincidentes: gallery={len(all_subjects)} NN={len(nn_subjects)} NR={len(nr_subjects)}")

    gal_gb_a  = _norm({s: gal_gb[s] for s in all_subjects})
    gal_sg_a  = _norm({s: gal_sg[s] for s in all_subjects})
    pnn_gb_a  = _norm({s: pnn_gb[s] for s in nn_subjects})
    pnn_sg_a  = _norm({s: pnn_sg[s] for s in nn_subjects})
    pnr_gb_a  = _norm({s: pnr_gb[s] for s in nr_subjects})
    pnr_sg_a  = _norm({s: pnr_sg[s] for s in nr_subjects})

    gnames = sorted(all_subjects)
    pnames_nn = sorted(nn_subjects)
    pnames_nr = sorted(nr_subjects)

    G_gb  = np.stack([gal_gb_a[g] for g in gnames])
    G_sg  = np.stack([gal_sg_a[g] for g in gnames])
    PNN_gb = np.stack([pnn_gb_a[p] for p in pnames_nn])
    PNN_sg = np.stack([pnn_sg_a[p] for p in pnames_nn])
    PNR_gb = np.stack([pnr_gb_a[p] for p in pnames_nr])
    PNR_sg = np.stack([pnr_sg_a[p] for p in pnames_nr])

    sim_nn_gb = PNN_gb @ G_gb.T
    sim_nn_sg = PNN_sg @ G_sg.T
    sim_nr_gb = PNR_gb @ G_gb.T
    sim_nr_sg = PNR_sg @ G_sg.T

    # ── Barrido de alpha ──
    alphas = [round(a / 10, 1) for a in range(11)]
    print("\n  a(GB)  | EER_NN  EER_NR  EER_ALL | TAR@0%_NN TAR@0%_NR TAR@0%_ALL")
    print("  -------|" + "-" * 60)

    best = {"eer_all": 1.0, "alpha": None, "res": None}
    all_results = []

    for alpha in alphas:
        g_nn, i_nn = eval_protocol(sim_nn_gb, sim_nn_sg, pnames_nn, gnames, "NN", alpha)
        g_nr, i_nr = eval_protocol(sim_nr_gb, sim_nr_sg, pnames_nr, gnames, "NR", alpha)

        r_nn  = sweep_tau(g_nn, i_nn)
        r_nr  = sweep_tau(g_nr, i_nr)
        r_all = sweep_tau(g_nn + g_nr, i_nn + i_nr)

        row = {
            "alpha_gaitbase": alpha,
            "alpha_sgpp": round(1 - alpha, 1),
            "NN": r_nn, "NR": r_nr, "ALL": r_all,
        }
        all_results.append(row)

        print(f"  {alpha:.1f}      | "
              f"{r_nn['EER']*100:5.2f}%  {r_nr['EER']*100:5.2f}%  {r_all['EER']*100:5.2f}%  | "
              f"{r_nn['TAR_FAR0']*100:5.1f}%  {r_nr['TAR_FAR0']*100:5.1f}%  {r_all['TAR_FAR0']*100:5.1f}%")

        if r_all["EER"] < best["eer_all"]:
            best = {"eer_all": r_all["EER"], "alpha": alpha, "res": row}

    print(f"\n[best] a_GB={best['alpha']:.1f}  EER_ALL={best['eer_all']*100:.2f}%"
          f"  TAR@FAR=0%_ALL={best['res']['ALL']['TAR_FAR0']*100:.1f}%"
          f"  TAR@FAR=0%_NR={best['res']['NR']['TAR_FAR0']*100:.1f}%")

    # Detalle de errores con el mejor alpha
    alpha_best = best["alpha"]
    g_nn_b, i_nn_b = eval_protocol(sim_nn_gb, sim_nn_sg, pnames_nn, gnames, "NN", alpha_best)
    g_nr_b, i_nr_b = eval_protocol(sim_nr_gb, sim_nr_sg, pnames_nr, gnames, "NR", alpha_best)

    print(f"\n[detalle a={alpha_best}] genuinos con sim mas baja (top 5):")
    for row in sorted(g_nn_b + g_nr_b, key=lambda r: r["top1_sim"])[:5]:
        flag = "OK" if row["correct_top1"] else "ERR"
        print(f"  {row['protocol']} {row['probe']:28s} -> {row['top1']:28s} {row['top1_sim']:.4f} [{flag}]")

    print(f"[detalle a={alpha_best}] impostores con sim mas alta (top 5):")
    for row in sorted(i_nn_b + i_nr_b, key=lambda r: -r["top1_sim"])[:5]:
        print(f"  {row['protocol']} {row['probe']:28s} -> {row['top1']:28s} {row['top1_sim']:.4f}")

    # Guardar
    out = {
        "description": "Late fusion GaitBase + SkeletonGait++ (score averaging)",
        "checkpoints": {"gaitbase": CKPT_GB.name, "skeletongaitpp": CKPT_SGPP.name},
        "best_alpha_gaitbase": best["alpha"],
        "all_results": all_results,
    }
    out_json = REPORTS / "11_openset_fusion.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out_json.relative_to(REPO)}")


if __name__ == "__main__":
    main()
