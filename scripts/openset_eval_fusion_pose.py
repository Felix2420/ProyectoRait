"""Open-set fusion tardia GaitBase + ST-GCN ligero (Fase 4 — Ruta B).

Combina similitudes coseno de los dos modelos:
    score(q,g) = a * sim_gaitbase(q,g) + (1-a) * sim_stgcn(q,g)

Barre a in {0.0, 0.1, ..., 1.0} y reporta el mejor EER y TAR@FAR=0%.

Protocolo identico a openset_eval_fusion.py:
  Gallery: s1_normal (1 template/sujeto)
  Probes:  s2_normal (NN) y s2_rapido (NR)
  19 sujetos LOSO cross-session.

Uso:
    C:/Proyecto3/venv/Scripts/python.exe scripts/openset_eval_fusion_pose.py
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
OPENGAIT = REPO / "third_party" / "OpenGait"
REPORTS = REPO / "reports"

sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))
sys.path.insert(0, str(REPO))

from src.models.stgcn_lite import STGCNLite  # noqa: E402

PKL_S1 = REPO / "data" / "pkl"
PKL_S2 = REPO / "data" / "pkl_s2"
PROCESSED_S1 = REPO / "data" / "processed"
PROCESSED_S2 = REPO / "data" / "processed_s2"

CKPT_GB = REPO / "checkpoints" / "finetune" / "gaitbase_ft_multisession_best_iter1200.pt"
CKPT_STGCN = REPO / "checkpoints" / "finetune" / "stgcn_lite_best.pt"


# ───────────── GaitBase ───────────────────────────────────────────

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


# ───────────── ST-GCN lite ────────────────────────────────────────

def _build_stgcn(device: torch.device) -> Tuple[nn.Module, int]:
    ckpt = torch.load(CKPT_STGCN, map_location=device, weights_only=False)
    n_classes = ckpt["model"]["bnneck.fc.weight"].shape[0]
    emb_dim = ckpt.get("emb_dim", 128)
    win_t = ckpt.get("win_t", 30)
    model = STGCNLite(in_channels=3, num_nodes=17, emb_dim=emb_dim, n_classes=n_classes)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval(), win_t


@torch.no_grad()
def _embed_stgcn(model: nn.Module, kp_path: Path, win_t: int, device: torch.device) -> Optional[np.ndarray]:
    if not kp_path.exists():
        return None
    kp = np.load(kp_path).astype(np.float32)
    # crop temporal central; pad si hace falta
    T = kp.shape[0]
    if T < win_t:
        pad = win_t - T
        kp = np.concatenate([kp, np.zeros((pad, *kp.shape[1:]), dtype=kp.dtype)], axis=0)
    else:
        s = (T - win_t) // 2
        kp = kp[s:s + win_t]
    # low-conf dropout consistente con train
    mask = kp[..., 2] < 0.3
    kp[mask, :2] = 0.0
    x = torch.from_numpy(kp.transpose(2, 0, 1)).unsqueeze(0).to(device)  # (1,3,T,17)
    emb, _ = model(x)
    emb = F.normalize(emb, dim=1)
    return emb.squeeze(0).cpu().numpy()


# ───────────── open-set mechanics ─────────────────────────────────

def _norm(embs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v / (np.linalg.norm(v) + 1e-12) for k, v in embs.items()}


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
        "EER": round(eer, 4), "tau_EER": round(float(taus[eer_idx]), 4),
        "TAR_FAR0": round(tar0, 4), "tau_FAR0": round(tau0, 4),
        "TAR_FAR1": round(tar1, 4), "tau_FAR1": round(tau1, 4),
        "gap_mean": round(float(g_sim.mean() - i_sim.mean()), 4),
        "rank1": round(float(g_ok.mean()), 4),
        "n_genuine": len(genuine), "n_impostor": len(impostor),
    }


def eval_protocol(sim_gb, sim_pose, pnames, gnames, protocol, alpha):
    sim = alpha * sim_gb + (1 - alpha) * sim_pose
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


# ───────────── main ───────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("[info] cargando GaitBase...")
    gb = _build_gaitbase(device)

    print("[info] cargando ST-GCN lite...")
    stgcn, win_t = _build_stgcn(device)
    print(f"[info] win_t = {win_t}")

    # ── Embeddings GaitBase (siluetas) ──
    print("[info] embeddings GaitBase...")
    t0 = time.time()
    gal_gb: Dict[str, np.ndarray] = {}
    pnn_gb: Dict[str, np.ndarray] = {}
    pnr_gb: Dict[str, np.ndarray] = {}
    for subj_dir in sorted(PKL_S1.iterdir()):
        if not subj_dir.is_dir():
            continue
        s = subj_dir.name
        e = _embed_gb(gb, subj_dir / "normal" / "090" / "seq00.pkl", device)
        if e is not None:
            gal_gb[s] = e
    for subj_dir in sorted(PKL_S2.iterdir()):
        if not subj_dir.is_dir():
            continue
        s = subj_dir.name
        for cond, store in (("normal", pnn_gb), ("rapido", pnr_gb)):
            e = _embed_gb(gb, subj_dir / cond / "090" / "seq00.pkl", device)
            if e is not None:
                store[s] = e
    print(f"[info] GB: gal={len(gal_gb)} NN={len(pnn_gb)} NR={len(pnr_gb)}  ({time.time()-t0:.1f}s)")

    # ── Embeddings ST-GCN (keypoints) ──
    print("[info] embeddings ST-GCN...")
    t0 = time.time()
    gal_st: Dict[str, np.ndarray] = {}
    pnn_st: Dict[str, np.ndarray] = {}
    pnr_st: Dict[str, np.ndarray] = {}
    for subj_dir in sorted(PROCESSED_S1.iterdir()):
        if not subj_dir.is_dir():
            continue
        s = subj_dir.name
        e = _embed_stgcn(stgcn, subj_dir / "normal" / "keypoints.npy", win_t, device)
        if e is not None:
            gal_st[s] = e
    for subj_dir in sorted(PROCESSED_S2.iterdir()):
        if not subj_dir.is_dir():
            continue
        s = subj_dir.name
        for cond, store in (("normal", pnn_st), ("rapido", pnr_st)):
            e = _embed_stgcn(stgcn, subj_dir / cond / "keypoints.npy", win_t, device)
            if e is not None:
                store[s] = e
    print(f"[info] ST: gal={len(gal_st)} NN={len(pnn_st)} NR={len(pnr_st)}  ({time.time()-t0:.1f}s)")

    # ── Alinear sujetos coincidentes ──
    all_subjects = sorted(set(gal_gb) & set(gal_st))
    nn_subjects = sorted(set(pnn_gb) & set(pnn_st))
    nr_subjects = sorted(set(pnr_gb) & set(pnr_st))
    print(f"[info] coincidentes  gal={len(all_subjects)}  NN={len(nn_subjects)}  NR={len(nr_subjects)}")

    gal_gb_a = _norm({s: gal_gb[s] for s in all_subjects})
    gal_st_a = _norm({s: gal_st[s] for s in all_subjects})
    pnn_gb_a = _norm({s: pnn_gb[s] for s in nn_subjects})
    pnn_st_a = _norm({s: pnn_st[s] for s in nn_subjects})
    pnr_gb_a = _norm({s: pnr_gb[s] for s in nr_subjects})
    pnr_st_a = _norm({s: pnr_st[s] for s in nr_subjects})

    gnames = sorted(all_subjects)
    pnames_nn = sorted(nn_subjects)
    pnames_nr = sorted(nr_subjects)

    G_gb = np.stack([gal_gb_a[g] for g in gnames])
    G_st = np.stack([gal_st_a[g] for g in gnames])
    PNN_gb = np.stack([pnn_gb_a[p] for p in pnames_nn])
    PNN_st = np.stack([pnn_st_a[p] for p in pnames_nn])
    PNR_gb = np.stack([pnr_gb_a[p] for p in pnames_nr])
    PNR_st = np.stack([pnr_st_a[p] for p in pnames_nr])

    sim_nn_gb = PNN_gb @ G_gb.T
    sim_nn_st = PNN_st @ G_st.T
    sim_nr_gb = PNR_gb @ G_gb.T
    sim_nr_st = PNR_st @ G_st.T

    # ── Barrido alpha ──
    alphas = [round(a / 10, 1) for a in range(11)]
    print("\n  a(GB)  | EER_NN  EER_NR  EER_ALL | TAR@0%_NN TAR@0%_NR TAR@0%_ALL")
    print("  -------|" + "-" * 60)

    best = {"eer_all": 1.0, "alpha": None, "res": None}
    all_results = []
    for alpha in alphas:
        g_nn, i_nn = eval_protocol(sim_nn_gb, sim_nn_st, pnames_nn, gnames, "NN", alpha)
        g_nr, i_nr = eval_protocol(sim_nr_gb, sim_nr_st, pnames_nr, gnames, "NR", alpha)
        r_nn = sweep_tau(g_nn, i_nn)
        r_nr = sweep_tau(g_nr, i_nr)
        r_all = sweep_tau(g_nn + g_nr, i_nn + i_nr)
        row = {"alpha_gaitbase": alpha, "alpha_stgcn": round(1 - alpha, 1),
               "NN": r_nn, "NR": r_nr, "ALL": r_all}
        all_results.append(row)
        print(f"  {alpha:.1f}      | "
              f"{r_nn['EER']*100:5.2f}%  {r_nr['EER']*100:5.2f}%  {r_all['EER']*100:5.2f}%  | "
              f"{r_nn['TAR_FAR0']*100:5.1f}%  {r_nr['TAR_FAR0']*100:5.1f}%  {r_all['TAR_FAR0']*100:5.1f}%")
        if r_all["EER"] < best["eer_all"]:
            best = {"eer_all": r_all["EER"], "alpha": alpha, "res": row}

    print(f"\n[best] a_GB={best['alpha']:.1f}  "
          f"EER_ALL={best['eer_all']*100:.2f}%  "
          f"TAR@FAR=0%_ALL={best['res']['ALL']['TAR_FAR0']*100:.1f}%  "
          f"TAR@FAR=0%_NR={best['res']['NR']['TAR_FAR0']*100:.1f}%")

    alpha_best = best["alpha"]
    g_nn_b, i_nn_b = eval_protocol(sim_nn_gb, sim_nn_st, pnames_nn, gnames, "NN", alpha_best)
    g_nr_b, i_nr_b = eval_protocol(sim_nr_gb, sim_nr_st, pnames_nr, gnames, "NR", alpha_best)

    print(f"\n[detalle a={alpha_best}] genuinos con sim mas baja (top 5):")
    for row in sorted(g_nn_b + g_nr_b, key=lambda r: r["top1_sim"])[:5]:
        flag = "OK" if row["correct_top1"] else "ERR"
        print(f"  {row['protocol']} {row['probe']:28s} -> {row['top1']:28s} {row['top1_sim']:.4f} [{flag}]")
    print(f"[detalle a={alpha_best}] impostores con sim mas alta (top 5):")
    for row in sorted(i_nn_b + i_nr_b, key=lambda r: -r["top1_sim"])[:5]:
        print(f"  {row['protocol']} {row['probe']:28s} -> {row['top1']:28s} {row['top1_sim']:.4f}")

    out = {
        "description": "Late fusion GaitBase + ST-GCN lite (score averaging)",
        "checkpoints": {"gaitbase": CKPT_GB.name, "stgcn": CKPT_STGCN.name},
        "best_alpha_gaitbase": best["alpha"],
        "all_results": all_results,
    }
    out_json = REPORTS / "13_openset_fusion_pose.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out_json.relative_to(REPO)}")


if __name__ == "__main__":
    main()
