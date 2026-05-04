"""Evaluación cross-session (Fase 3 → Fase 4 bridge).

Gallery en sesión 1 (data/pkl/), probe en sesión 2 (data/pkl_s2/). Mismo
código base que zeroshot_gaitbase.py pero con dos rutas pkl y soporte para
cargar cualquier checkpoint (pretrained o fine-tune de Fase 3).

Dos protocolos:
  - NN: gallery=normal_s1  probe=normal_s2  (cross-clothing+session, misma velocidad)
  - NR: gallery=normal_s1  probe=rapido_s2  (cross-clothing+session+speed)

Uso:
    python scripts/cross_session_eval.py \
        --checkpoint checkpoints/pretrained/GaitBase_Gait3D_120000.pt \
        --tag pretrained \
        --gallery-root data/pkl \
        --probe-root   data/pkl_s2

Salidas:
    reports/04_cross_session_<tag>.json
    reports/04_cross_session_<tag>_topk.csv
"""

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
    """Misma arquitectura que zeroshot_gaitbase.GaitBaseInfer."""

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
        embed_1 = self.FCs(feat)
        return embed_1


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
    return emb.float().squeeze(0).cpu().numpy()


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


def cosine_matrix(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    p = probe / (np.linalg.norm(probe, axis=1, keepdims=True) + 1e-12)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    return p @ g.T


def eval_protocol(
    embeddings_g: Dict[str, np.ndarray],
    embeddings_p: Dict[str, np.ndarray],
    seq_lengths_g: Dict[str, int],
    seq_lengths_p: Dict[str, int],
) -> Dict:
    subjects = sorted(set(embeddings_g.keys()) & set(embeddings_p.keys()))
    gallery = np.stack([embeddings_g[s] for s in subjects], axis=0)
    probe = np.stack([embeddings_p[s] for s in subjects], axis=0)
    sim = cosine_matrix(probe, gallery)
    n = len(subjects)
    correct_idx = np.arange(n)
    order = np.argsort(-sim, axis=1)
    rank_pos = np.where(order == correct_idx[:, None])[1]
    rank1 = float(np.mean(rank_pos == 0))
    rank5 = float(np.mean(rank_pos < 5))
    sim_correct = sim[correct_idx, correct_idx]
    sim_best_wrong = np.array([np.max(np.delete(sim[i], i)) for i in range(n)])
    margin = sim_correct - sim_best_wrong

    rows = []
    for i, subj in enumerate(subjects):
        top3 = order[i, :3]
        rows.append({
            "probe_subject": subj,
            "probe_T": seq_lengths_p[subj],
            "gallery_T": seq_lengths_g[subj],
            "rank_of_correct": int(rank_pos[i]) + 1,
            "sim_correct": round(float(sim_correct[i]), 4),
            "sim_best_wrong": round(float(sim_best_wrong[i]), 4),
            "margin": round(float(margin[i]), 4),
            "top1": subjects[top3[0]],
            "top1_sim": round(float(sim[i, top3[0]]), 4),
            "top2": subjects[top3[1]],
            "top2_sim": round(float(sim[i, top3[1]]), 4),
            "top3": subjects[top3[2]],
            "top3_sim": round(float(sim[i, top3[2]]), 4),
        })

    return {
        "n_subjects": n,
        "rank1": round(rank1, 4),
        "rank5": round(rank5, 4),
        "mean_sim_correct": round(float(np.mean(sim_correct)), 4),
        "mean_sim_best_wrong": round(float(np.mean(sim_best_wrong)), 4),
        "mean_margin": round(float(np.mean(margin)), 4),
        "min_margin": round(float(np.min(margin)), 4),
        "per_probe": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="ruta al .pt")
    ap.add_argument("--tag", required=True, help="etiqueta para salidas (p.ej. pretrained, ft_iter400)")
    ap.add_argument("--gallery-root", default=str(REPO / "data" / "pkl"))
    ap.add_argument("--probe-root", default=str(REPO / "data" / "pkl_s2"))
    ap.add_argument("--class-num", type=int, default=3000,
                    help="class_num de BNNecks del checkpoint (3000 Gait3D, 13 fine-tune)")
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
    if missing[:3]:
        print(f"       sample missing : {missing[:3]}")
    if unexpected[:3]:
        print(f"       sample unexp.  : {unexpected[:3]}")

    gallery_root = Path(args.gallery_root)
    probe_root = Path(args.probe_root)

    g_pkls = load_pkls(gallery_root, ("normal",))
    p_pkls = load_pkls(probe_root, ("normal", "rapido"))
    print(f"[info] gallery normal (s1): {len(g_pkls)} seqs | probe normal+rapido (s2): {len(p_pkls)} seqs")

    embs_g_n: Dict[str, np.ndarray] = {}
    seqL_g_n: Dict[str, int] = {}
    embs_p_n: Dict[str, np.ndarray] = {}
    seqL_p_n: Dict[str, int] = {}
    embs_p_r: Dict[str, np.ndarray] = {}
    seqL_p_r: Dict[str, int] = {}

    t0 = time.time()
    for (subj, cond), path in g_pkls.items():
        sils = load_silhouettes(path)
        emb = extract_embedding(model, sils, device).reshape(-1)
        if cond == "normal":
            embs_g_n[subj] = emb
            seqL_g_n[subj] = int(sils.shape[0])
    for (subj, cond), path in p_pkls.items():
        sils = load_silhouettes(path)
        emb = extract_embedding(model, sils, device).reshape(-1)
        if cond == "normal":
            embs_p_n[subj] = emb
            seqL_p_n[subj] = int(sils.shape[0])
        elif cond == "rapido":
            embs_p_r[subj] = emb
            seqL_p_r[subj] = int(sils.shape[0])
    dt = time.time() - t0
    total = len(g_pkls) + len(p_pkls)
    print(f"[info] extracción completa {total} seqs en {dt:.1f}s ({total/dt:.1f} seq/s)")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[info] VRAM pico: {peak:.0f} MB")

    results = {
        "model": "GaitBase",
        "checkpoint": str(Path(args.checkpoint).name),
        "tag": args.tag,
        "gallery_root": str(gallery_root),
        "probe_root": str(probe_root),
        "protocols": {},
    }

    print("\n=== NN: gallery=normal_s1  probe=normal_s2 (cross-session, misma velocidad) ===")
    r_nn = eval_protocol(embs_g_n, embs_p_n, seqL_g_n, seqL_p_n)
    results["protocols"]["NN"] = r_nn
    print(f"  rank-1 {r_nn['rank1']*100:5.1f}%   rank-5 {r_nn['rank5']*100:5.1f}%   margen {r_nn['mean_margin']:+.4f}  min {r_nn['min_margin']:+.4f}")

    print("\n=== NR: gallery=normal_s1  probe=rapido_s2 (cross-session + cross-speed) ===")
    r_nr = eval_protocol(embs_g_n, embs_p_r, seqL_g_n, seqL_p_r)
    results["protocols"]["NR"] = r_nr
    print(f"  rank-1 {r_nr['rank1']*100:5.1f}%   rank-5 {r_nr['rank5']*100:5.1f}%   margen {r_nr['mean_margin']:+.4f}  min {r_nr['min_margin']:+.4f}")

    out_json = REPORTS / f"04_cross_session_{args.tag}.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out_json.relative_to(REPO)}")

    # CSV combinado
    out_csv = REPORTS / f"04_cross_session_{args.tag}_topk.csv"
    lines = ["protocol,probe_subject,probe_T,gallery_T,rank_of_correct,sim_correct,sim_best_wrong,margin,top1,top1_sim,top2,top2_sim,top3,top3_sim"]
    for proto, rdata in results["protocols"].items():
        for r in rdata["per_probe"]:
            lines.append(",".join([proto] + [str(r[k]) for k in [
                "probe_subject","probe_T","gallery_T","rank_of_correct",
                "sim_correct","sim_best_wrong","margin",
                "top1","top1_sim","top2","top2_sim","top3","top3_sim",
            ]]))
    out_csv.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out_csv.relative_to(REPO)}")

    # impresión detalle
    for proto, rdata in results["protocols"].items():
        print(f"\n[{proto}] detalle por probe:")
        print(f"{'probe':30s} {'rk':>3s} {'sim_ok':>8s} {'sim_wr':>8s} {'top1':>30s}")
        for r in rdata["per_probe"]:
            marker = "" if r["rank_of_correct"] == 1 else "  <-- error"
            print(f"{r['probe_subject']:30s} {r['rank_of_correct']:3d} {r['sim_correct']:8.4f} {r['sim_best_wrong']:8.4f} {r['top1']:>30s}{marker}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
