"""Zero-shot rank-1 / rank-5 video-disjoint con GaitBase pre-entrenado en Gait3D.

NO entrena. NO finetunea. Extrae embeddings sobre los 19 sujetos x 2 condiciones
(38 secuencias) usando GaitBase_Gait3D_120000.pt, computa similitud coseno entre
gallery (normal) y probe (rapido), y reporta:

- rank-1, rank-5 video-disjoint global
- top-3 vecinos por probe (matriz para inspeccion)
- aciertos/errores por sujeto
- distribucion de similitudes correctas vs incorrectas

Este reporte es el **piso baseline** de Fase 3. Cualquier fine-tuning posterior
debe superarlo o no tiene sentido.

Salidas:
- reports/03_zeroshot_results.json (todas las metricas)
- reports/03_zeroshot_topk.csv (matriz por probe)
- impresion en consola con resumen
"""

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
CKPT = REPO / "checkpoints" / "pretrained" / "GaitBase_Gait3D_120000.pt"
PKL_ROOT = REPO / "data" / "pkl"
REPORTS = REPO / "reports"

sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))

from modeling.backbones.resnet import ResNet9  # noqa: E402
from modeling.modules import (  # noqa: E402
    SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
    SeparateFCs, SeparateBNNecks,
)


GALLERY_COND = "normal"
PROBE_COND = "rapido"


class GaitBaseInfer(nn.Module):
    """Misma arquitectura que smoke_test_gaitbase.GaitBaseSmoke."""

    def __init__(self) -> None:
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
        self.BNNecks = SeparateBNNecks(class_num=3000, in_channels=256, parts_num=16)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def forward(self, sils: torch.Tensor, seqL: List[torch.Tensor]) -> torch.Tensor:
        if sils.dim() == 4:
            sils = sils.unsqueeze(1)  # (N, 1, S, H, W)
        outs = self.Backbone(sils)
        outs = self.TP(outs, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs)
        embed_1 = self.FCs(feat)  # (N, 256, 16)
        # No usamos BNNecks/logits — embed_1 es el feature de matching.
        return embed_1


def list_sequences() -> List[Tuple[str, str, Path]]:
    out = []
    for subj_dir in sorted(PKL_ROOT.iterdir()):
        if not subj_dir.is_dir():
            continue
        for cond in (GALLERY_COND, PROBE_COND):
            seq_dir = subj_dir / cond / "090"
            if not seq_dir.is_dir():
                print(f"[warn] falta {seq_dir}")
                continue
            pkls = sorted(seq_dir.glob("*.pkl"))
            if not pkls:
                print(f"[warn] sin .pkl en {seq_dir}")
                continue
            out.append((subj_dir.name, cond, pkls[0]))
    return out


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
        emb = model(sils, seqL)  # (1, 256, 16)
    return emb.float().squeeze(0).cpu().numpy()  # (256, 16)


def cosine_matrix(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """probe: (Np, D), gallery: (Ng, D). Devuelve (Np, Ng)."""
    p = probe / (np.linalg.norm(probe, axis=1, keepdims=True) + 1e-12)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    return p @ g.T


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 1. construir modelo + cargar pesos Gait3D
    t0 = time.time()
    model = GaitBaseInfer().to(device).eval()
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[info] modelo+pesos cargados en {time.time()-t0:.1f}s | missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"       sample missing : {missing[:3]}")
    if unexpected[:3]:
        print(f"       sample unexp.  : {unexpected[:3]}")

    # 2. extraer embeddings
    seqs = list_sequences()
    print(f"[info] {len(seqs)} secuencias detectadas (esperado 38)")
    embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    seq_lengths: Dict[Tuple[str, str], int] = {}
    t0 = time.time()
    for subj, cond, path in seqs:
        sils = load_silhouettes(path)
        emb = extract_embedding(model, sils, device)
        embeddings[(subj, cond)] = emb.reshape(-1)  # (4096,)
        seq_lengths[(subj, cond)] = int(sils.shape[0])
    dt = time.time() - t0
    print(f"[info] extraccion completa en {dt:.1f}s ({len(seqs)/dt:.1f} seq/s)")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[info] VRAM pico: {peak:.0f} MB")

    # 3. matching gallery (normal) vs probe (rapido)
    subjects = sorted({s for s, _ in embeddings.keys()})
    gallery = np.stack([embeddings[(s, GALLERY_COND)] for s in subjects], axis=0)
    probe = np.stack([embeddings[(s, PROBE_COND)] for s in subjects], axis=0)
    sim = cosine_matrix(probe, gallery)  # (19, 19)

    # 4. metricas rank-K
    n = len(subjects)
    correct_idx = np.arange(n)
    order = np.argsort(-sim, axis=1)  # descendente
    rank_pos = np.where(order == correct_idx[:, None])[1]  # posicion del correcto por probe
    rank1 = float(np.mean(rank_pos == 0))
    rank5 = float(np.mean(rank_pos < 5))

    # similitud al correcto vs al mejor incorrecto
    sim_correct = sim[correct_idx, correct_idx]
    sim_best_wrong = np.array([
        np.max(np.delete(sim[i], i)) for i in range(n)
    ])
    margin = sim_correct - sim_best_wrong

    # 5. tabla por probe
    rows = []
    for i, subj in enumerate(subjects):
        top3 = order[i, :3]
        rows.append({
            "probe_subject": subj,
            "probe_T": seq_lengths[(subj, PROBE_COND)],
            "gallery_T": seq_lengths[(subj, GALLERY_COND)],
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

    # 6. salida JSON + CSV
    results = {
        "model": "GaitBase",
        "checkpoint": CKPT.name,
        "n_subjects": n,
        "gallery_condition": GALLERY_COND,
        "probe_condition": PROBE_COND,
        "rank1": round(rank1, 4),
        "rank5": round(rank5, 4),
        "mean_sim_correct": round(float(np.mean(sim_correct)), 4),
        "mean_sim_best_wrong": round(float(np.mean(sim_best_wrong)), 4),
        "mean_margin": round(float(np.mean(margin)), 4),
        "embedding_dim": int(gallery.shape[1]),
        "per_probe": rows,
    }
    out_json = REPORTS / "03_zeroshot_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV minimo manual (no pandas)
    out_csv = REPORTS / "03_zeroshot_topk.csv"
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in headers))
    out_csv.write_text("\n".join(lines), encoding="utf-8")

    # 7. resumen consola
    print("\n" + "=" * 60)
    print(f"  RANK-1 : {rank1*100:5.1f}%   ({int(rank1*n)}/{n})")
    print(f"  RANK-5 : {rank5*100:5.1f}%   ({int(rank5*n)}/{n})")
    print(f"  margen medio (sim_correct - sim_best_wrong) : {np.mean(margin):+.4f}")
    print(f"  sim media correcto       : {np.mean(sim_correct):.4f}")
    print(f"  sim media mejor erroneo  : {np.mean(sim_best_wrong):.4f}")
    print("=" * 60)

    print(f"\nDetalle por probe (rank del correcto, top-1):")
    print(f"{'probe':30s} {'rk':>3s} {'sim_ok':>8s} {'sim_wr':>8s} {'top1':>30s}")
    for r in rows:
        marker = "" if r["rank_of_correct"] == 1 else "  <-- error" if r["rank_of_correct"] > 5 else "  (rk>1)"
        print(f"{r['probe_subject']:30s} {r['rank_of_correct']:3d} {r['sim_correct']:8.4f} {r['sim_best_wrong']:8.4f} {r['top1']:>30s}{marker}")

    print(f"\nresultados completos: {out_json.relative_to(REPO)}")
    print(f"matriz por probe   : {out_csv.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
