"""Construye la gallery multi-template de producción para Fase 6 (Opción A).

Para cada uno de los 19 sujetos:
  - Carga todas las pkls disponibles en `data/pkl_multisession/<subj>/<cond>/090/seq*.pkl`
    (normal_s1, normal_s2, rapido_s1, rapido_s2 — algunos sujetos tienen 3).
  - Calcula embedding con GaitBase iter1200 (mismo pipeline que `openset_eval.py`).
  - L2-normaliza cada embedding sobre el vector aplanado.
  - Guarda TODOS los templates por sujeto, sin promediar.

En runtime el matcher usa **max sim** sobre los K templates del sujeto:
  sim(probe, subj) = max_k cos(probe, gallery[subj, k]).
Esta estrategia preserva la separación inter-sujeto (no comprime los centroides)
y cubre la varianza intra-sujeto (ropa, velocidad, sesion).

Salidas:
  - gallery/embeddings.npy  : float32 (19, K_max, 256, 16) padded con ceros.
  - gallery/valid_mask.npy  : bool (19, K_max) — True donde hay template real.
  - gallery/index.json      : metadata + sanity de separación bajo max-sim.

Uso:
    python scripts/build_gallery.py \\
        --checkpoint checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt \\
        --class-num 13

Notas:
  * τ=0.9807 fue calibrada en Fase 5 con gallery=normal_s1 (1 template). Con
    multi-template + max-sim la τ óptima cambia. Recalibrar en 6.5 con
    `openset_eval.py` adaptado (o validar empíricamente en cámara).
  * `jesusantonioaguilarfelix` tiene solo 3 pkls (falta normal_s2) → padding.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
OPENGAIT = REPO / "third_party" / "OpenGait"

sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))

from modeling.backbones.resnet import ResNet9  # noqa: E402
from modeling.modules import (  # noqa: E402
    SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
    SeparateFCs, SeparateBNNecks,
)


class GaitBaseInfer(nn.Module):
    """Misma arquitectura que `openset_eval.py` (Fase 5)."""

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
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{pkl_path}: esperaba ndarray, obtuve {type(arr)}")
    if arr.ndim != 3 or arr.shape[1:] != (64, 44):
        raise ValueError(f"{pkl_path}: shape inesperada {arr.shape}")
    return arr


@torch.no_grad()
def extract_embedding(model: GaitBaseInfer, sils_np: np.ndarray, device: torch.device) -> np.ndarray:
    """Devuelve embedding (256, 16) float32, sin normalizar todavía."""
    sils = torch.from_numpy(sils_np.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    seqL = [torch.tensor([sils.shape[1]], device=device)]
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        emb = model(sils, seqL)
    return emb.float().squeeze(0).cpu().numpy()  # (256, 16)


def list_subject_pkls(multisession_root: Path, subject: str) -> List[Path]:
    """Devuelve todas las pkls disponibles del sujeto en pkl_multisession."""
    out: List[Path] = []
    subj_dir = multisession_root / subject
    if not subj_dir.is_dir():
        return out
    for cond_dir in sorted(subj_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        seq_dir = cond_dir / "090"
        if not seq_dir.is_dir():
            continue
        out.extend(sorted(seq_dir.glob("*.pkl")))
    return out


def l2_normalize(v: np.ndarray) -> np.ndarray:
    flat = v.reshape(-1)
    n = float(np.linalg.norm(flat))
    if n < 1e-12:
        return v.copy()
    return (v / n).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints" / "finetune" /
                                                 "gaitbase_ft_multisession_best_iter1200.pt"))
    ap.add_argument("--class-num", type=int, default=13)
    ap.add_argument("--multisession-root", default=str(REPO / "data" / "pkl_multisession"))
    ap.add_argument("--out-dir", default=str(REPO / "gallery"))
    ap.add_argument("--tau", type=float, default=0.9807,
                    help="Umbral calibrado en Fase 5 (referencia, ver caveat en docstring).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    multisession_root = Path(args.multisession_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device} | checkpoint={Path(args.checkpoint).name}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model = GaitBaseInfer(class_num=args.class_num).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[info] pesos cargados en {time.time()-t0:.1f}s | "
          f"missing={len(missing)} unexpected={len(unexpected)}")

    subjects = sorted([p.name for p in multisession_root.iterdir() if p.is_dir()])
    if len(subjects) != 19:
        print(f"[warn] esperaba 19 sujetos, encontrados: {len(subjects)}")

    # Acumuladores
    per_subject_embs: List[np.ndarray] = []   # cada elemento: (k_subj, 256, 16)
    n_sequences_used: Dict[str, int] = {}
    per_subject_files: Dict[str, List[str]] = {}

    t0 = time.time()
    for subj in subjects:
        pkls = list_subject_pkls(multisession_root, subj)
        if not pkls:
            print(f"[error] sujeto {subj} sin pkls en pkl_multisession; abortando")
            return 1

        embs = []
        for p in pkls:
            sils = load_silhouettes(p)
            e = extract_embedding(model, sils, device)
            embs.append(l2_normalize(e))
        embs_arr = np.stack(embs, axis=0)                      # (k, 256, 16)
        per_subject_embs.append(embs_arr)
        n_sequences_used[subj] = int(len(pkls))
        per_subject_files[subj] = [str(p.relative_to(REPO)).replace("\\", "/") for p in pkls]

        # Cohesión intra-sujeto bajo max-sim semantics: similitud media entre
        # cada template y el resto del set de su sujeto (LOO).
        flat = embs_arr.reshape(embs_arr.shape[0], -1)
        gram = flat @ flat.T
        np.fill_diagonal(gram, -np.inf)
        loo_max = gram.max(axis=1)  # cada template vs sus hermanos -> max
        intra_min = float(loo_max.min())
        intra_mean = float(loo_max.mean())
        print(f"  {subj:30s}  k={len(pkls)}  intra_loo_max(min/mean)="
              f"{intra_min:.4f}/{intra_mean:.4f}")

    # Padding a tensor uniforme (19, K_max, 256, 16)
    k_max = max(arr.shape[0] for arr in per_subject_embs)
    n = len(subjects)
    feat_shape = per_subject_embs[0].shape[1:]
    gallery_arr = np.zeros((n, k_max, *feat_shape), dtype=np.float32)
    valid_mask = np.zeros((n, k_max), dtype=bool)
    for i, arr in enumerate(per_subject_embs):
        gallery_arr[i, :arr.shape[0]] = arr
        valid_mask[i, :arr.shape[0]] = True

    out_npy = out_dir / "embeddings.npy"
    out_mask = out_dir / "valid_mask.npy"
    np.save(out_npy, gallery_arr)
    np.save(out_mask, valid_mask)
    print(f"\n[ok] gallery shape={gallery_arr.shape} (n_subjects, K_max, 256, 16) "
          f"-> {out_npy.relative_to(REPO)}")
    print(f"[ok] valid_mask shape={valid_mask.shape} -> {out_mask.relative_to(REPO)}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[info] VRAM pico: {peak:.0f} MB | tiempo total: {time.time()-t0:.1f}s")

    # --- Sanity inter-sujeto bajo max-sim ---
    # Para cada par (i, j) i!=j: sim = max sobre todos los pares de templates.
    flat_all = gallery_arr.reshape(n, k_max, -1)  # (n, K, D)
    inter_max = np.full((n, n), -np.inf, dtype=np.float32)
    for i in range(n):
        ki = int(valid_mask[i].sum())
        Ti = flat_all[i, :ki]                      # (ki, D)
        for j in range(n):
            if i == j:
                continue
            kj = int(valid_mask[j].sum())
            Tj = flat_all[j, :kj]                  # (kj, D)
            sims = Ti @ Tj.T                       # (ki, kj)
            inter_max[i, j] = sims.max()

    upper = inter_max[np.triu_indices(n, k=1)]
    max_inter = float(upper.max())
    arg = int(upper.argmax())
    iu, ju = np.triu_indices(n, k=1)
    pair_subj = (subjects[int(iu[arg])], subjects[int(ju[arg])])
    mean_inter = float(upper.mean())

    # Top-5 pares más cercanos (worst-case impostors bajo max-sim)
    order = np.argsort(-upper)
    top5_pairs = []
    for k in order[:5]:
        a, b = int(iu[k]), int(ju[k])
        top5_pairs.append({
            "pair": [subjects[a], subjects[b]],
            "max_sim": round(float(upper[k]), 4),
        })

    # --- Simulación leave-one-out (LOO) bajo max-sim semantics ---
    # Cada template hace de "probe", el resto de su sujeto + todos los otros sujetos
    # forman la gallery efectiva. Reporta separación genuino vs impostor real.
    g_sims, i_sims, n_correct = [], [], 0
    n_total = 0
    for i in range(n):
        ki = int(valid_mask[i].sum())
        if ki < 2:
            continue
        Ti = flat_all[i, :ki]
        for q in range(ki):
            probe = Ti[q]
            # Genuino: max sim contra hermanos (excluyendo el propio probe)
            sib = np.concatenate([Ti[:q], Ti[q + 1:]], axis=0)
            g_sim = float((sib @ probe).max())
            # Impostor: max sim contra todos los templates de otros sujetos
            best_imp_sim, best_imp_subj = -np.inf, -1
            for j in range(n):
                if j == i:
                    continue
                kj = int(valid_mask[j].sum())
                Tj = flat_all[j, :kj]
                s = float((Tj @ probe).max())
                if s > best_imp_sim:
                    best_imp_sim, best_imp_subj = s, j
            g_sims.append(g_sim)
            i_sims.append(best_imp_sim)
            n_total += 1
            if g_sim > best_imp_sim:
                n_correct += 1

    g_sims_arr = np.array(g_sims) if g_sims else np.array([])
    i_sims_arr = np.array(i_sims) if i_sims else np.array([])
    loo_summary = {
        "n_probes": int(n_total),
        "rank1_top1_correct": round(n_correct / max(1, n_total), 4),
        "genuine": {
            "mean": round(float(g_sims_arr.mean()), 4) if len(g_sims_arr) else None,
            "min":  round(float(g_sims_arr.min()), 4) if len(g_sims_arr) else None,
        },
        "impostor": {
            "mean": round(float(i_sims_arr.mean()), 4) if len(i_sims_arr) else None,
            "max":  round(float(i_sims_arr.max()), 4) if len(i_sims_arr) else None,
        },
        "gap_mean": (round(float(g_sims_arr.mean() - i_sims_arr.mean()), 4)
                     if len(g_sims_arr) else None),
    }

    metadata = {
        "subjects": subjects,
        "checkpoint": Path(args.checkpoint).name,
        "class_num": args.class_num,
        "tau_phase5": args.tau,
        "feat_dim": list(gallery_arr.shape[2:]),
        "n_subjects": int(n),
        "k_max": int(k_max),
        "n_sequences_used": n_sequences_used,
        "per_subject_files": per_subject_files,
        "build_strategy": (
            "multi-template per subject (no averaging). Runtime matcher uses "
            "max sim over the K templates of each subject. Preserves inter-subject "
            "separation; covers intra-subject variance (clothing, speed, session)."
        ),
        "inter_subject_sim_under_max_sim": {
            "max": round(max_inter, 4),
            "max_pair": list(pair_subj),
            "mean_off_diagonal": round(mean_inter, 4),
            "top5_closest_pairs": top5_pairs,
        },
        "leave_one_out_simulation": loo_summary,
        "calibration_caveat": (
            "Tau=0.9807 fue calibrado en Fase 5 con gallery=normal_s1 (1 template/sujeto, "
            "matching directo). Con multi-template + max-sim los rangos de sim cambian. "
            "Recalibrar tau usando esta gallery: ejecutar openset_eval con probes externos "
            "y matcher max-sim, o validar en 6.5 contra camara real."
        ),
    }
    out_json = out_dir / "index.json"
    out_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] metadata -> {out_json.relative_to(REPO)}")

    print(f"\n[sanity inter-sujeto bajo max-sim]")
    print(f"  max={max_inter:.4f} entre {pair_subj}")
    print(f"  media off-diag={mean_inter:.4f}")
    print(f"  top-5 pares mas cercanos:")
    for tp in top5_pairs:
        print(f"    {tp['pair'][0]:30s} <-> {tp['pair'][1]:30s} sim={tp['max_sim']:.4f}")

    print(f"\n[LOO simulation, n_probes={n_total}]")
    print(f"  rank1_top1_correct = {loo_summary['rank1_top1_correct']*100:.1f}%")
    if loo_summary['genuine']['mean'] is not None:
        print(f"  genuine: mean={loo_summary['genuine']['mean']:.4f}  "
              f"min={loo_summary['genuine']['min']:.4f}")
        print(f"  impostor: mean={loo_summary['impostor']['mean']:.4f}  "
              f"max={loo_summary['impostor']['max']:.4f}")
        print(f"  gap_mean={loo_summary['gap_mean']:+.4f}")

    if max_inter >= args.tau:
        print(f"\n[warn] inter_max >= tau Phase5 ({args.tau:.4f}). "
              "Recalibracion de tau pendiente para 6.5.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
