"""Fine-tuning de GaitBase (preentrenado en Gait3D) sobre 13 sujetos de train.

Plan aprobado en Fase 3:
- backbone + FCs cargados de GaitBase_Gait3D_120000.pt; BNNecks reinicializado (class_num 3000 -> 13).
- TripletSampler P=4 IDs x K=2 secuencias por id = batch 8.
- CollateFn fixed_unordered con frames_num_fixed=30 (cada batch ve sub-ventanas distintas).
- TripletLoss(margin=0.2) sobre embed_1 (8,256,16) + CE sobre logits BNNecks (8,13,16).
- AMP, optimizador SGD lr=0.01, momentum 0.9, wd 5e-4.
- Scheduler MultiStepLR con milestones [1500, 2500].
- Eval cada 200 iters sobre los 3 sujetos val (gallery=normal, probe=rapido).
- Early stopping: detener si margen val no sube por 3 evals seguidos (patience=3).
- Guarda mejor checkpoint por margen val.

Eval final sobre 3 sujetos test (intocado hasta el final).
Reporta rank-1, rank-5, sim media correcto/erroneo, margen.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
OPENGAIT = REPO / "third_party" / "OpenGait"
CKPT_PRETRAINED = REPO / "checkpoints" / "pretrained" / "GaitBase_Gait3D_120000.pt"
PKL_ROOT = REPO / "data" / "pkl"
SPLITS_YAML = REPO / "configs" / "splits.yaml"
PARTITION_JSON = REPO / "configs" / "partition_finetune.json"
CKPT_OUT_DIR = REPO / "checkpoints" / "finetune"
REPORTS = REPO / "reports"

sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))


# -----------------------------------------------------------------------------
#                         distributed (1-proc) y logging
# -----------------------------------------------------------------------------

def init_single_proc_dist() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
#                              modelo (mismo que smoke + heads)
# -----------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    from modeling.backbones.resnet import ResNet9
    from modeling.modules import (
        SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
        SeparateFCs, SeparateBNNecks,
    )

    class GaitBaseFT(nn.Module):
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
            self.BNNecks = SeparateBNNecks(class_num=num_classes, in_channels=256, parts_num=16)
            self.TP = PackSequenceWrapper(torch.max)
            self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        def forward(self, sils: torch.Tensor, seqL: List[torch.Tensor]):
            if sils.dim() == 4:
                sils = sils.unsqueeze(1)  # (N,1,S,H,W)
            outs = self.Backbone(sils)
            outs = self.TP(outs, seqL, options={"dim": 2})[0]
            feat = self.HPP(outs)
            embed_1 = self.FCs(feat)              # (N, 256, 16)
            embed_2, logits = self.BNNecks(embed_1)  # logits (N, num_classes, 16)
            return embed_1, logits

    return GaitBaseFT()


def load_pretrained_strip_bnnecks(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    # Filtrar BNNecks: el class_num cambia (3000 -> 13), las shapes no calzan
    state_clean = {k: v for k, v in state.items() if not k.startswith("BNNecks.")}
    missing, unexpected = model.load_state_dict(state_clean, strict=False)
    bnnecks_missing = [k for k in missing if k.startswith("BNNecks.")]
    other_missing = [k for k in missing if not k.startswith("BNNecks.")]
    print(f"[ckpt] cargados {len(state_clean)} tensores")
    print(f"[ckpt] missing BNNecks (esperado, se reinicializa): {len(bnnecks_missing)}")
    print(f"[ckpt] missing otros (deberia ser 0): {len(other_missing)}  -> {other_missing[:3]}")
    print(f"[ckpt] unexpected: {len(unexpected)}  -> {unexpected[:3]}")


# -----------------------------------------------------------------------------
#                            losses (locales, sin DDP)
# -----------------------------------------------------------------------------

class TripletLossLocal(nn.Module):
    """Batch-all triplet loss per-part. Equivalente al de OpenGait sin gather DDP."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # embeddings: (N, C, P) -> (P, N, C)
        emb = embeddings.permute(2, 0, 1).contiguous().float()
        # squared euclidean: (P, N, N)
        x2 = (emb ** 2).sum(-1, keepdim=True)
        dist = x2 + x2.transpose(1, 2) - 2 * emb @ emb.transpose(1, 2)
        dist = torch.sqrt(F.relu(dist) + 1e-12)

        # mascaras anchor/positivo/negativo
        same = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (N, N)
        diff = ~same
        # ap_dist (P, N_anchors, K_pos), an_dist (P, N_anchors, K_neg)
        # OpenGait usa: dist[:, same] reshape; same es 2D bool, broadcasting funciona
        # pero el shape es por-anchor variable. Asumimos batch P x K homogeneo (mismo K_pos por anchor).
        p_dim, n_dim, _ = dist.size()
        ap = dist[:, same].view(p_dim, n_dim, -1, 1)  # incluye d(a,a)=0 — OK porque luego se compara con an
        an = dist[:, diff].view(p_dim, n_dim, 1, -1)
        loss_per_triplet = F.relu(ap - an + self.margin).view(p_dim, -1)  # (P, n_triplets)
        # promedio sobre triplets activos (>0), por parte; luego promedio sobre partes
        nonzero_count = (loss_per_triplet > 0).sum(dim=1).float()
        nonzero_sum = loss_per_triplet.sum(dim=1)
        per_part_loss = nonzero_sum / (nonzero_count + 1e-9)
        per_part_loss = torch.where(nonzero_count > 0, per_part_loss, torch.zeros_like(per_part_loss))
        loss = per_part_loss.mean()

        info = {
            "triplet_loss": float(loss.detach()),
            "triplet_active_frac": float((loss_per_triplet > 0).float().mean().detach()),
            "triplet_mean_dist": float(dist.mean().detach()),
        }
        return loss, info


class CrossEntropyLossLocal(nn.Module):
    """CE per-part con label smoothing y scale, equivalente al de OpenGait."""

    def __init__(self, scale: float = 16.0, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.scale = scale
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # logits: (N, C, P), labels: (N,)
        n, c, p = logits.size()
        logits = logits.float() * self.scale
        labels_rep = labels.unsqueeze(1).repeat(1, p)  # (N, P)
        loss = F.cross_entropy(logits, labels_rep, label_smoothing=self.label_smoothing)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            accu = (pred == labels_rep).float().mean()
        return loss, {"ce_loss": float(loss.detach()), "ce_accuracy": float(accu.detach())}


# -----------------------------------------------------------------------------
#                              data: train (OpenGait) + eval (directo)
# -----------------------------------------------------------------------------

def _patch_sampler_for_single_proc() -> None:
    """En 1 proc, dist.broadcast(cuda_tensor) por gloo en Windows es flaky.
    Reemplaza sync_random_sample_list con version local sin broadcast."""
    from data import sampler as sampler_mod

    def _local_sync_random_sample_list(obj_list, k, common_choice=False):
        if common_choice or len(obj_list) < k:
            idx = random.choices(range(len(obj_list)), k=k)
        else:
            idx = torch.randperm(len(obj_list))[:k].tolist()
        return [obj_list[i] for i in idx]

    sampler_mod.sync_random_sample_list = _local_sync_random_sample_list


def build_train_loader(batch_p: int, batch_k: int, frames_fixed: int, num_workers: int):
    from data.dataset import DataSet
    from data.sampler import TripletSampler
    from data.collate_fn import CollateFn
    from utils import get_msg_mgr

    _patch_sampler_for_single_proc()

    tmp_log_dir = REPORTS / "_finetune_logs"
    tmp_log_dir.mkdir(parents=True, exist_ok=True)
    get_msg_mgr().init_manager(str(tmp_log_dir), log_to_file=False, log_iter=50)

    data_cfg = {
        "dataset_root": str(PKL_ROOT),
        "dataset_partition": str(PARTITION_JSON),
        "cache": True,  # precarga los 26 pkl en RAM, son ~5 MB total
        "num_workers": num_workers,
    }
    train_ds = DataSet(data_cfg, training=True)

    sample_cfg = {
        "sample_type": "fixed_unordered",
        "frames_num_fixed": frames_fixed,
    }
    collate = CollateFn(label_set=train_ds.label_set, sample_config=sample_cfg)
    sampler = TripletSampler(train_ds, batch_size=[batch_p, batch_k], batch_shuffle=False)

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
    )
    return train_ds, loader


def load_split(name: str) -> List[str]:
    with SPLITS_YAML.open("r", encoding="utf-8") as f:
        s = yaml.safe_load(f)
    return sorted(s["subject_disjoint"][name])


def load_silhouettes(pkl_path: Path) -> np.ndarray:
    with pkl_path.open("rb") as f:
        arr = pickle.load(f)
    return arr


@torch.no_grad()
def extract_embeddings_for_subjects(model: nn.Module, subjects: List[str],
                                    device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Devuelve (gallery, probe, subjects_in_order). Gallery=normal, Probe=rapido."""
    model.eval()
    g_embs, p_embs, used = [], [], []
    for s in subjects:
        g_path = PKL_ROOT / s / "normal" / "090" / "seq00.pkl"
        p_path = PKL_ROOT / s / "rapido" / "090" / "seq00.pkl"
        if not g_path.exists() or not p_path.exists():
            print(f"[eval] falta pkl para {s}, skip")
            continue
        for path, store in ((g_path, g_embs), (p_path, p_embs)):
            sils = load_silhouettes(path)
            x = torch.from_numpy(sils.astype(np.float32) / 255.0).unsqueeze(0).to(device)
            seqL = [torch.tensor([x.shape[1]], device=device)]
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                emb_1, _ = model(x, seqL)
            store.append(emb_1.float().squeeze(0).cpu().numpy().reshape(-1))
        used.append(s)
    return np.stack(g_embs, 0), np.stack(p_embs, 0), used


def cosine_matrix(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    p = probe / (np.linalg.norm(probe, axis=1, keepdims=True) + 1e-12)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    return p @ g.T


def evaluate(model: nn.Module, subjects: List[str], device: torch.device) -> dict:
    g, p, used = extract_embeddings_for_subjects(model, subjects, device)
    sim = cosine_matrix(p, g)
    n = len(used)
    correct = np.arange(n)
    order = np.argsort(-sim, axis=1)
    rank_pos = np.where(order == correct[:, None])[1]
    sim_correct = sim[correct, correct]
    sim_best_wrong = np.array([np.max(np.delete(sim[i], i)) for i in range(n)])
    margin = sim_correct - sim_best_wrong
    return {
        "n_subjects": n,
        "rank1": float(np.mean(rank_pos == 0)),
        "rank5": float(np.mean(rank_pos < 5)),
        "mean_sim_correct": float(np.mean(sim_correct)),
        "mean_sim_best_wrong": float(np.mean(sim_best_wrong)),
        "mean_margin": float(np.mean(margin)),
        "min_margin": float(np.min(margin)),
        "subjects": used,
    }


# -----------------------------------------------------------------------------
#                                  loop principal
# -----------------------------------------------------------------------------

@dataclass
class Config:
    seed: int = 42
    batch_p: int = 4
    batch_k: int = 2
    frames_fixed: int = 30
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    total_iter: int = 1500
    milestones: Tuple[int, ...] = (750, 1250)
    gamma: float = 0.1
    log_iter: int = 50
    eval_iter: int = 200
    triplet_margin: float = 0.2
    triplet_weight: float = 1.0
    ce_weight: float = 1.0
    ce_scale: float = 16.0
    ce_smoothing: float = 0.1
    early_stop_patience: int = 3
    num_workers: int = 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true", help="50 iters, eval cada 25, para debug")
    args = ap.parse_args()

    cfg = Config()
    if args.dry_run:
        cfg.total_iter = 50
        cfg.log_iter = 10
        cfg.eval_iter = 25
        cfg.early_stop_patience = 99
    print("=" * 70)
    print(json.dumps(cfg.__dict__, indent=2, default=str))
    print("=" * 70)

    init_single_proc_dist()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    CKPT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # 1. modelo + pesos preentrenados
    train_subjects = load_split("train")
    val_subjects = load_split("val")
    test_subjects = load_split("test")
    print(f"[init] train={len(train_subjects)} val={len(val_subjects)} test={len(test_subjects)}")

    num_classes = len(train_subjects)  # 13
    model = build_model(num_classes).to(device)
    load_pretrained_strip_bnnecks(model, CKPT_PRETRAINED, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] modelo construido: {n_params:,} params")

    # 2. datos
    train_ds, train_loader = build_train_loader(
        batch_p=cfg.batch_p, batch_k=cfg.batch_k,
        frames_fixed=cfg.frames_fixed, num_workers=cfg.num_workers,
    )
    print(f"[data] train_ds: {len(train_ds)} secuencias, {len(train_ds.label_set)} sujetos")
    print(f"[data] label_set sample: {train_ds.label_set[:3]}")

    # 3. losses + optim
    triplet = TripletLossLocal(margin=cfg.triplet_margin).to(device)
    ce = CrossEntropyLossLocal(scale=cfg.ce_scale, label_smoothing=cfg.ce_smoothing).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                            momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=list(cfg.milestones), gamma=cfg.gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. eval baseline pre-finetune
    print("\n[eval-pre] evaluando pre-finetune en val...")
    pre = evaluate(model, val_subjects, device)
    print(f"[eval-pre] rank1={pre['rank1']*100:.1f}% margen={pre['mean_margin']:+.4f} "
          f"sim_ok={pre['mean_sim_correct']:.4f} sim_wr={pre['mean_sim_best_wrong']:.4f}")

    # 5. training loop
    history = {"train": [], "val": [pre | {"iter": 0}]}
    best = {"margin": pre["mean_margin"], "iter": 0, "state": None}
    patience_left = cfg.early_stop_patience
    log_buf = {"triplet": 0.0, "ce": 0.0, "active": 0.0, "accu": 0.0, "n": 0}
    train_iter = iter(train_loader)
    t_start = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for it in range(1, cfg.total_iter + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # batch = [fras_batch, labs_batch, typs_batch, vies_batch, None]
        # fras_batch = [feature_num=1, batch_size] de arrays (T,64,44)
        fras = batch[0][0]  # lista de N arrays (frames_fixed, 64, 44) uint8
        labels = torch.tensor(batch[1], device=device, dtype=torch.long)

        # OpenGait packed format: (1, sum_S, H, W) con seqL=[S_1, S_2, ...]
        # PackSequenceWrapper hace split por seqL y max-pool por sub-secuencia.
        seq_lens = [int(f.shape[0]) for f in fras]
        sils_np = np.concatenate(fras, axis=0).astype(np.float32) / 255.0  # (sum_S, H, W)
        sils = torch.from_numpy(sils_np).unsqueeze(0).to(device)            # (1, sum_S, H, W)
        seqL = [torch.tensor(seq_lens, device=device)]

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            emb_1, logits = model(sils, seqL)
            l_tri, info_tri = triplet(emb_1, labels)
            l_ce, info_ce = ce(logits, labels)
            loss = cfg.triplet_weight * l_tri + cfg.ce_weight * l_ce

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        log_buf["triplet"] += info_tri["triplet_loss"]
        log_buf["ce"] += info_ce["ce_loss"]
        log_buf["active"] += info_tri["triplet_active_frac"]
        log_buf["accu"] += info_ce["ce_accuracy"]
        log_buf["n"] += 1

        if it % cfg.log_iter == 0:
            n = log_buf["n"]
            dt = time.time() - t_start
            cur_lr = scheduler.get_last_lr()[0]
            print(f"[it {it:>5}/{cfg.total_iter}] lr={cur_lr:.4f} "
                  f"tri={log_buf['triplet']/n:.4f} ce={log_buf['ce']/n:.4f} "
                  f"active={log_buf['active']/n:.3f} ce_acc={log_buf['accu']/n:.3f} "
                  f"({dt/it*1000:.0f} ms/it)")
            history["train"].append({
                "iter": it, "lr": cur_lr,
                "triplet": log_buf["triplet"]/n, "ce": log_buf["ce"]/n,
                "active": log_buf["active"]/n, "ce_acc": log_buf["accu"]/n,
            })
            log_buf = {"triplet": 0.0, "ce": 0.0, "active": 0.0, "accu": 0.0, "n": 0}

        if it % cfg.eval_iter == 0:
            v = evaluate(model, val_subjects, device)
            v["iter"] = it
            history["val"].append(v)
            print(f"[eval@{it}] val rank1={v['rank1']*100:.1f}% rank5={v['rank5']*100:.1f}% "
                  f"margen={v['mean_margin']:+.4f} (min={v['min_margin']:+.4f}) "
                  f"sim_ok={v['mean_sim_correct']:.4f} sim_wr={v['mean_sim_best_wrong']:.4f}")
            if v["mean_margin"] > best["margin"]:
                best = {"margin": v["mean_margin"], "iter": it,
                        "state": {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}}
                patience_left = cfg.early_stop_patience
                print(f"[eval@{it}] >> nuevo mejor margen, guardando")
            else:
                patience_left -= 1
                print(f"[eval@{it}] sin mejora, patience={patience_left}")
                if patience_left <= 0:
                    print(f"[eval@{it}] early stopping (best margin {best['margin']:+.4f} @ iter {best['iter']})")
                    break

    # 6. restaurar mejor checkpoint y eval test
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        ckpt_out = CKPT_OUT_DIR / f"gaitbase_ft_best_iter{best['iter']}.pt"
        torch.save({"model": best["state"], "iter": best["iter"], "margin": best["margin"],
                    "config": cfg.__dict__}, ckpt_out)
        print(f"\n[ckpt] mejor modelo guardado en {ckpt_out.relative_to(REPO)}")

    print("\n[eval-test] evaluando mejor modelo en test (intocado hasta ahora)...")
    test_metrics = evaluate(model, test_subjects, device)
    print(f"[eval-test] rank1={test_metrics['rank1']*100:.1f}% rank5={test_metrics['rank5']*100:.1f}% "
          f"margen={test_metrics['mean_margin']:+.4f} (min={test_metrics['min_margin']:+.4f}) "
          f"sim_ok={test_metrics['mean_sim_correct']:.4f} sim_wr={test_metrics['mean_sim_best_wrong']:.4f}")

    # 7. dump report
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[done] VRAM pico: {peak:.0f} MB")
    total_dt = time.time() - t_start
    print(f"[done] tiempo total: {total_dt/60:.1f} min")

    report = {
        "config": cfg.__dict__,
        "pretrained_ckpt": str(CKPT_PRETRAINED.name),
        "best_iter": best["iter"],
        "best_val_margin": best["margin"],
        "test": test_metrics,
        "val_history": history["val"],
        "train_history": history["train"],
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "vram_peak_mb": float(peak) if device.type == "cuda" else None,
        "total_minutes": round(total_dt / 60, 2),
    }
    out_json = REPORTS / "03_finetune_results.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[done] reporte: {out_json.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
