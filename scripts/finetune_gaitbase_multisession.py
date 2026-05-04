"""Fine-tuning GaitBase (Fase 3.5) — multisession con cross-session supervision.

Diferencias con scripts/finetune_gaitbase.py (Fase 3):
  - PKL_ROOT = data/pkl_multisession (contiene 4 tipos por sujeto:
    normal_s1, normal_s2, rapido_s1, rapido_s2).
  - TripletSampler P=4 K=2 ahora puede muestrear sesión cruzada dentro del
    mismo batch → triplets cross-clothing entran en la supervisión.
  - Eval durante entrenamiento usa protocolo NR cross-session:
    gallery = normal_s1 (data/pkl/), probe = rapido_s2 (data/pkl_s2/).
    Es la métrica que bloqueaba Fase 5; early stopping por margen val NR.
  - Eval test final reporta ambos protocolos (NN y NR) cross-session sobre
    los 3 sujetos test intocados.
  - Checkpoint out: gaitbase_ft_multisession_best_iter{N}.pt.
  - Reporte JSON: reports/05_finetune_multisession_results.json.

Hparams idénticos a Fase 3 (para comparación limpia):
  SGD lr=0.01 mom=0.9 wd=5e-4, P=4 K=2, 30 frames, 1500 iter max,
  milestones [750, 1250], patience=3, TripletLoss(margin=0.2) + CE(scale=16).
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
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

PKL_ROOT_TRAIN = REPO / "data" / "pkl_multisession"
PKL_ROOT_S1 = REPO / "data" / "pkl"      # gallery cross-session
PKL_ROOT_S2 = REPO / "data" / "pkl_s2"   # probe cross-session

SPLITS_YAML = REPO / "configs" / "splits.yaml"
PARTITION_JSON = REPO / "configs" / "partition_finetune.json"
CKPT_OUT_DIR = REPO / "checkpoints" / "finetune"
REPORTS = REPO / "reports"

sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))


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


# ---------------------------- modelo ----------------------------

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
                sils = sils.unsqueeze(1)
            outs = self.Backbone(sils)
            outs = self.TP(outs, seqL, options={"dim": 2})[0]
            feat = self.HPP(outs)
            embed_1 = self.FCs(feat)
            embed_2, logits = self.BNNecks(embed_1)
            return embed_1, logits

    return GaitBaseFT()


def load_pretrained_strip_bnnecks(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state_clean = {k: v for k, v in state.items() if not k.startswith("BNNecks.")}
    missing, unexpected = model.load_state_dict(state_clean, strict=False)
    bnnecks_missing = [k for k in missing if k.startswith("BNNecks.")]
    other_missing = [k for k in missing if not k.startswith("BNNecks.")]
    print(f"[ckpt] cargados {len(state_clean)} tensores")
    print(f"[ckpt] missing BNNecks (esperado, se reinicializa): {len(bnnecks_missing)}")
    print(f"[ckpt] missing otros (deberia ser 0): {len(other_missing)} -> {other_missing[:3]}")
    print(f"[ckpt] unexpected: {len(unexpected)} -> {unexpected[:3]}")


# ---------------------------- losses ----------------------------

class TripletLossLocal(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        emb = embeddings.permute(2, 0, 1).contiguous().float()
        x2 = (emb ** 2).sum(-1, keepdim=True)
        dist = x2 + x2.transpose(1, 2) - 2 * emb @ emb.transpose(1, 2)
        dist = torch.sqrt(F.relu(dist) + 1e-12)

        same = (labels.unsqueeze(0) == labels.unsqueeze(1))
        diff = ~same
        p_dim, n_dim, _ = dist.size()
        ap = dist[:, same].view(p_dim, n_dim, -1, 1)
        an = dist[:, diff].view(p_dim, n_dim, 1, -1)
        loss_per_triplet = F.relu(ap - an + self.margin).view(p_dim, -1)
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
    def __init__(self, scale: float = 16.0, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.scale = scale
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        n, c, p = logits.size()
        logits = logits.float() * self.scale
        labels_rep = labels.unsqueeze(1).repeat(1, p)
        loss = F.cross_entropy(logits, labels_rep, label_smoothing=self.label_smoothing)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            accu = (pred == labels_rep).float().mean()
        return loss, {"ce_loss": float(loss.detach()), "ce_accuracy": float(accu.detach())}


# ---------------------------- data: train ----------------------------

def _patch_sampler_for_single_proc() -> None:
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

    tmp_log_dir = REPORTS / "_finetune_multisession_logs"
    tmp_log_dir.mkdir(parents=True, exist_ok=True)
    get_msg_mgr().init_manager(str(tmp_log_dir), log_to_file=False, log_iter=50)

    data_cfg = {
        "dataset_root": str(PKL_ROOT_TRAIN),
        "dataset_partition": str(PARTITION_JSON),
        "cache": True,
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
def _embed(model: nn.Module, pkl_path: Path, device: torch.device) -> np.ndarray:
    sils = load_silhouettes(pkl_path)
    x = torch.from_numpy(sils.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    seqL = [torch.tensor([x.shape[1]], device=device)]
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        emb_1, _ = model(x, seqL)
    return emb_1.float().squeeze(0).cpu().numpy().reshape(-1)


def _pkl_path(root: Path, subj: str, cond: str) -> Path | None:
    p = root / subj / cond / "090" / "seq00.pkl"
    return p if p.exists() else None


def cosine_matrix(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    p = probe / (np.linalg.norm(probe, axis=1, keepdims=True) + 1e-12)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    return p @ g.T


def _metrics(g: np.ndarray, p: np.ndarray) -> Dict:
    sim = cosine_matrix(p, g)
    n = g.shape[0]
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
    }


def evaluate_cross_session(model: nn.Module, subjects: List[str], device: torch.device) -> Dict:
    """gallery = s1/normal, probe = s2/normal (NN) y s2/rapido (NR)."""
    model.eval()

    g_embs_n, p_embs_nn, used_nn = [], [], []
    g_embs_r, p_embs_nr, used_nr = [], [], []

    for s in subjects:
        g_s1_n = _pkl_path(PKL_ROOT_S1, s, "normal")
        p_s2_n = _pkl_path(PKL_ROOT_S2, s, "normal")
        p_s2_r = _pkl_path(PKL_ROOT_S2, s, "rapido")
        if g_s1_n is None:
            print(f"[eval] falta gallery s1/normal para {s}, skip")
            continue
        if p_s2_n is not None:
            g_embs_n.append(_embed(model, g_s1_n, device))
            p_embs_nn.append(_embed(model, p_s2_n, device))
            used_nn.append(s)
        if p_s2_r is not None:
            g_embs_r.append(_embed(model, g_s1_n, device))
            p_embs_nr.append(_embed(model, p_s2_r, device))
            used_nr.append(s)

    out = {"subjects_nn": used_nn, "subjects_nr": used_nr}
    if used_nn:
        out["NN"] = _metrics(np.stack(g_embs_n, 0), np.stack(p_embs_nn, 0))
    if used_nr:
        out["NR"] = _metrics(np.stack(g_embs_r, 0), np.stack(p_embs_nr, 0))
    return out


# ---------------------------- loop principal ----------------------------

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
    ap.add_argument("--dry_run", action="store_true", help="50 iters, eval cada 25")
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

    train_subjects = load_split("train")
    val_subjects = load_split("val")
    test_subjects = load_split("test")
    print(f"[init] train={len(train_subjects)} val={len(val_subjects)} test={len(test_subjects)}")

    num_classes = len(train_subjects)
    model = build_model(num_classes).to(device)
    load_pretrained_strip_bnnecks(model, CKPT_PRETRAINED, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] modelo construido: {n_params:,} params")

    train_ds, train_loader = build_train_loader(
        batch_p=cfg.batch_p, batch_k=cfg.batch_k,
        frames_fixed=cfg.frames_fixed, num_workers=cfg.num_workers,
    )
    print(f"[data] train_ds: {len(train_ds)} secuencias, {len(train_ds.label_set)} sujetos")
    print(f"[data] types encontrados: {train_ds.types_set}")

    triplet = TripletLossLocal(margin=cfg.triplet_margin).to(device)
    ce = CrossEntropyLossLocal(scale=cfg.ce_scale, label_smoothing=cfg.ce_smoothing).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                            momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=list(cfg.milestones), gamma=cfg.gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print("\n[eval-pre] evaluando pre-finetune cross-session en val...")
    pre = evaluate_cross_session(model, val_subjects, device)
    if "NR" in pre:
        print(f"[eval-pre] NR rank1={pre['NR']['rank1']*100:.1f}% margen={pre['NR']['mean_margin']:+.4f}")
    if "NN" in pre:
        print(f"[eval-pre] NN rank1={pre['NN']['rank1']*100:.1f}% margen={pre['NN']['mean_margin']:+.4f}")

    # metrica para early stop: margen NR (la dura)
    def stop_metric(metrics: Dict) -> float:
        return metrics.get("NR", {}).get("mean_margin", -1.0)

    history = {"train": [], "val": [pre | {"iter": 0}]}
    best = {"margin": stop_metric(pre), "iter": 0, "state": None}
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

        fras = batch[0][0]
        labels = torch.tensor(batch[1], device=device, dtype=torch.long)

        seq_lens = [int(f.shape[0]) for f in fras]
        sils_np = np.concatenate(fras, axis=0).astype(np.float32) / 255.0
        sils = torch.from_numpy(sils_np).unsqueeze(0).to(device)
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
            v = evaluate_cross_session(model, val_subjects, device)
            v["iter"] = it
            history["val"].append(v)
            nr = v.get("NR", {})
            nn_m = v.get("NN", {})
            print(f"[eval@{it}] NR rank1={nr.get('rank1',0)*100:.1f}% margen={nr.get('mean_margin',0):+.4f} "
                  f"(min={nr.get('min_margin',0):+.4f}) | "
                  f"NN rank1={nn_m.get('rank1',0)*100:.1f}% margen={nn_m.get('mean_margin',0):+.4f}")
            cur = stop_metric(v)
            if cur > best["margin"]:
                best = {"margin": cur, "iter": it,
                        "state": {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}}
                patience_left = cfg.early_stop_patience
                print(f"[eval@{it}] >> nuevo mejor margen NR, guardando")
            else:
                patience_left -= 1
                print(f"[eval@{it}] sin mejora, patience={patience_left}")
                if patience_left <= 0:
                    print(f"[eval@{it}] early stopping (best NR margin {best['margin']:+.4f} @ iter {best['iter']})")
                    break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        ckpt_out = CKPT_OUT_DIR / f"gaitbase_ft_multisession_best_iter{best['iter']}.pt"
        torch.save({"model": best["state"], "iter": best["iter"], "margin_nr_val": best["margin"],
                    "config": cfg.__dict__}, ckpt_out)
        print(f"\n[ckpt] mejor modelo guardado en {ckpt_out.relative_to(REPO)}")

    print("\n[eval-test] evaluando mejor modelo cross-session en test (intocado)...")
    test_metrics = evaluate_cross_session(model, test_subjects, device)
    for proto in ("NN", "NR"):
        if proto in test_metrics:
            m = test_metrics[proto]
            print(f"[eval-test {proto}] rank1={m['rank1']*100:.1f}% rank5={m['rank5']*100:.1f}% "
                  f"margen={m['mean_margin']:+.4f} (min={m['min_margin']:+.4f})")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[done] VRAM pico: {peak:.0f} MB")
    total_dt = time.time() - t_start
    print(f"[done] tiempo total: {total_dt/60:.1f} min")

    report = {
        "config": cfg.__dict__,
        "pretrained_ckpt": str(CKPT_PRETRAINED.name),
        "pkl_root_train": str(PKL_ROOT_TRAIN),
        "train_n_sequences": len(train_ds),
        "best_iter": best["iter"],
        "best_val_margin_nr": best["margin"],
        "test": test_metrics,
        "val_history": history["val"],
        "train_history": history["train"],
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "vram_peak_mb": float(peak) if device.type == "cuda" else None,
        "total_minutes": round(total_dt / 60, 2),
    }
    out_json = REPORTS / "05_finetune_multisession_results.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[done] reporte: {out_json.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
