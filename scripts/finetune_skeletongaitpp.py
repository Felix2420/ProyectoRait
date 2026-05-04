"""Fine-tuning SkeletonGait++ (Fase 4) — dual-stream silueta + heatmap.

Diferencias clave vs finetune_gaitbase_multisession.py (Fase 3.5):
  - Modelo: SkeletonGaitPP (dual-stream, sin checkpoint público previo).
  - Datos: data/pkl_multimodal/ — 0_heatmap.pkl (T,2,64,64) + 1_sil.pkl (T,64,44).
  - Batch: cada muestra se procesa como (B, T, 3, 64, 44) tras recortar heatmap
    (CUTTING=10 → 64×64→64×44) y concatenar silueta.
  - Eval: lee de pkl_multimodal (no de pkl/ ni pkl_s2/).
  - Protocolo eval cross-session: gallery=normal_s1, probe NN=normal_s2, NR=rapido_s2.
  - Sin pretrained checkpoint público para SkeletonGait++; se entrena desde random init.

Hparams iguales a Fase 3.5 (para comparación limpia):
  SGD lr=0.01 mom=0.9 wd=5e-4, P=4 K=2, 30 frames, 1500 iter max,
  milestones [750, 1250], patience=3, TripletLoss(0.2) + CE(scale=16).

Uso:
    python scripts/finetune_skeletongaitpp.py
    python scripts/finetune_skeletongaitpp.py --dry_run
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
OPENGAIT = REPO / "third_party" / "OpenGait"

sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))

MULTIMODAL_ROOT = REPO / "data" / "pkl_multimodal"
SPLITS_YAML = REPO / "configs" / "splits.yaml"
PARTITION_JSON = REPO / "configs" / "partition_finetune.json"
CKPT_OUT_DIR = REPO / "checkpoints" / "finetune"
REPORTS = REPO / "reports"
GB_INIT_CKPT = REPO / "checkpoints" / "finetune" / "gaitbase_ft_multisession_best_iter1200.pt"

SIL_H, SIL_W = 64, 44
CUTTING = (SIL_H - SIL_W) // 2  # = 10 píxeles a recortar por lado del heatmap

MODEL_CFG = {
    "Backbone": {
        "in_channels": 3,
        "blocks": [1, 4, 4, 1],
        "C": 2,
    },
    "SeparateBNNecks": {
        "class_num": 13,
    },
    "use_emb2": False,
}


# ─────────────────────────── utilidades ───────────────────────────

def init_single_proc_dist() -> None:
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")
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


def load_split(name: str) -> List[str]:
    with SPLITS_YAML.open("r", encoding="utf-8") as f:
        s = yaml.safe_load(f)
    return sorted(s["subject_disjoint"][name])


# ─────────────────────────── modelo ───────────────────────────────

def _load_skeletongait_pp_class() -> type:
    """Carga SkeletonGaitPP desde skeletongait++.py via importlib (el '+' impide import normal)."""
    import importlib
    import importlib.util
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


def build_model() -> nn.Module:
    """Instancia SkeletonGaitPP sin BaseModel.__init__ (sin DDP/distributed)."""
    SkeletonGaitPP = _load_skeletongait_pp_class()
    model = SkeletonGaitPP.__new__(SkeletonGaitPP)
    nn.Module.__init__(model)
    model.training = True
    model.build_network(MODEL_CFG)
    return model


def _gb_to_sgpp_pairs() -> List[Tuple[str, str]]:
    """Mapping (src_key_gaitbase, dst_key_sgpp) de capas con shape compatible.
    No incluye stem heatmap, fusion, shortcut3d/sbn ni downsample 5D."""
    pairs: List[Tuple[str, str]] = [
        ("Backbone.forward_block.conv1.conv.weight", "sil_layer0.forward_block.0.weight"),
    ]
    for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
        pairs.append((f"Backbone.forward_block.bn1.{k}", f"sil_layer0.forward_block.1.{k}"))

    # layer1 (1 bloque) -> sil_layer1.forward_block.0
    for sub_conv, sub_bn in (("conv1", "bn1"), ("conv2", "bn2")):
        pairs.append((
            f"Backbone.forward_block.layer1.0.{sub_conv}.weight",
            f"sil_layer1.forward_block.0.{sub_conv}.weight",
        ))
        for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            pairs.append((
                f"Backbone.forward_block.layer1.0.{sub_bn}.{k}",
                f"sil_layer1.forward_block.0.{sub_bn}.{k}",
            ))

    # layer{2,3,4} bloque [0] (sin downsample 5D, sin shortcut3d/sbn)
    for L in (2, 3, 4):
        for sub_conv, sub_bn in (("conv1", "bn1"), ("conv2", "bn2")):
            pairs.append((
                f"Backbone.forward_block.layer{L}.0.{sub_conv}.weight",
                f"layer{L}.0.{sub_conv}.forward_block.0.weight",
            ))
            for k in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                pairs.append((
                    f"Backbone.forward_block.layer{L}.0.{sub_bn}.{k}",
                    f"layer{L}.0.{sub_conv}.forward_block.1.{k}",
                ))

    # FCs y BNNecks: mismo nombre
    for k in ("FCs.fc_bin", "BNNecks.fc_bin",
              "BNNecks.bn1d.weight", "BNNecks.bn1d.bias",
              "BNNecks.bn1d.running_mean", "BNNecks.bn1d.running_var",
              "BNNecks.bn1d.num_batches_tracked"):
        pairs.append((k, k))
    return pairs


def init_from_gaitbase(model: nn.Module, gb_ckpt_path: Path) -> List[str]:
    """Transfiere pesos compatibles GaitBase -> SkeletonGait++.
    Devuelve la lista de keys SGPP transferidas (uso en gating de warm-up)."""
    gb = torch.load(gb_ckpt_path, map_location="cpu", weights_only=False)
    gb_state = gb.get("model", gb)
    sgpp_state = model.state_dict()
    new_state = {k: v.clone() for k, v in sgpp_state.items()}
    transferred: List[str] = []
    skipped_shape = []
    for sk, dk in _gb_to_sgpp_pairs():
        if sk not in gb_state or dk not in new_state:
            continue
        if gb_state[sk].shape != new_state[dk].shape:
            skipped_shape.append((sk, dk))
            continue
        new_state[dk] = gb_state[sk].clone()
        transferred.append(dk)
    model.load_state_dict(new_state, strict=True)
    total_p = sum(v.numel() for v in sgpp_state.values())
    transf_p = sum(sgpp_state[k].numel() for k in transferred)
    print(f"[init] desde {gb_ckpt_path.name}: {len(transferred)} keys ({transf_p/total_p*100:.1f}% de params)")
    if skipped_shape:
        print(f"[init] skipped por shape: {len(skipped_shape)}")
    return transferred


def _param_name_from_state_key(state_key: str) -> str:
    """Las keys del state_dict pueden incluir 'num_batches_tracked' (no son params)
    o BN running_*. Para gating de requires_grad nos quedamos con los param tensors."""
    # named_parameters NO incluye running_mean/var/num_batches_tracked.
    return state_key


def freeze_silhouette_branch(model: nn.Module, transferred_keys: set, freeze: bool) -> int:
    """Congela/descongela parametros que vinieron del init de GaitBase.
    Vuelve el numero de tensores afectados."""
    n = 0
    for name, p in model.named_parameters():
        if name in transferred_keys:
            p.requires_grad = not freeze
            n += 1
    return n


# ─────────────────────────── losses ───────────────────────────────

class TripletLossLocal(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # embeddings: (B, C, P)
        emb = embeddings.permute(2, 0, 1).contiguous().float()  # (P, B, C)
        x2 = (emb ** 2).sum(-1, keepdim=True)
        dist = x2 + x2.transpose(1, 2) - 2 * emb @ emb.transpose(1, 2)
        dist = torch.sqrt(F.relu(dist) + 1e-12)

        same = (labels.unsqueeze(0) == labels.unsqueeze(1))
        diff = ~same
        p_dim, n_dim, _ = dist.size()
        ap = dist[:, same].view(p_dim, n_dim, -1, 1)
        an = dist[:, diff].view(p_dim, n_dim, 1, -1)
        loss_per_triplet = F.relu(ap - an + self.margin).view(p_dim, -1)
        nz = (loss_per_triplet > 0).sum(dim=1).float()
        per_part = loss_per_triplet.sum(dim=1) / (nz + 1e-9)
        per_part = torch.where(nz > 0, per_part, torch.zeros_like(per_part))
        loss = per_part.mean()
        return loss, {
            "triplet_loss": float(loss.detach()),
            "triplet_active_frac": float((loss_per_triplet > 0).float().mean().detach()),
            "triplet_mean_dist": float(dist.mean().detach()),
        }


class CrossEntropyLossLocal(nn.Module):
    def __init__(self, scale: float = 16.0, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.scale = scale
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        n, c, p = logits.size()
        logits = logits.float() * self.scale
        labels_rep = labels.unsqueeze(1).repeat(1, p)
        loss = F.cross_entropy(logits, labels_rep, label_smoothing=self.label_smoothing)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            accu = (pred == labels_rep).float().mean()
        return loss, {"ce_loss": float(loss.detach()), "ce_accuracy": float(accu.detach())}


# ─────────────────────────── data: train ──────────────────────────

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

    tmp_log_dir = REPORTS / "_finetune_skeletongaitpp_logs"
    tmp_log_dir.mkdir(parents=True, exist_ok=True)
    get_msg_mgr().init_manager(str(tmp_log_dir), log_to_file=False, log_iter=50)

    data_cfg = {
        "dataset_root": str(MULTIMODAL_ROOT),
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


def build_batch_tensor(hms: List[np.ndarray], sils: List[np.ndarray],
                       device: torch.device) -> torch.Tensor:
    """Recorta heatmaps y concatena siluetas → (B, T, 3, 64, 44) float32."""
    B = len(hms)
    T = hms[0].shape[0]
    buf = np.empty((B, T, 3, SIL_H, SIL_W), dtype=np.float32)
    for i, (h, s) in enumerate(zip(hms, sils)):
        h_crop = h[:, :, :, CUTTING:-CUTTING].astype(np.float32) / 255.0
        s_exp = s[:, np.newaxis, :, :].astype(np.float32) / 255.0
        buf[i] = np.concatenate([h_crop, s_exp], axis=1)
    return torch.from_numpy(buf).to(device)


# ─────────────────────────── eval ─────────────────────────────────

@torch.no_grad()
def _embed(model: nn.Module, seq_dir: Path,
           device: torch.device) -> Optional[np.ndarray]:
    """Lee 0_heatmap.pkl + 1_sil.pkl, devuelve embedding plano (C*P,)."""
    hm_path = seq_dir / "0_heatmap.pkl"
    sil_path = seq_dir / "1_sil.pkl"
    if not hm_path.exists() or not sil_path.exists():
        return None
    with hm_path.open("rb") as f:
        hm = pickle.load(f)   # (T, 2, 64, 64)
    with sil_path.open("rb") as f:
        sil = pickle.load(f)  # (T, 64, 44)

    h_crop = hm[:, :, :, CUTTING:-CUTTING].astype(np.float32) / 255.0
    s_exp = sil[:, np.newaxis, :, :].astype(np.float32) / 255.0
    combined = np.concatenate([h_crop, s_exp], axis=1)  # (T, 3, 64, 44)
    x = torch.from_numpy(combined).unsqueeze(0).to(device)  # (1, T, 3, 64, 44)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        retval = model(([x], None, None, None, None))
    embed = retval["inference_feat"]["embeddings"]  # (1, 256, 16)
    return embed.float().squeeze(0).cpu().numpy().reshape(-1)


def _seq_dir(subj: str, cond_session: str) -> Path:
    return MULTIMODAL_ROOT / subj / cond_session / "090"


def cosine_matrix(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    p = probe / (np.linalg.norm(probe, axis=1, keepdims=True) + 1e-12)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    return p @ g.T


def _metrics(g: np.ndarray, p: np.ndarray) -> Dict:
    sim = cosine_matrix(p, g)
    n = g.shape[0]
    order = np.argsort(-sim, axis=1)
    rank_pos = np.where(order == np.arange(n)[:, None])[1]
    sim_correct = sim[np.arange(n), np.arange(n)]
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


def evaluate_cross_session(model: nn.Module, subjects: List[str],
                           device: torch.device) -> Dict:
    """gallery = normal_s1, probe NN = normal_s2, probe NR = rapido_s2."""
    model.eval()
    g_n, p_nn, used_nn = [], [], []
    g_r, p_nr, used_nr = [], [], []

    for s in subjects:
        gal_emb = _embed(model, _seq_dir(s, "normal_s1"), device)
        if gal_emb is None:
            print(f"[eval] falta normal_s1 para {s}, skip")
            continue

        nn_emb = _embed(model, _seq_dir(s, "normal_s2"), device)
        if nn_emb is not None:
            g_n.append(gal_emb)
            p_nn.append(nn_emb)
            used_nn.append(s)

        nr_emb = _embed(model, _seq_dir(s, "rapido_s2"), device)
        if nr_emb is not None:
            g_r.append(gal_emb)
            p_nr.append(nr_emb)
            used_nr.append(s)

    out: Dict = {"subjects_nn": used_nn, "subjects_nr": used_nr}
    if used_nn:
        out["NN"] = _metrics(np.stack(g_n, 0), np.stack(p_nn, 0))
    if used_nr:
        out["NR"] = _metrics(np.stack(g_r, 0), np.stack(p_nr, 0))
    return out


# ─────────────────────────── config ───────────────────────────────

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
    eval_iter: int = 100
    triplet_margin: float = 0.2
    triplet_weight: float = 1.0
    ce_weight: float = 1.0
    ce_scale: float = 16.0
    ce_smoothing: float = 0.1
    early_stop_patience: int = 6
    early_stop_min_iter: int = 1250
    warmup_iter: int = 200
    num_workers: int = 0


# ─────────────────────────── main ─────────────────────────────────

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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        total_vram = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
        print(f"[init] GPU={torch.cuda.get_device_name(0)}  VRAM={total_vram} MB")

    CKPT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    train_subjects = load_split("train")
    val_subjects = load_split("val")
    test_subjects = load_split("test")
    print(f"[init] train={len(train_subjects)} val={len(val_subjects)} test={len(test_subjects)}")

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] SkeletonGait++ params={n_params:,}")
    transferred_keys = init_from_gaitbase(model, GB_INIT_CKPT)
    transferred_set = set(transferred_keys)

    # Warm-up: congelar la rama silueta + heads heredados de GaitBase.
    # Solo se entrenan rama heatmap (map_*), fusion y los 3D shortcuts/sbn.
    n_frozen = freeze_silhouette_branch(model, transferred_set, freeze=True)
    print(f"[warmup] congelados {n_frozen} tensores heredados de GaitBase "
          f"durante {cfg.warmup_iter} iter")

    train_ds, train_loader = build_train_loader(
        batch_p=cfg.batch_p, batch_k=cfg.batch_k,
        frames_fixed=cfg.frames_fixed, num_workers=cfg.num_workers,
    )
    print(f"[data] train_ds: {len(train_ds)} secuencias, {len(train_ds.label_set)} sujetos")
    print(f"[data] types encontrados: {train_ds.types_set}")

    triplet = TripletLossLocal(margin=cfg.triplet_margin).to(device)
    ce = CrossEntropyLossLocal(scale=cfg.ce_scale, label_smoothing=cfg.ce_smoothing).to(device)

    def _build_optim(m):
        params = [p for p in m.parameters() if p.requires_grad]
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay)

    optim = _build_optim(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=list(cfg.milestones), gamma=cfg.gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print("\n[eval-pre] evaluando pre-finetune cross-session en val...")
    pre = evaluate_cross_session(model, val_subjects, device)
    if "NR" in pre:
        print(f"[eval-pre] NR rank1={pre['NR']['rank1']*100:.1f}%  margen={pre['NR']['mean_margin']:+.4f}")
    if "NN" in pre:
        print(f"[eval-pre] NN rank1={pre['NN']['rank1']*100:.1f}%  margen={pre['NN']['mean_margin']:+.4f}")

    def stop_metric(metrics: Dict) -> float:
        return metrics.get("NR", {}).get("mean_margin", -1.0)

    history: Dict = {"train": [], "val": [pre | {"iter": 0}]}
    best = {"margin": stop_metric(pre), "iter": 0, "state": None}
    patience_left = cfg.early_stop_patience
    log_buf = {"triplet": 0.0, "ce": 0.0, "active": 0.0, "accu": 0.0, "n": 0}
    train_iter = iter(train_loader)
    t_start = time.time()

    for it in range(1, cfg.total_iter + 1):
        # Fin del warm-up: descongelar rama silueta y rebuildear el optimizer
        # (manteniendo el mismo schedule de milestones).
        if it == cfg.warmup_iter + 1 and cfg.warmup_iter > 0:
            n_un = freeze_silhouette_branch(model, transferred_set, freeze=False)
            optim = _build_optim(model)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optim, milestones=list(cfg.milestones), gamma=cfg.gamma)
            # avanzar el scheduler hasta `it` para mantener el schedule global
            for _ in range(it):
                scheduler.step()
            print(f"[warmup-end @ {it}] descongelados {n_un} tensores; optimizer rebuildeado")

        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # batch[0][0] = lista B de heatmaps (T, 2, 64, 64)
        # batch[0][1] = lista B de siluetas  (T, 64, 44)
        hms = batch[0][0]
        sils_raw = batch[0][1]
        labels = torch.tensor(batch[1], device=device, dtype=torch.long)

        x = build_batch_tensor(hms, sils_raw, device)  # (B, T, 3, 64, 44)

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            retval = model(([x], None, None, None, None))
            embed_1 = retval["training_feat"]["triplet"]["embeddings"]  # (B, 256, 16)
            logits = retval["training_feat"]["softmax"]["logits"]        # (B, 13, 16)
            l_tri, info_tri = triplet(embed_1, labels)
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
                "triplet": log_buf["triplet"] / n, "ce": log_buf["ce"] / n,
                "active": log_buf["active"] / n, "ce_acc": log_buf["accu"] / n,
            })
            log_buf = {"triplet": 0.0, "ce": 0.0, "active": 0.0, "accu": 0.0, "n": 0}

        if it % cfg.eval_iter == 0:
            v = evaluate_cross_session(model, val_subjects, device)
            v["iter"] = it
            history["val"].append(v)
            nr = v.get("NR", {})
            nn_m = v.get("NN", {})
            print(f"[eval@{it}] NR rank1={nr.get('rank1', 0)*100:.1f}%  "
                  f"margen={nr.get('mean_margin', 0):+.4f} "
                  f"(min={nr.get('min_margin', 0):+.4f}) | "
                  f"NN rank1={nn_m.get('rank1', 0)*100:.1f}%  "
                  f"margen={nn_m.get('mean_margin', 0):+.4f}")
            cur = stop_metric(v)
            if cur > best["margin"]:
                best = {
                    "margin": cur, "iter": it,
                    "state": {k: t.detach().cpu().clone()
                               for k, t in model.state_dict().items()},
                }
                patience_left = cfg.early_stop_patience
                print(f"[eval@{it}] >> nuevo mejor margen NR, guardando")
            else:
                patience_left -= 1
                print(f"[eval@{it}] sin mejora, patience={patience_left}")
                if patience_left <= 0:
                    if it < cfg.early_stop_min_iter:
                        print(f"[eval@{it}] patience agotado pero it<{cfg.early_stop_min_iter}: "
                              f"NO se corta (guardia post-segundo-milestone)")
                        patience_left = cfg.early_stop_patience
                    else:
                        print(f"[eval@{it}] early stopping "
                              f"(best NR margin {best['margin']:+.4f} @ iter {best['iter']})")
                        break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        ckpt_out = CKPT_OUT_DIR / f"skeletongaitpp_best_iter{best['iter']}.pt"
        torch.save({
            "model": best["state"],
            "iter": best["iter"],
            "margin_nr_val": best["margin"],
            "model_cfg": MODEL_CFG,
            "config": cfg.__dict__,
        }, ckpt_out)
        print(f"\n[ckpt] mejor modelo guardado en {ckpt_out.relative_to(REPO)}")
    else:
        print("\n[warn] no se encontró mejora — no se guarda checkpoint")

    print("\n[eval-test] evaluando mejor modelo cross-session en test (intocado)...")
    test_metrics = evaluate_cross_session(model, test_subjects, device)
    for proto in ("NN", "NR"):
        if proto in test_metrics:
            m = test_metrics[proto]
            print(f"[eval-test {proto}] rank1={m['rank1']*100:.1f}%  "
                  f"rank5={m['rank5']*100:.1f}%  "
                  f"margen={m['mean_margin']:+.4f} (min={m['min_margin']:+.4f})")

    peak_mb: Optional[float] = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"[done] VRAM pico: {peak_mb:.0f} MB")

    total_dt = time.time() - t_start
    print(f"[done] tiempo total: {total_dt / 60:.1f} min")

    report = {
        "config": cfg.__dict__,
        "model_cfg": MODEL_CFG,
        "multimodal_root": str(MULTIMODAL_ROOT),
        "train_n_sequences": len(train_ds),
        "best_iter": best["iter"],
        "best_val_margin_nr": best["margin"],
        "test": test_metrics,
        "val_history": history["val"],
        "train_history": history["train"],
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "vram_peak_mb": peak_mb,
        "total_minutes": round(total_dt / 60, 2),
    }
    out_json = REPORTS / "09_finetune_skeletongaitpp_results.json"
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[done] reporte: {out_json.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
