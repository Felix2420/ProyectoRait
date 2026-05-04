"""Entrenamiento ST-GCN ligero sobre keypoints COCO-17 (Fase 4 — Ruta B).

Datos:
  - data/processed/<subj>/<cond>/keypoints.npy  (sesion s1)
  - data/processed_s2/<subj>/<cond>/keypoints.npy  (sesion s2)
  - keypoints shape: (T, 17, 3) float32 normalizados (centrado en cadera, escala por altura)

Splits subject-disjoint segun configs/splits.yaml:
  - train: 13 sujetos (s1 + s2 × {normal, rapido})
  - val:   3 sujetos
  - test:  3 sujetos (no se toca aqui — la eval final es LOSO open-set)

Loss: triplet (margin=0.2) + 0.1 × CE.
Sampling: P×K (4 sujetos × 4 muestras) por batch, ventana temporal T=30 con augmentation.
Early-stop en val rank-1.

Uso:
    C:/Proyecto3/venv/Scripts/python.exe scripts/train_stgcn.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.stgcn_lite import STGCNLite, COCO17_FLIP_PAIRS  # noqa: E402

S1_ROOT = REPO / "data" / "processed"
S2_ROOT = REPO / "data" / "processed_s2"
SPLITS = REPO / "configs" / "splits.yaml"
CKPT_DIR = REPO / "checkpoints" / "finetune"
REPORTS = REPO / "reports"

SEED = 42
WIN_T = 30          # ventana temporal en frames
P_SUBJ = 4          # sujetos por batch
K_SAMP = 4          # muestras por sujeto
N_ITERS = 1000
LR = 1e-3
WD = 5e-4
TRIPLET_MARGIN = 0.2
CE_WEIGHT = 0.1
EVAL_EVERY = 50
PATIENCE = 5
EMB_DIM = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits() -> Dict[str, List[str]]:
    with SPLITS.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "train": cfg["subject_disjoint"]["train"],
        "val": cfg["subject_disjoint"]["val"],
        "test": cfg["subject_disjoint"]["test"],
    }


def collect_sequences(subjects: List[str]) -> List[Dict]:
    """Devuelve lista de dicts con keys: subj, cond, sess, kp (T,17,3)."""
    out = []
    for sess_tag, root in (("s1", S1_ROOT), ("s2", S2_ROOT)):
        for subj in subjects:
            sub_dir = root / subj
            if not sub_dir.is_dir():
                continue
            for cond_dir in sub_dir.iterdir():
                if not cond_dir.is_dir():
                    continue
                kp_p = cond_dir / "keypoints.npy"
                if not kp_p.exists():
                    continue
                kp = np.load(kp_p).astype(np.float32)
                out.append({
                    "subj": subj,
                    "cond": cond_dir.name,
                    "sess": sess_tag,
                    "kp": kp,
                })
    return out


# ─────────── Augmentation ─────────────────────────────────────────

def aug_flip_horizontal(x: np.ndarray) -> np.ndarray:
    """x: (T,17,3). Niega la coordenada X y reordena pares L↔R."""
    x = x.copy()
    x[..., 0] = -x[..., 0]
    for a, b in COCO17_FLIP_PAIRS:
        x[:, [a, b]] = x[:, [b, a]]
    return x


def aug_jitter(x: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    noise = np.random.randn(*x.shape[:2], 2).astype(np.float32) * sigma
    out = x.copy()
    out[..., :2] += noise
    return out


def aug_lowconf_dropout(x: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    out = x.copy()
    mask = out[..., 2] < thresh
    out[mask, :2] = 0.0
    return out


def temporal_crop(kp: np.ndarray, win: int, train: bool) -> np.ndarray:
    """Crop temporal aleatorio (train) o central (eval). Pad si T<win."""
    T = kp.shape[0]
    if T < win:
        pad = win - T
        kp = np.concatenate([kp, np.zeros((pad, *kp.shape[1:]), dtype=kp.dtype)], axis=0)
        return kp
    if train:
        s = random.randint(0, T - win)
    else:
        s = (T - win) // 2
    return kp[s:s + win]


# ─────────── Datasets ─────────────────────────────────────────────

class GaitPoseDataset(Dataset):
    """Sampler P×K: indexa por sujeto, devuelve K muestras random por subject."""

    def __init__(self, sequences: List[Dict], subj_to_label: Dict[str, int],
                 win: int = WIN_T, train: bool = True):
        self.win = win
        self.train = train
        self.subj_to_label = subj_to_label
        self.by_subject: Dict[str, List[Dict]] = {}
        for s in sequences:
            self.by_subject.setdefault(s["subj"], []).append(s)
        self.subjects = sorted(self.by_subject.keys())

    def __len__(self):
        return len(self.subjects)

    def sample_clip(self, seq: Dict) -> np.ndarray:
        kp = temporal_crop(seq["kp"], self.win, self.train)
        if self.train:
            if random.random() < 0.5:
                kp = aug_flip_horizontal(kp)
            kp = aug_jitter(kp, 0.01)
        kp = aug_lowconf_dropout(kp, 0.3)
        return kp  # (T, 17, 3)

    def get_K(self, subj: str, K: int) -> torch.Tensor:
        seqs = self.by_subject[subj]
        clips = []
        for _ in range(K):
            seq = random.choice(seqs)
            clips.append(self.sample_clip(seq))
        arr = np.stack(clips, axis=0)         # (K, T, 17, 3)
        arr = arr.transpose(0, 3, 1, 2)        # (K, 3, T, 17)
        return torch.from_numpy(arr)


def make_pk_batch(ds: GaitPoseDataset, P: int, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    subjs = random.sample(ds.subjects, P)
    xs, labels = [], []
    for s in subjs:
        xs.append(ds.get_K(s, K))
        labels.extend([ds.subj_to_label[s]] * K)
    x = torch.cat(xs, dim=0)             # (P*K, 3, T, 17)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


# ─────────── Triplet loss (BatchAll) ──────────────────────────────

def batch_all_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Triplet BatchAll con embeddings ya BNNecked. Distancia cosine."""
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.t()
    dist = 1.0 - sim                      # cosine distance
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff = ~same
    # Para cada anchor, candidates positivos y negativos
    losses = []
    for a in range(emb.size(0)):
        pos_idx = torch.where(same[a])[0]
        pos_idx = pos_idx[pos_idx != a]
        neg_idx = torch.where(diff[a])[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        d_ap = dist[a, pos_idx].unsqueeze(1)   # (P,1)
        d_an = dist[a, neg_idx].unsqueeze(0)   # (1,N)
        l = F.relu(d_ap - d_an + margin)
        # mantener solo triplets activos
        active = l > 0
        if active.any():
            losses.append(l[active].mean())
    if not losses:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    return torch.stack(losses).mean()


# ─────────── Eval rank-1 sobre val ────────────────────────────────

@torch.no_grad()
def eval_rank1(model: nn.Module, sequences: List[Dict], device: torch.device, win: int) -> Tuple[float, int]:
    """Rank-1 leave-one-out cross-session sobre val.
    Gallery: una secuencia random por sujeto. Probes: el resto."""
    model.eval()
    embs = []
    labels = []
    subjs = sorted({s["subj"] for s in sequences})
    s_to_l = {s: i for i, s in enumerate(subjs)}
    for seq in sequences:
        kp = temporal_crop(seq["kp"], win, train=False)
        kp = aug_lowconf_dropout(kp, 0.3)
        x = torch.from_numpy(kp.transpose(2, 0, 1)).unsqueeze(0).to(device)  # (1,3,T,17)
        e, _ = model(x)
        embs.append(F.normalize(e, dim=1).squeeze(0).cpu().numpy())
        labels.append(s_to_l[seq["subj"]])
    E = np.stack(embs)
    L = np.array(labels)

    # Para cada secuencia: compararla contra todas las demas y ver si rank-1 == su sujeto
    n = len(E)
    sim = E @ E.T
    np.fill_diagonal(sim, -np.inf)
    pred = L[np.argmax(sim, axis=1)]
    correct = int((pred == L).sum())
    model.train()
    return correct / max(1, n), n


# ─────────── Main ─────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}  seed={SEED}")

    splits = load_splits()
    print(f"[info] splits  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    train_seqs = collect_sequences(splits["train"])
    val_seqs = collect_sequences(splits["val"])
    print(f"[info] secuencias  train={len(train_seqs)}  val={len(val_seqs)}")

    train_subjs = sorted({s["subj"] for s in train_seqs})
    subj_to_label = {s: i for i, s in enumerate(train_subjs)}
    n_classes = len(train_subjs)
    print(f"[info] n_classes (train) = {n_classes}")

    train_ds = GaitPoseDataset(train_seqs, subj_to_label, win=WIN_T, train=True)

    model = STGCNLite(in_channels=3, num_nodes=17, emb_dim=EMB_DIM,
                      n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] params = {n_params/1e6:.3f}M")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ITERS)
    ce = nn.CrossEntropyLoss()

    best_r1 = 0.0
    best_iter = 0
    no_improve = 0
    history = []

    t_start = time.time()
    model.train()
    for it in range(1, N_ITERS + 1):
        x, y = make_pk_batch(train_ds, P_SUBJ, K_SAMP)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        emb, logits = model(x)
        loss_tri = batch_all_triplet_loss(emb, y, TRIPLET_MARGIN)
        loss_ce = ce(logits, y)
        loss = loss_tri + CE_WEIGHT * loss_ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        if it % EVAL_EVERY == 0 or it == 1:
            r1, n = eval_rank1(model, val_seqs, device, WIN_T)
            elapsed = time.time() - t_start
            print(f"[iter {it:4d}/{N_ITERS}] loss={loss.item():.4f} (tri={loss_tri.item():.4f} ce={loss_ce.item():.4f}) "
                  f"lr={sched.get_last_lr()[0]:.2e}  val_rank1={r1*100:.2f}% (n={n})  t={elapsed:.0f}s")
            history.append({"iter": it, "loss": float(loss.item()),
                            "tri": float(loss_tri.item()), "ce": float(loss_ce.item()),
                            "val_rank1": r1})
            if r1 > best_r1:
                best_r1, best_iter = r1, it
                no_improve = 0
                torch.save({"model": model.state_dict(),
                            "subj_to_label": subj_to_label,
                            "iter": it, "val_rank1": r1,
                            "win_t": WIN_T, "emb_dim": EMB_DIM},
                           CKPT_DIR / "stgcn_lite_best.pt")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"[early-stop] sin mejora en {PATIENCE} evals (best_r1={best_r1*100:.2f}% @ iter {best_iter})")
                    break

    print(f"\n[done] best_val_rank1 = {best_r1*100:.2f}% @ iter {best_iter}")
    out = {
        "best_iter": best_iter,
        "best_val_rank1": best_r1,
        "n_classes": n_classes,
        "n_params": n_params,
        "history": history,
        "config": {
            "win_t": WIN_T, "P": P_SUBJ, "K": K_SAMP,
            "lr": LR, "wd": WD, "n_iters": N_ITERS,
            "triplet_margin": TRIPLET_MARGIN, "ce_weight": CE_WEIGHT,
            "emb_dim": EMB_DIM,
        },
    }
    (REPORTS / "12_train_stgcn_lite.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> reports/12_train_stgcn_lite.json")
    print(f"-> checkpoints/finetune/stgcn_lite_best.pt")


if __name__ == "__main__":
    main()
