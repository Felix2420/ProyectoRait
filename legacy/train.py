"""
train.py v2 — Entrenamiento con Semi-Hard Triplet Mining + AMP Float16
Estrategia Anti-Overfitting:
  • Semi-Hard Negative Mining online (real, no aleatorio)
  • SkeletonAugment + GEIAugment agresivos
  • Dropout + BatchNorm + Weight Decay + Gradient Clipping
  • CosineAnnealingLR + Warmup
  • Early Stopping (patience configurable)
  • Guarda galería calibrada automáticamente al finalizar
"""

import os, random, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from model import (HybridGaitNet, TripletLoss, semi_hard_triplet_mining,
                   GalleryManager, SkeletonAugment, GEIAugment, EMBED_DIM)

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ── Dataset de muestras individuales (para mining online) ─────────────────────
class GaitSampleDataset(Dataset):
    """
    Carga muestras individuales (kp, gei, label).
    El mining de tripletas se hace en el loop de entrenamiento con todos los
    embeddings del batch → semi-hard negatives reales.
    """
    def __init__(self, feature_dir: str, subjects: list, augment: bool = True):
        self.samples   = []
        self.augment   = augment
        self.skel_aug  = SkeletonAugment()
        self.gei_aug   = GEIAugment()
        feat_path      = Path(feature_dir)

        for kp_f in sorted(feat_path.glob("*_kp.npy")):
            stem  = kp_f.stem.replace("_kp", "")
            lf    = feat_path / f"{stem}_label.npy"
            gf    = feat_path / f"{stem}_gei.npy"
            if not lf.exists() or not gf.exists():
                continue
            label = int(np.load(lf))
            if label not in subjects:
                continue
            self.samples.append({"kp": str(kp_f), "gei": str(gf), "label": label})

        assert len(self.samples) >= 2, "Sin muestras suficientes"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        kp  = np.load(s["kp"])
        gei = np.load(s["gei"])
        if self.augment:
            kp  = self.skel_aug(kp)
            gei = self.gei_aug(gei)
        kp_t  = torch.from_numpy(kp).float()
        gei_t = torch.from_numpy(gei).float().unsqueeze(0)
        return kp_t, gei_t, s["label"]


# ── Construcción y evaluación de galería ──────────────────────────────────────
@torch.no_grad()
def build_gallery(model, feature_dir: str, subjects: list, device, gm: GalleryManager):
    model.eval()
    gm.gallery.clear()
    feat_path = Path(feature_dir)
    by_label  = defaultdict(list)

    for kp_f in sorted(feat_path.glob("*_kp.npy")):
        stem  = kp_f.stem.replace("_kp", "")
        lf    = feat_path / f"{stem}_label.npy"
        gf    = feat_path / f"{stem}_gei.npy"
        if not lf.exists() or not gf.exists():
            continue
        label = int(np.load(lf))
        if label not in subjects:
            continue
        kp  = torch.from_numpy(np.load(kp_f)).float().unsqueeze(0).to(device)
        gei = torch.from_numpy(np.load(gf)).float().unsqueeze(0).unsqueeze(0).to(device)
        with autocast(enabled=(device.type == "cuda")):
            emb = model(kp, gei).squeeze(0).float().cpu().numpy()
        by_label[label].append(emb)

    for lbl, embs in by_label.items():
        gm.gallery[lbl] = np.stack(embs).mean(0)
    return gm


@torch.no_grad()
def evaluate(model, gm: GalleryManager, feature_dir: str,
             val_subjects: list, device, threshold: float):
    model.eval()
    correct = total = 0
    for kp_f in sorted(Path(feature_dir).glob("*_kp.npy")):
        stem = kp_f.stem.replace("_kp", "")
        lf   = Path(feature_dir) / f"{stem}_label.npy"
        gf   = Path(feature_dir) / f"{stem}_gei.npy"
        if not lf.exists() or not gf.exists():
            continue
        true_lbl = int(np.load(lf))
        if true_lbl not in val_subjects:
            continue
        kp  = torch.from_numpy(np.load(kp_f)).float().unsqueeze(0).to(device)
        gei = torch.from_numpy(np.load(gf)).float().unsqueeze(0).unsqueeze(0).to(device)
        with autocast(enabled=(device.type == "cuda")):
            emb = model(kp, gei).squeeze(0).float().cpu().numpy()
        pred, _ = gm.match(emb, threshold)
        if pred == true_lbl:
            correct += 1
        total += 1
    return correct / total if total else 0.0


# ── Warmup scheduler ─────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.opt           = optimizer
        self.warmup        = warmup_epochs
        self.total         = total_epochs
        self.base_lr       = base_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        e = self.current_epoch
        if e <= self.warmup:
            lr = self.base_lr * e / self.warmup
        else:
            import math
            progress = (e - self.warmup) / (self.total - self.warmup)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# ── Ciclo de entrenamiento ────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device  : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU     : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] VRAM    : {vram:.1f} GB")

    label_map    = np.load(os.path.join(args.features, "label_map.npy"), allow_pickle=True).item()
    all_subjects = list(label_map.keys())
    random.shuffle(all_subjects)
    val_size      = max(2, int(len(all_subjects) * 0.15))
    val_subj      = all_subjects[:val_size]
    train_subj    = all_subjects[val_size:]
    print(f"[INFO] Train   : {[label_map[s] for s in train_subj]}")
    print(f"[INFO] Val     : {[label_map[s] for s in val_subj]}")

    # Garantizar ≥2 muestras por sujeto en cada batch (para mining)
    train_ds = GaitSampleDataset(args.features, train_subj, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0,
                          pin_memory=(device.type == "cuda"), drop_last=True)

    # Detectar dimensión de keypoints automáticamente
    sample_kp = np.load(train_ds.samples[0]["kp"])
    kp_dim    = sample_kp.shape[-1]
    print(f"[INFO] KP dim  : {kp_dim}")

    model     = HybridGaitNet(kp_input_dim=kp_dim, embed_dim=EMBED_DIM).to(device)
    criterion = TripletLoss(margin=args.margin)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5,
                                      total_epochs=args.epochs, base_lr=args.lr)
    scaler    = GradScaler(enabled=(device.type == "cuda"))
    gm        = GalleryManager(os.path.join(args.save_dir, "gallery_embeddings.npy"))

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_model.pth")
    best_acc  = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for kp_b, gei_b, lbl_b in train_dl:
            kp_b  = kp_b.to(device)
            gei_b = gei_b.to(device)
            lbl_t = torch.tensor(lbl_b, device=device)

            optimizer.zero_grad()

            # ── Fase 1: obtener embeddings del batch ─────────────────────────
            with autocast(enabled=(device.type == "cuda")):
                embs = model(kp_b, gei_b)   # (B, 128)

            # ── Fase 2: Semi-Hard Mining (sin gradiente) ──────────────────────
            with torch.no_grad():
                ia, ip, in_ = semi_hard_triplet_mining(embs.float(), lbl_t, args.margin)

            if len(ia) == 0:
                continue

            ia_t  = torch.tensor(ia,  device=device)
            ip_t  = torch.tensor(ip,  device=device)
            in_t  = torch.tensor(in_, device=device)

            # ── Fase 3: Recalcular embeddings de la tripleta con gradiente ────
            with autocast(enabled=(device.type == "cuda")):
                e_a = model(kp_b[ia_t], gei_b[ia_t])
                e_p = model(kp_b[ip_t], gei_b[ip_t])
                e_n = model(kp_b[in_t], gei_b[in_t])
                loss = criterion(e_a, e_p, e_n)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        lr = scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # ── Evaluación periódica ───────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            gm = build_gallery(model, args.features, train_subj, device, gm)
            threshold = gm.calibrate_threshold(args.threshold_percentile)
            if threshold < 0.3:          # evitar umbral colapsado al inicio
                threshold = 0.8
                print(f"  [WARN] Umbral colapsado corregido → {threshold}")
            acc = evaluate(model, gm, args.features, val_subj, device, threshold)
            print(f"Epoch {epoch:03d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  lr={lr:.6f}  "
                  f"val_acc={acc*100:.1f}%  threshold={threshold:.4f}")

            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "val_acc":     best_acc,
                    "label_map":   label_map,
                    "threshold":   threshold,
                    "kp_dim":      kp_dim,
                }, ckpt_path)
                gm.save()
                print(f"  ✓ Mejor modelo guardado → {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early Stopping en epoch {epoch} (patience={args.patience})")
                    break
        else:
            print(f"Epoch {epoch:03d}/{args.epochs}  loss={avg_loss:.4f}  lr={lr:.6f}")

    print(f"\n✓ Entrenamiento finalizado. Mejor Val Acc: {best_acc*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento v2 Gait Recognition")
    parser.add_argument("--features",            type=str,   default="features")
    parser.add_argument("--save_dir",            type=str,   default="checkpoints")
    parser.add_argument("--epochs",              type=int,   default=120)
    parser.add_argument("--batch_size",          type=int,   default=8,
                        help="Recomendado 8 para GTX 1650")
    parser.add_argument("--lr",                  type=float, default=1e-3)
    parser.add_argument("--margin",              type=float, default=0.5)
    parser.add_argument("--threshold_percentile",type=float, default=95.0,
                        help="Percentil para calibración automática del umbral")
    parser.add_argument("--eval_every",          type=int,   default=5)
    parser.add_argument("--patience",            type=int,   default=20,
                        help="Early stopping patience (en evaluaciones)")
    args = parser.parse_args()
    train(args)
