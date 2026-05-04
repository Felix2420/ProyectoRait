"""
train.py v3 — Corregido para dataset pequeno (19 sujetos x 8 muestras)
Problemas resueltos vs v2:
  1. Split correcto: galeria incluye TODOS los sujetos al evaluar
  2. Batch-All Triplet Mining en vez de semi-hard (mas estable con pocos datos)
  3. Threshold fijo durante entrenamiento, calibracion solo al guardar
  4. batch_size=16 para maximizar triplets disponibles
"""

import os
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from model import (HybridGaitNet, GalleryManager,
                   SkeletonAugment, GEIAugment, EMBED_DIM)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ── Dataset individual ────────────────────────────────────────────────────────
class GaitSampleDataset(Dataset):
    def __init__(self, feature_dir, subjects, augment=True):
        self.samples  = []
        self.augment  = augment
        self.skel_aug = SkeletonAugment()
        self.gei_aug  = GEIAugment()

        for kp_f in sorted(Path(feature_dir).glob("*_kp.npy")):
            stem = kp_f.stem.replace("_kp", "")
            lf   = Path(feature_dir) / (stem + "_label.npy")
            gf   = Path(feature_dir) / (stem + "_gei.npy")
            if not lf.exists() or not gf.exists():
                continue
            label = int(np.load(lf))
            if label not in subjects:
                continue
            self.samples.append({"kp": str(kp_f), "gei": str(gf), "label": label})

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


# ── Batch-All Triplet Loss ────────────────────────────────────────────────────
def batch_all_triplet_loss(embeddings, labels, margin=0.5):
    """
    Genera TODAS las triplets validas del batch y promedia la loss.
    Mas estable que semi-hard con datasets pequenos.
    """
    dist      = torch.cdist(embeddings, embeddings, p=2)
    N         = embeddings.size(0)
    loss_vals = []

    for i in range(N):
        for j in range(N):
            if labels[i] != labels[j]:
                continue
            if i == j:
                continue
            for k in range(N):
                if labels[i] == labels[k]:
                    continue
                l = torch.relu(dist[i, j] - dist[i, k] + margin)
                if l.item() > 0:
                    loss_vals.append(l)

    if not loss_vals:
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
    return torch.stack(loss_vals).mean()


# ── Construccion de galeria ───────────────────────────────────────────────────
@torch.no_grad()
def build_gallery(model, feature_dir, subjects, device):
    model.eval()
    by_label = defaultdict(list)

    for kp_f in sorted(Path(feature_dir).glob("*_kp.npy")):
        stem  = kp_f.stem.replace("_kp", "")
        lf    = Path(feature_dir) / (stem + "_label.npy")
        gf    = Path(feature_dir) / (stem + "_gei.npy")
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

    return {lbl: np.stack(embs).mean(0) for lbl, embs in by_label.items()}


# ── Evaluacion ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, gallery, feature_dir, val_subjects, device, threshold):
    model.eval()
    correct = total = 0

    for kp_f in sorted(Path(feature_dir).glob("*_kp.npy")):
        stem     = kp_f.stem.replace("_kp", "")
        lf       = Path(feature_dir) / (stem + "_label.npy")
        gf       = Path(feature_dir) / (stem + "_gei.npy")
        if not lf.exists() or not gf.exists():
            continue
        true_lbl = int(np.load(lf))
        if true_lbl not in val_subjects:
            continue

        kp  = torch.from_numpy(np.load(kp_f)).float().unsqueeze(0).to(device)
        gei = torch.from_numpy(np.load(gf)).float().unsqueeze(0).unsqueeze(0).to(device)
        with autocast(enabled=(device.type == "cuda")):
            emb = model(kp, gei).squeeze(0).float().cpu().numpy()

        best_lbl, best_dist = -1, float("inf")
        for lbl, ref in gallery.items():
            d = float(np.linalg.norm(emb - ref))
            if d < best_dist:
                best_dist, best_lbl = d, lbl

        predicted = best_lbl if best_dist < threshold else -1
        if predicted == true_lbl:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc, "correct": correct, "total": total}


# ── Ciclo principal ───────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device   : " + str(device))
    if device.type == "cuda":
        print("GPU      : " + torch.cuda.get_device_name(0))

    label_map    = np.load(os.path.join(args.features, "label_map.npy"),
                           allow_pickle=True).item()
    all_subjects = list(label_map.keys())
    n            = len(all_subjects)

    # Split 80/20
    random.shuffle(all_subjects)
    val_size   = max(2, int(n * 0.20))
    val_subj   = all_subjects[:val_size]
    train_subj = all_subjects[val_size:]

    print("Train (" + str(len(train_subj)) + "): " + str([label_map[s] for s in train_subj]))
    print("Val   (" + str(len(val_subj))   + "): " + str([label_map[s] for s in val_subj]))

    train_ds = GaitSampleDataset(args.features, train_subj, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0,
                          pin_memory=(device.type == "cuda"))

    kp_dim = np.load(train_ds.samples[0]["kp"]).shape[-1]
    print("KP dim   : " + str(kp_dim) + "  |  Muestras train: " + str(len(train_ds)))

    model     = HybridGaitNet(kp_input_dim=kp_dim, embed_dim=EMBED_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=1, eta_min=1e-5)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_model.pth")
    gall_path = os.path.join(args.save_dir, "gallery_embeddings.npy")

    best_acc  = -1.0
    threshold = args.threshold

    print("")
    print("  Epoch    Loss         LR     Val Acc   Threshold")
    print("  " + "-" * 52)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for kp_b, gei_b, lbl_b in train_dl:
            kp_b  = kp_b.to(device)
            gei_b = gei_b.to(device)
            lbl_t = torch.tensor(lbl_b, device=device)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                embs = model(kp_b, gei_b)
                loss = batch_all_triplet_loss(embs.float(), lbl_t, args.margin)

            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        lr_now   = optimizer.param_groups[0]["lr"]

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            # Galeria con TODOS los sujetos para evaluacion justa
            gallery = build_gallery(model, args.features, all_subjects, device)
            metrics = evaluate(model, gallery, args.features,
                               val_subj, device, threshold)
            acc = metrics["accuracy"]

            status = (str(epoch).rjust(7) + "  " +
                      str(round(avg_loss, 4)).ljust(8) + "  " +
                      str(round(lr_now, 6)).ljust(10) + "  " +
                      str(round(acc * 100, 1)).rjust(6) + "%  " +
                      str(round(threshold, 4)).rjust(9) +
                      "  (" + str(metrics["correct"]) + "/" + str(metrics["total"]) + ")")
            print("  " + status)

            if acc > best_acc:
                best_acc = acc

                # Calibrar threshold al guardar
                if len(gallery) >= 2:
                    keys  = list(gallery.keys())
                    dists = []
                    for i in range(len(keys)):
                        for j in range(i + 1, len(keys)):
                            dists.append(np.linalg.norm(gallery[keys[i]] - gallery[keys[j]]))
                    save_t = max(float(np.percentile(dists, 40)), 0.3)
                else:
                    save_t = threshold

                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_acc":     best_acc,
                    "label_map":   label_map,
                    "threshold":   save_t,
                    "kp_dim":      kp_dim,
                }, ckpt_path)
                np.save(gall_path, gallery)
                print("  => Guardado! acc=" + str(round(best_acc * 100, 1)) +
                      "%  threshold=" + str(round(save_t, 4)))
        else:
            print("  " + str(epoch).rjust(7) + "  " +
                  str(round(avg_loss, 4)).ljust(8) + "  " +
                  str(round(lr_now, 6)))

    print("")
    print("=" * 54)
    print("Entrenamiento finalizado.")
    print("Mejor Val Acc : " + str(round(best_acc * 100, 1)) + "%")
    print("Checkpoint    : " + ckpt_path)
    print("Galeria       : " + gall_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento v3 Gait Recognition")
    parser.add_argument("--features",   type=str,   default="features")
    parser.add_argument("--save_dir",   type=str,   default="checkpoints")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--margin",     type=float, default=0.5)
    parser.add_argument("--threshold",  type=float, default=0.8)
    parser.add_argument("--eval_every", type=int,   default=10)
    args = parser.parse_args()
    train(args)