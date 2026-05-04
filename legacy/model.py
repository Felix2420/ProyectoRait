"""
model.py v2 — Arquitectura Híbrida de Gait Recognition
  Rama A : CNN ligera → GEI (128×88)
  Rama B : LSTM Bidireccional → Keypoints (T×D)
  Fusión : MLP → Embedding L2-normalizado (128-d)
  Extra  : utilidad para guardar/cargar galería en .npy
Optimizado para GTX 1650 (4 GB VRAM) con AMP Float16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────────
GEI_H, GEI_W = 128, 88
KP_DIM_MP    = 66    # MediaPipe 33×2
KP_DIM_YOLO  = 34    # YOLO COCO-17×2
SEQ_LEN      = 60
EMBED_DIM    = 128


# ── Rama A: CNN para GEI ──────────────────────────────────────────────────────
class GEI_CNN(nn.Module):
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128,128,3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, out_features),
            nn.ReLU(True),
        )

    def forward(self, x):          # (B,1,H,W)
        return self.head(self.features(x).flatten(1))


# ── Rama B: LSTM Bidireccional para keypoints ─────────────────────────────────
class KeypointLSTM(nn.Module):
    def __init__(self, input_dim: int = KP_DIM_MP, hidden_dim: int = 128,
                 num_layers: int = 2, out_features: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, 128)
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=0.3 if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden_dim * 2, out_features), nn.ReLU(True))

    def forward(self, x):          # (B,T,D)
        x = F.relu(self.proj(x))
        _, (hn, _) = self.lstm(x)
        return self.head(torch.cat([hn[-2], hn[-1]], dim=1))


# ── Modelo híbrido ────────────────────────────────────────────────────────────
class HybridGaitNet(nn.Module):
    """
    Uso:
        emb = model(kp, gei)   # (B, EMBED_DIM) L2-normalizado
    """
    def __init__(self, kp_input_dim: int = KP_DIM_MP, gei_feat: int = 256,
                 kp_feat: int = 256, embed_dim: int = EMBED_DIM, dropout: float = 0.4):
        super().__init__()
        self.gei_branch = GEI_CNN(gei_feat)
        self.kp_branch  = KeypointLSTM(kp_input_dim, out_features=kp_feat)
        self.fusion = nn.Sequential(
            nn.Linear(gei_feat + kp_feat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
        )

    def forward(self, kp, gei):    # kp:(B,T,D)  gei:(B,1,H,W)
        return F.normalize(self.fusion(torch.cat([self.gei_branch(gei), self.kp_branch(kp)], 1)), p=2, dim=1)


# ── Triplet Loss ──────────────────────────────────────────────────────────────
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        return F.relu(F.pairwise_distance(a,p) - F.pairwise_distance(a,n) + self.margin).mean()


# ── Semi-Hard Negative Mining ─────────────────────────────────────────────────
def semi_hard_triplet_mining(embeddings: torch.Tensor, labels: torch.Tensor,
                              margin: float = 0.5):
    """
    Para cada anchor selecciona:
      - positive : mismo sujeto, MÁXIMA distancia (hard positive)
      - negative : distinto sujeto, dist > dist_ap pero < dist_ap + margin (semi-hard)
    Returns: índices (idx_a, idx_p, idx_n)
    """
    dist  = torch.cdist(embeddings, embeddings, p=2)  # (N,N)
    N     = len(labels)
    idx_a, idx_p, idx_n = [], [], []

    for i in range(N):
        pos_mask = (labels == labels[i]) & (torch.arange(N, device=labels.device) != i)
        neg_mask = labels != labels[i]
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        # Hard positive
        j = dist[i][pos_mask].argmax()
        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        p_idx = pos_indices[j].item()
        d_ap  = dist[i, p_idx].item()

        # Semi-hard negative: d_an > d_ap  AND  d_an < d_ap + margin
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        d_an  = dist[i][neg_indices]
        sh    = (d_an > d_ap) & (d_an < d_ap + margin)
        if sh.sum() == 0:
            # Fallback: negativo más cercano
            n_idx = neg_indices[d_an.argmin()].item()
        else:
            n_idx = neg_indices[sh.nonzero(as_tuple=True)[0][0]].item()

        idx_a.append(i); idx_p.append(p_idx); idx_n.append(n_idx)

    return idx_a, idx_p, idx_n


# ── Galería de embeddings ─────────────────────────────────────────────────────
class GalleryManager:
    """
    Guarda y carga embeddings medios por sujeto en un archivo .npy.
    Permite actualizar la galería sin reentrenar.
    """
    def __init__(self, path: str = "gallery_embeddings.npy"):
        self.path = path
        self.gallery: dict[int, np.ndarray] = {}  # {label: emb (EMBED_DIM,)}

    def update(self, label: int, emb: np.ndarray):
        if label in self.gallery:
            self.gallery[label] = (self.gallery[label] + emb) / 2.0
        else:
            self.gallery[label] = emb.copy()

    def save(self):
        np.save(self.path, self.gallery)
        print(f"[Gallery] Guardada → {self.path}  ({len(self.gallery)} sujetos)")

    def load(self):
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Galería no encontrada: {self.path}")
        self.gallery = np.load(self.path, allow_pickle=True).item()
        print(f"[Gallery] Cargada → {self.path}  ({len(self.gallery)} sujetos)")
        return self.gallery

    def match(self, emb: np.ndarray, threshold: float) -> tuple[int, float]:
        """1-NN. Devuelve (label, distancia). label=-1 si desconocido."""
        best_lbl, best_dist = -1, float("inf")
        for lbl, ref in self.gallery.items():
            d = float(np.linalg.norm(emb - ref))
            if d < best_dist:
                best_dist, best_lbl = d, lbl
        if best_dist > threshold:
            return -1, best_dist
        return best_lbl, best_dist

    def calibrate_threshold(self, percentile: float = 95.0) -> float:
        """
        Calcula automáticamente el umbral de rechazo como el percentil P
        de las distancias inter-clase dentro de la galería.
        """
        labels = list(self.gallery.keys())
        dists  = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                d = np.linalg.norm(self.gallery[labels[i]] - self.gallery[labels[j]])
                dists.append(d)
        if not dists:
            return 1.0
        t = float(np.percentile(dists, percentile))
        print(f"[Gallery] Umbral calibrado (p{percentile:.0f}): {t:.4f}")
        return t


# ── Data Augmentation ─────────────────────────────────────────────────────────
class SkeletonAugment:
    def __init__(self, noise_std=0.02, scale_range=(0.85, 1.15), flip_prob=0.5, time_shift=5):
        self.noise_std   = noise_std
        self.scale_range = scale_range
        self.flip_prob   = flip_prob
        self.time_shift  = time_shift

    def __call__(self, kp: np.ndarray) -> np.ndarray:
        kp  = kp.copy()
        kp += np.random.normal(0, self.noise_std, kp.shape).astype(np.float32)
        kp *= np.random.uniform(*self.scale_range)
        if np.random.random() < self.flip_prob:
            kp[:, 0::2] *= -1.0
        kp  = np.roll(kp, np.random.randint(-self.time_shift, self.time_shift + 1), axis=0)
        # Velocidad aleatoria: subsampleo + interpolación
        speed = np.random.uniform(0.85, 1.15)
        T     = kp.shape[0]
        idx   = np.clip(np.round(np.arange(T) * speed).astype(int), 0, T - 1)
        kp    = kp[idx]
        return kp


class GEIAugment:
    def __init__(self, rot_deg=5, shift_px=5, noise_std=0.02, brightness_range=(-0.1, 0.1)):
        self.rot_deg          = rot_deg
        self.shift_px         = shift_px
        self.noise_std        = noise_std
        self.brightness_range = brightness_range

    def __call__(self, gei: np.ndarray) -> np.ndarray:
        import cv2
        h, w   = gei.shape
        center = (w // 2, h // 2)

        angle  = np.random.uniform(-self.rot_deg, self.rot_deg)
        M      = cv2.getRotationMatrix2D(center, angle, 1.0)
        gei    = cv2.warpAffine(gei, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        tx, ty = np.random.randint(-self.shift_px, self.shift_px+1, size=2)
        M      = np.float32([[1, 0, tx], [0, 1, ty]])
        gei    = cv2.warpAffine(gei, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        gei   += np.random.uniform(*self.brightness_range)
        gei   += np.random.normal(0, self.noise_std, gei.shape).astype(np.float32)
        return np.clip(gei, 0.0, 1.0).astype(np.float32)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HybridGaitNet().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Device      : {device}")
    print(f"Parámetros  : {params:,}")
    kp  = torch.randn(4, SEQ_LEN, KP_DIM_MP).to(device)
    gei = torch.randn(4, 1, GEI_H, GEI_W).to(device)
    emb = model(kp, gei)
    print(f"Embedding   : {emb.shape}   norma≈{emb.norm(dim=1).mean().item():.4f}")
    gm  = GalleryManager()
    gm.update(0, emb[0].detach().cpu().numpy())
    gm.update(1, emb[1].detach().cpu().numpy())
    t   = gm.calibrate_threshold(95)
    print(f"Threshold   : {t:.4f}")
