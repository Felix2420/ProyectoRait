"""ST-GCN ligero para gait recognition sobre keypoints COCO-17.

Entrada:  (B, C_in=3, T, V=17)  con (x, y, conf) normalizados.
Salida:   embedding (B, emb_dim) y logits CE (B, n_classes).

~0.3M params. Sin pretrained — entrenado from scratch sobre 13 sujetos.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# COCO-17 edges (pares simétricos). Usados para construir la adyacencia.
COCO17_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Pares L↔R para flip horizontal (índices que se intercambian).
COCO17_FLIP_PAIRS: List[Tuple[int, int]] = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
]


def build_adjacency(num_nodes: int = 17) -> torch.Tensor:
    """Adyacencia normalizada simétrica con auto-bucles: D^-1/2 (A+I) D^-1/2."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in COCO17_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += np.eye(num_nodes, dtype=np.float32)
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-6))
    D = np.diag(d_inv_sqrt)
    A_norm = D @ A @ D
    return torch.from_numpy(A_norm)


class STGCNBlock(nn.Module):
    """Bloque ST-GCN: GCN espacial 1x1 + conv temporal 1D (kernel=9)."""

    def __init__(self, in_ch: int, out_ch: int, t_kernel: int = 9, stride_t: int = 1, dropout: float = 0.1):
        super().__init__()
        # GCN espacial: 1x1 conv tras multiplicar X·A.
        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        pad = (t_kernel - 1) // 2
        self.tcn = nn.Conv2d(
            out_ch, out_ch,
            kernel_size=(t_kernel, 1),
            stride=(stride_t, 1),
            padding=(pad, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout)

        if in_ch != out_ch or stride_t != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride_t, 1), bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        res = self.residual(x)
        # GCN espacial: agregación con A (V,V) sobre la dim de nodos.
        h = torch.einsum("bctv,vw->bctw", x, A)
        h = self.gcn(h)
        h = F.relu(self.bn1(h), inplace=True)
        h = self.tcn(h)
        h = self.bn2(h)
        h = self.dropout(h)
        return F.relu(h + res, inplace=True)


class BNNeck(nn.Module):
    """BN + FC para clasificación; embedding pre-BN para matching."""

    def __init__(self, emb_dim: int, n_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(emb_dim)
        self.bn.bias.requires_grad_(False)
        self.fc = nn.Linear(emb_dim, n_classes, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_bn = self.bn(x)
        logits = self.fc(x_bn)
        return x, logits


class STGCNLite(nn.Module):
    """ST-GCN ligero: 4 bloques, ~0.3M params, embedding 128-D."""

    def __init__(self, in_channels: int = 3, num_nodes: int = 17,
                 emb_dim: int = 128, n_classes: int = 13,
                 channels: Tuple[int, ...] = (32, 64, 64, 128),
                 t_kernel: int = 9, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("A", build_adjacency(num_nodes))
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        ch_in = in_channels
        blocks = []
        for ch_out in channels:
            blocks.append(STGCNBlock(ch_in, ch_out, t_kernel=t_kernel, dropout=dropout))
            ch_in = ch_out
        self.blocks = nn.ModuleList(blocks)
        self.emb_dim = emb_dim
        self.proj = nn.Linear(channels[-1], emb_dim) if channels[-1] != emb_dim else nn.Identity()
        self.bnneck = BNNeck(emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        # data BN sobre canales × nodos (estabiliza escala global).
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.data_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()

        for blk in self.blocks:
            x = blk(x, self.A)
        # Pool temporal (max) y espacial (mean) — coherente con OpenGait pipeline.
        x = x.max(dim=2)[0]      # (B, C, V)
        x = x.mean(dim=2)        # (B, C)
        emb = self.proj(x)
        emb_bnn, logits = self.bnneck(emb)
        return emb_bnn, logits

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        emb, _ = self.forward(x)
        return emb
