"""Segmentación con RVM mobilenetv3 + crop/normalize a 64×44.

Reusa la misma lógica de `src/preprocess/extract_sequence.py` para mantener
paridad pixel-a-pixel con el preproceso de entrenamiento. El estado recurrente
de RVM se mantiene entre frames y se resetea cuando el FSM emite RESET.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch

from src.preprocess.extract_sequence import crop_and_normalize_silhouette
from .types import BBox

log = logging.getLogger("pipeline.segment")

SIL_H, SIL_W = 64, 44


class Segmenter:
    def __init__(self, downsample_ratio: float = 0.25, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.downsample_ratio = downsample_ratio
        log.info("RVM mobilenetv3 device=%s downsample=%.3f", self.device, downsample_ratio)
        self.rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True)
        self.rvm = self.rvm.to(self.device).eval()
        self._rec = [None, None, None, None]  # estado recurrente

    def reset_state(self) -> None:
        self._rec = [None, None, None, None]

    @torch.inference_mode()
    def segment(self, frame_bgr: np.ndarray, bbox: Optional[BBox]) -> Optional[np.ndarray]:
        """Devuelve silueta uint8 (64,44) ∈ {0,255} o None si no se puede."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = (torch.from_numpy(rgb).to(self.device).float()
             .permute(2, 0, 1).unsqueeze(0) / 255.0)
        fgr, pha, *self._rec = self.rvm(t, *self._rec, downsample_ratio=self.downsample_ratio)
        alpha = pha[0, 0].clamp(0, 1).cpu().numpy()
        if bbox is None:
            return None
        sil = crop_and_normalize_silhouette(
            alpha, np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=np.float32)
        )
        if sil is None or sil.shape != (SIL_H, SIL_W):
            return None
        if (sil > 0).sum() < int(0.05 * SIL_H * SIL_W):
            # silueta vacía / muy pobre
            return None
        return sil
