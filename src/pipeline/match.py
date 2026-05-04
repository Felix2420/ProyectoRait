"""Matcher multi-template max-sim contra la gallery.

Gallery: (n_subjects, K_max, 256, 16) + valid_mask (n_subjects, K_max).
Para un probe `e` (4096,), similitud por sujeto = max sobre sus templates
válidos. Top-1 = argmax. Si sim_top1 < τ → unknown.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .types import MatchResult

log = logging.getLogger("pipeline.match")


class GalleryMatcher:
    def __init__(self, gallery_npy: Path, valid_mask_npy: Path,
                 index_json: Path, tau: float) -> None:
        self.tau = float(tau)
        gallery = np.load(gallery_npy).astype(np.float32)  # (N, K, 256, 16)
        mask = np.load(valid_mask_npy).astype(bool)        # (N, K)
        meta = json.loads(Path(index_json).read_text(encoding="utf-8"))
        subjects: List[str] = meta["subjects"]
        if gallery.shape[0] != len(subjects) or mask.shape != gallery.shape[:2]:
            raise ValueError("Gallery / mask / index inconsistentes")
        n, k = gallery.shape[:2]
        # Aplanar a (N, K, D) y poner ceros donde no hay template (mask=False)
        flat = gallery.reshape(n, k, -1)  # (N, K, D)
        flat = flat * mask[:, :, None]
        self._flat = flat                  # (N, K, D)
        self._mask = mask                  # (N, K)
        self.subjects = subjects
        self.n = n
        self.k = k
        log.info("gallery cargada: N=%d K_max=%d D=%d τ=%.4f",
                 n, k, flat.shape[-1], self.tau)

    def match(self, emb: np.ndarray) -> MatchResult:
        """emb: (D,) L2-normalizado. Devuelve MatchResult con top-5."""
        # sims contra cada template: (N, K)
        sims = (self._flat @ emb)
        # Inválidos a -inf
        sims = np.where(self._mask, sims, -np.inf)
        # max por sujeto
        per_subj = sims.max(axis=1)         # (N,)
        order = np.argsort(-per_subj)
        top5: List[Tuple[str, float]] = []
        for i in order[:5]:
            top5.append((self.subjects[int(i)], float(per_subj[int(i)])))
        top1_idx = int(order[0])
        sim_top1 = float(per_subj[top1_idx])
        unknown = sim_top1 < self.tau
        return MatchResult(
            subject=self.subjects[top1_idx] if not unknown else None,
            sim=sim_top1,
            unknown=unknown,
            top5=top5,
        )
