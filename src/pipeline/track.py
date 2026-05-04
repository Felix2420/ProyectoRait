"""FSM simple de tracking para una persona a la vez.

Estados:
    IDLE     — sin persona en cuadro.
    WARMING  — persona aparece; acumulando frames consecutivos con bbox válido.
    ACTIVE   — persona estable (>= warmup_frames consecutivos); el seq_buffer
               recibe siluetas.
    RESET    — emitido por una sola iteración cuando ACTIVE pierde a la persona
               por más de reset_frames consecutivos. El consumidor debe limpiar
               el buffer y vuelve a IDLE en el siguiente step.

Continuidad de bbox: se requiere IOU >= iou_min con el bbox del frame previo
(cuando ambos existen). Si IOU es bajo se considera "ruptura" y se trata como
sin bbox.
"""

from __future__ import annotations

import logging
from typing import Optional

from .types import BBox, iou as bbox_iou

log = logging.getLogger("pipeline.track")


class TrackerFSM:
    STATES = ("IDLE", "WARMING", "ACTIVE", "RESET")

    def __init__(self, warmup_frames: int = 15, reset_frames: int = 15,
                 iou_min: float = 0.3) -> None:
        self.warmup_frames = warmup_frames
        self.reset_frames = reset_frames
        self.iou_min = iou_min
        self.state = "IDLE"
        self._consec_with = 0
        self._consec_without = 0
        self._last_bbox: Optional[BBox] = None

    def step(self, bbox: Optional[BBox]) -> str:
        # Evaluar continuidad
        if bbox is not None and self._last_bbox is not None:
            if bbox_iou(bbox, self._last_bbox) < self.iou_min:
                # Ruptura: tratar como sin bbox para conservadurismo
                bbox = None

        if self.state == "RESET":
            # ya emitido un step antes; limpiar y caer a IDLE
            self.state = "IDLE"
            self._consec_with = 0
            self._consec_without = 0
            self._last_bbox = None

        if bbox is not None:
            self._consec_with += 1
            self._consec_without = 0
            self._last_bbox = bbox
            if self.state == "IDLE":
                self.state = "WARMING"
            elif self.state == "WARMING" and self._consec_with >= self.warmup_frames:
                self.state = "ACTIVE"
        else:
            self._consec_without += 1
            self._consec_with = 0
            if self.state in ("WARMING", "ACTIVE") and self._consec_without >= self.reset_frames:
                prev = self.state
                self.state = "RESET"
                log.info("tracker: %s -> RESET (no bbox %d frames)", prev, self._consec_without)

        return self.state

    @property
    def last_bbox(self) -> Optional[BBox]:
        return self._last_bbox
