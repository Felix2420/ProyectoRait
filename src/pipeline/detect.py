"""Detección de personas con YOLOv11n."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

from .types import BBox

log = logging.getLogger("pipeline.detect")


class PersonDetector:
    def __init__(self, weights: Path, conf: float = 0.35,
                 iou: float = 0.5, imgsz: int = 640, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        log.info("YOLO weights=%s device=%s imgsz=%d", weights, self.device, imgsz)
        self.model = YOLO(str(weights))

    def detect(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        """Devuelve la persona de mayor área·confianza, o None."""
        res = self.model.predict(
            frame_bgr, classes=[0], verbose=False,
            device=self.device, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz,
        )
        if not res or len(res[0].boxes) == 0:
            return None
        boxes = res[0].boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = res[0].boxes.conf.cpu().numpy()  # (N,)
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best = int(np.argmax(areas * confs))
        b = boxes[best]
        return BBox(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(confs[best]))
