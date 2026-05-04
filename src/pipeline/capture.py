"""Captura de frames con decimación a target_fps.

Soporta video file (str) o cámara (int). No redimensiona: el resto del
pipeline opera sobre la resolución nativa para mantener paridad con el
preproceso de entrenamiento (`src/preprocess/extract_sequence.py`).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("pipeline.capture")

BACKENDS = {
    "msmf":  cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "any":   cv2.CAP_ANY,
}


class Capture:
    def __init__(self, source: int | str, target_fps: int = 15,
                 backend: str | None = None,
                 cam_width: int | None = 1280,
                 cam_height: int | None = 720,
                 cam_fps: int | None = 30,
                 buffer_size: int = 1) -> None:
        """source: int (índice de cámara) o str (ruta de video).
        backend: "msmf" | "dshow" | "any" | None (default OpenCV). Solo aplica
        a fuentes int; para archivos de video se ignora.
        cam_width/height/fps: solo aplican a cámara (int). 720p@30 es el sweet
        spot para C920 + GTX 1650 (1080p satura RVM, 480p degrada silueta).
        buffer_size: solo cámara. 1 = OpenCV descarta frames viejos en vez de
        encolarlos → mata el lag acumulado a costa de "tirar" frames cuando el
        pipeline va más lento que la cámara (es el comportamiento deseado en
        vivo).
        """
        self.source = source
        self.target_fps = target_fps
        is_cam = isinstance(source, int)
        if is_cam and backend is not None:
            key = backend.lower()
            if key not in BACKENDS:
                raise ValueError(f"backend desconocido: {backend} "
                                 f"(usa: {list(BACKENDS.keys())})")
            self.cap = cv2.VideoCapture(source, BACKENDS[key])
            log.info("backend forzado: %s", key)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir source={source} "
                               f"(backend={backend})")
        if is_cam:
            if cam_width is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_width))
            if cam_height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_height))
            if cam_fps is not None:
                self.cap.set(cv2.CAP_PROP_FPS, int(cam_fps))
            # Anti-lag: que el driver no encole frames atrasados.
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))
            except Exception:
                pass
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info("cámara configurada: %dx%d (pedí %sx%s)",
                     actual_w, actual_h, cam_width, cam_height)
        self.src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        # Cada cuántos frames de origen tomamos uno
        self.skip = max(1, int(round(self.src_fps / target_fps)))
        self._idx_src = 0
        self._idx_emit = 0
        log.info("source=%s src_fps=%.1f skip=%d → target_fps≈%.1f",
                 source, self.src_fps, self.skip, self.src_fps / self.skip)

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        """Devuelve (frame_bgr, ts_seg) o None si EOF/error.

        En modo cámara hace `grab()` + `retrieve()` para que el último frame
        decodificado siempre sea el más reciente del driver, no uno encolado.
        """
        is_cam = isinstance(self.source, int)
        while True:
            if is_cam:
                # Vaciar lo que haya en el buffer del driver y quedarnos con
                # el último frame: skip-1 grabs descartados + 1 retrieve.
                for _ in range(self.skip - 1):
                    if not self.cap.grab():
                        return None
                ok, frame = self.cap.read()
                if not ok:
                    return None
                self._idx_src += self.skip
                ts = self._idx_emit / float(self.target_fps)
                self._idx_emit += 1
                return frame, ts
            # Archivo de video: comportamiento original (decimación por skip).
            ok, frame = self.cap.read()
            if not ok:
                return None
            self._idx_src += 1
            if (self._idx_src - 1) % self.skip != 0:
                continue
            ts = self._idx_emit / float(self.target_fps)
            self._idx_emit += 1
            return frame, ts

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
