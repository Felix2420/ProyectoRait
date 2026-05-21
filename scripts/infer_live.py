"""Pipeline en vivo (Fase 6.4): cámara C920 → log de asistencia + UI.

Igual que `scripts/infer_video.py` pero con:
  - fuente = cámara (índice configurable),
  - UI OpenCV con overlay (bbox + estado FSM + top-1 + sim + buffer + FPS),
  - confirmaciones múltiples por corrida (tras RESET se permite la siguiente
    persona),
  - anti-duplicado por jornada: si alguien ya fue marcado en esta corrida no se
    vuelve a escribir su fila en el CSV (la UI sí lo muestra como "ya marcado").

Uso:
    python scripts/infer_live.py
    python scripts/infer_live.py --camera 1 --config configs/pipeline.yaml
    python scripts/infer_live.py --no-show          # corrida headless (debug)

Hotkeys (sobre la ventana de OpenCV):
    q   salir
    r   forzar reset del tracker / buffer / acumulador
    f   alternar pantalla completa

Salida:
    logs/attendance_YYYY-MM-DD.csv   (append, una fila por sujeto/día)
    stdout: trazas de eventos.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.pipeline.capture import Capture                  # noqa: E402
from src.pipeline.detect import PersonDetector            # noqa: E402
from src.pipeline.track import TrackerFSM                 # noqa: E402
from src.pipeline.segment import Segmenter                # noqa: E402
from src.pipeline.seq_buffer import SequenceBuffer        # noqa: E402
from src.pipeline.embed import GaitEmbedder               # noqa: E402
from src.pipeline.match import GalleryMatcher             # noqa: E402
from src.pipeline.confirm import ConfirmationAccumulator  # noqa: E402
from src.pipeline.log import AttendanceLogger             # noqa: E402
from src.pipeline.types import BBox, MatchResult          # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("infer_live")

WIN = "Pase de lista — gait (q=salir, r=reset, f=fullscreen)"

# Colores BGR por estado del FSM
STATE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "IDLE":    (160, 160, 160),
    "WARMING": (0, 200, 255),     # amarillo
    "ACTIVE":  (0, 220, 0),       # verde
    "RESET":   (0, 0, 220),       # rojo
}
BANNER_TTL_S = 2.0                 # tiempo visible del banner de confirmación


# ---------------------------------------------------------------------- utils

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO / pp)


# ----------------------------------------------------------------------- UI

def draw_overlay(frame: np.ndarray,
                 *,
                 state: str,
                 bbox: Optional[BBox],
                 buf_count: int,
                 buf_window: int,
                 last_match: Optional[MatchResult],
                 tau: float,
                 fps_eff: float,
                 confirmed_today: Set[str],
                 banner: Optional[Tuple[str, float, bool]]) -> None:
    """Dibuja el overlay in-place sobre `frame` (BGR)."""
    h, w = frame.shape[:2]
    color = STATE_COLORS.get(state, (200, 200, 200))

    # bbox
    if bbox is not None:
        cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)

    # barra superior con info de estado
    bar_h = 64
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), thickness=cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"estado: {state}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"buffer: {buf_count}/{buf_window}",
                (200, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240),
                1, cv2.LINE_AA)
    cv2.putText(frame, f"fps: {fps_eff:5.1f}", (380, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, f"presentes hoy: {len(confirmed_today)}", (520, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    # info de último match
    if last_match is not None:
        if last_match.unknown:
            txt = f"top1: {last_match.top5[0][0]:<20}  sim={last_match.sim:.4f} < tau={tau:.4f}  -> UNK"
            txt_col = (140, 140, 240)
        else:
            txt = f"top1: {last_match.subject:<20}  sim={last_match.sim:.4f}  >= tau={tau:.4f}"
            txt_col = (180, 255, 180)
        cv2.putText(frame, txt, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    txt_col, 1, cv2.LINE_AA)

    # banner de confirmación
    if banner is not None:
        subj, t_until, is_dup = banner
        if time.time() < t_until:
            band_h = 70
            y0 = h - band_h
            ov2 = frame.copy()
            band_col = (0, 80, 0) if not is_dup else (40, 40, 80)
            cv2.rectangle(ov2, (0, y0), (w, h), band_col, thickness=cv2.FILLED)
            cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)
            label = (f"PRESENTE: {subj}" if not is_dup
                     else f"{subj} (ya marcado hoy)")
            cv2.putText(frame, label, (16, y0 + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2,
                        cv2.LINE_AA)


# ----------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None,
                    help="Índice de cámara (default: el de configs/pipeline.yaml)")
    ap.add_argument("--backend", choices=["msmf", "dshow", "any"], default=None,
                    help="Backend OpenCV para la cámara (Windows: prueba dshow "
                         "si msmf falla). Si no se pasa, usa el default de OpenCV.")
    ap.add_argument("--config", default=str(REPO / "configs" / "pipeline.yaml"))
    ap.add_argument("--no-show", action="store_true",
                    help="No abre ventana de OpenCV (modo headless).")
    ap.add_argument("--quiet", action="store_true",
                    help="Solo loguea WARNING+ (silencia traza por evento).")
    args = ap.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    cfg = load_config(Path(args.config))
    cam_source = args.camera if args.camera is not None else cfg["camera"]["source"]

    # --- Inicializar módulos ---
    cap = Capture(cam_source, target_fps=cfg["camera"]["target_fps"],
                  backend=args.backend)
    det = PersonDetector(weights=resolve(cfg["detector"]["weights"]),
                         conf=cfg["detector"]["conf"],
                         iou=cfg["detector"]["iou"],
                         imgsz=cfg["detector"].get("imgsz", 640))
    fsm = TrackerFSM(warmup_frames=cfg["tracker"]["warmup_frames"],
                     reset_frames=cfg["tracker"]["reset_frames"],
                     iou_min=cfg["tracker"]["iou_min"])
    seg = Segmenter(downsample_ratio=cfg["segmenter"]["downsample_ratio"])
    buf = SequenceBuffer(window=cfg["sequence"]["window"],
                         stride=cfg["sequence"]["stride"])
    emb = GaitEmbedder(ckpt=resolve(cfg["embedder"]["ckpt"]),
                       class_num=cfg["embedder"]["class_num"],
                       compile=cfg["embedder"].get("compile", False))
    mch = GalleryMatcher(gallery_npy=resolve(cfg["matcher"]["gallery_npy"]),
                         valid_mask_npy=resolve(cfg["matcher"]["valid_mask_npy"]),
                         index_json=resolve(cfg["matcher"]["index_json"]),
                         tau=cfg["matcher"]["tau"])
    conf = ConfirmationAccumulator(n_consecutive=cfg["confirmation"]["n_consecutive"])
    logger = AttendanceLogger(csv_dir=resolve(cfg["logging"]["csv_dir"]))

    show = not args.no_show
    fullscreen = False
    if show:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    session_id = AttendanceLogger.new_session_id()
    confirmed_today: Set[str] = set()
    last_match: Optional[MatchResult] = None
    banner: Optional[Tuple[str, float, bool]] = None  # (subject, t_until, is_dup)

    # FPS efectivo (ventana móvil de los últimos N timestamps)
    t_window: deque[float] = deque(maxlen=30)
    n_frames = 0
    n_seqs = 0
    t0 = time.time()

    log.info("=== en vivo: cámara=%s τ=%.4f n_consec=%d ===",
             cam_source, cfg["matcher"]["tau"], cfg["confirmation"]["n_consecutive"])
    log.info("hotkeys: q=salir, r=reset manual, f=fullscreen")

    try:
        while True:
            item = cap.read()
            if item is None:
                log.warning("cámara devolvió None (¿desconexión?). Saliendo.")
                break
            frame, _ts = item
            n_frames += 1

            bbox = det.detect(frame)
            state = fsm.step(bbox)

            if state == "RESET":
                log.info("[%05d] RESET — limpio buffer / acumulador", n_frames)
                buf.reset()
                seg.reset_state()
                conf.reset()
                session_id = AttendanceLogger.new_session_id()
                last_match = None
                banner = None

            elif state != "ACTIVE":
                # IDLE / WARMING: avanzo RVM para preservar estado recurrente,
                # pero no acumulo silueta al buffer.
                _ = seg.segment(frame, fsm.last_bbox)

            else:
                # ACTIVE → silueta + buffer
                sil = seg.segment(frame, fsm.last_bbox)
                if sil is not None:
                    seq = buf.push(sil)
                    if seq is not None:
                        n_seqs += 1
                        e = emb.embed(seq)
                        m = mch.match(e)
                        last_match = m
                        verdict = "UNK" if m.unknown else m.subject
                        log.info("[%05d] seq#%d top1=%s sim=%.4f%s",
                                 n_frames, n_seqs, verdict, m.sim,
                                 " (UNK)" if m.unknown else "")
                        confirmed_now = conf.push(m)
                        if confirmed_now is not None:
                            is_dup = confirmed_now in confirmed_today
                            if not is_dup:
                                logger.log_confirmation(
                                    subject=confirmed_now,
                                    mean_sim=conf.mean_sim(),
                                    n_sequences=cfg["confirmation"]["n_consecutive"],
                                    frames_used=cfg["sequence"]["window"]
                                    + (cfg["confirmation"]["n_consecutive"] - 1)
                                    * cfg["sequence"]["stride"],
                                    session_id=session_id,
                                )
                                confirmed_today.add(confirmed_now)
                            else:
                                log.info("ya marcado hoy: %s (no re-loguea)",
                                         confirmed_now)
                            banner = (confirmed_now,
                                      time.time() + BANNER_TTL_S,
                                      is_dup)

            # FPS efectivo
            t_window.append(time.time())
            fps_eff = 0.0
            if len(t_window) >= 2:
                fps_eff = (len(t_window) - 1) / max(1e-6,
                                                    t_window[-1] - t_window[0])

            if show:
                draw_overlay(frame,
                             state=state,
                             bbox=fsm.last_bbox if state in ("WARMING",
                                                              "ACTIVE") else bbox,
                             buf_count=len(buf),
                             buf_window=cfg["sequence"]["window"],
                             last_match=last_match,
                             tau=cfg["matcher"]["tau"],
                             fps_eff=fps_eff,
                             confirmed_today=confirmed_today,
                             banner=banner)
                cv2.imshow(WIN, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    log.info("salida solicitada por usuario (q)")
                    break
                elif key == ord("r"):
                    log.info("reset manual solicitado (r)")
                    buf.reset()
                    seg.reset_state()
                    conf.reset()
                    fsm = TrackerFSM(
                        warmup_frames=cfg["tracker"]["warmup_frames"],
                        reset_frames=cfg["tracker"]["reset_frames"],
                        iou_min=cfg["tracker"]["iou_min"],
                    )
                    session_id = AttendanceLogger.new_session_id()
                    last_match = None
                    banner = None
                elif key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        WIN,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )

            # Limpieza de banner ya expirado para no llevar tuplas viejas
            if banner is not None and time.time() >= banner[1]:
                banner = None

    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    dt = time.time() - t0
    log.info("=== fin ===")
    log.info("frames=%d secs=%d tiempo=%.1fs (%.1f fps efectivo) presentes=%d",
             n_frames, n_seqs, dt, n_frames / dt if dt > 0 else 0.0,
             len(confirmed_today))
    if confirmed_today:
        log.info("hoy confirmados: %s", sorted(confirmed_today))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
