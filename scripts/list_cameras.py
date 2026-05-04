"""Enumera qué índices de cámara abren con qué backend de OpenCV.

Útil para cuando `infer_live.py` no detecta la C920 en Windows. Prueba los
backends MSMF (default Windows 10/11), DSHOW (DirectShow legacy, suele andar
mejor con USB UVC) y ANY. Para cada índice 0..N-1 reporta:
    - si abre,
    - resolución/FPS reportada por el driver,
    - si lee al menos 1 frame real,
    - el nombre del backend.

Uso:
    python scripts/list_cameras.py
    python scripts/list_cameras.py --max-index 6 --warm-frames 5
"""

from __future__ import annotations

import argparse
import time

import cv2

BACKENDS = [
    ("MSMF",  cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW),
    ("ANY",   cv2.CAP_ANY),
]


def probe(idx: int, backend_name: str, backend_id: int,
          warm_frames: int = 3, timeout_s: float = 4.0) -> dict:
    info = {"idx": idx, "backend": backend_name, "opened": False,
            "got_frame": False, "w": 0, "h": 0, "fps": 0.0,
            "elapsed_s": 0.0, "error": ""}
    t0 = time.time()
    try:
        cap = cv2.VideoCapture(idx, backend_id)
        if not cap.isOpened():
            info["error"] = "isOpened()=False"
            return info
        info["opened"] = True
        info["w"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["h"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        for _ in range(warm_frames):
            if time.time() - t0 > timeout_s:
                info["error"] = f"timeout {timeout_s}s leyendo frames"
                break
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                info["got_frame"] = True
                info["w"] = frame.shape[1]
                info["h"] = frame.shape[0]
                break
        cap.release()
    except Exception as e:
        info["error"] = f"exc: {e}"
    info["elapsed_s"] = time.time() - t0
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=4,
                    help="Probar índices 0..max_index-1 (default 4).")
    ap.add_argument("--warm-frames", type=int, default=3,
                    help="Frames a leer para confirmar que la cámara entrega imagen.")
    args = ap.parse_args()

    print(f"OpenCV {cv2.__version__}")
    print(f"Probando índices 0..{args.max_index - 1} con backends "
          f"{[b[0] for b in BACKENDS]}")
    print("-" * 78)
    print(f"{'idx':<4}{'backend':<8}{'open':<6}{'frame':<7}{'WxH':<14}"
          f"{'fps':<7}{'t(s)':<7}{'note'}")
    print("-" * 78)

    working = []
    for idx in range(args.max_index):
        for name, bid in BACKENDS:
            r = probe(idx, name, bid, warm_frames=args.warm_frames)
            wxh = f"{r['w']}x{r['h']}" if r["w"] else "—"
            print(f"{idx:<4}{name:<8}{str(r['opened']):<6}"
                  f"{str(r['got_frame']):<7}{wxh:<14}"
                  f"{r['fps']:<7.1f}{r['elapsed_s']:<7.2f}{r['error']}")
            if r["got_frame"]:
                working.append((idx, name, bid, r["w"], r["h"], r["fps"]))

    print("-" * 78)
    if not working:
        print("\n❌ NINGUNA combinación funcionó.")
        print("Causas típicas:")
        print("  1. Otra app está usando la cámara (Teams/Zoom/OBS/navegador).")
        print("  2. Permisos de cámara desactivados en Windows:")
        print("     Configuración → Privacidad y seguridad → Cámara → permitir apps.")
        print("  3. Driver dañado: desconecta C920, espera 5s, reconecta en otro USB.")
        print("  4. Cable USB dañado o puerto USB 2.0 sin ancho de banda suficiente.")
        return 1

    print(f"\n✅ Combinaciones funcionales: {len(working)}")
    for idx, name, _bid, w, h, fps in working:
        print(f"   idx={idx}  backend={name}  {w}x{h} @ {fps:.1f} fps")
    print("\nUsa el primer índice/backend de la lista. Para forzar el backend en "
          "infer_live.py corre:")
    idx, name, *_ = working[0]
    flag = "" if name == "MSMF" else f" --backend {name.lower()}"
    print(f"   python scripts/infer_live.py --camera {idx}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
