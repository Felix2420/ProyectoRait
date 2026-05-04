# Fase 6.1 — Diseño del pipeline de inferencia en vivo

> Documento de diseño previo a código. Define contratos, módulos, formatos y
> políticas. Una vez aprobado, sirve de referencia para 6.2–6.5. **No se
> escribe código hasta que el usuario apruebe este documento.**

Fecha: 2026-04-28
Modelo de producción: `checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt`
Umbral: τ=0.9807 (Fase 5)

---

## 1. Diagrama de bloques

```
                     ┌────────────────────────────────────────────────┐
                     │                  CÁMARA C920                   │
                     │              30 fps, 1280×720 BGR              │
                     └───────────────────────┬────────────────────────┘
                                             │ frame_t
                                             ▼
                     ┌────────────────────────────────────────────────┐
   1. capture        │  Grab frame, resize, decimate 30→15 fps        │
                     │  Salida: frame BGR 640×360 + ts                 │
                     └───────────────────────┬────────────────────────┘
                                             ▼
                     ┌────────────────────────────────────────────────┐
   2. detect         │  YOLO (yolo11n.pt) clase=person, conf>0.5      │
                     │  Salida: bbox de mayor área o None              │
                     └───────────────────────┬────────────────────────┘
                                             ▼
                     ┌────────────────────────────────────────────────┐
   3. track          │  Validar continuidad: IOU>0.3 con bbox prev    │
                     │  Estado: IDLE / WARMING / ACTIVE / RESET        │
                     └───────────────────────┬────────────────────────┘
                                             ▼
                     ┌────────────────────────────────────────────────┐
   4. segment        │  Crop bbox + RVM → silueta binaria 64×44       │
                     │  Salida: tensor uint8 (64,44)                   │
                     └───────────────────────┬────────────────────────┘
                                             ▼
                     ┌────────────────────────────────────────────────┐
   5. seq_buffer     │  FIFO 60 frames; emite secuencia cada 30        │
                     │  Salida: tensor (60,64,44) cuando lista         │
                     └───────────────────────┬────────────────────────┘
                                             ▼
                     ┌────────────────────────────────────────────────┐
   6. embed          │  GaitBase iter1200 → embedding (256,16)        │
                     │  L2-norm sobre dim feat                         │
                     └───────────────────────┬────────────────────────┘
                                             ▼
                     ┌────────────────────────────────────────────────┐
   7. match          │  cosine vs gallery (19 sujetos)                │
                     │  Top-1: si sim>τ → candidato; si no → unknown   │
                     └───────────────────────┬────────────────────────┘
                                             ▼
                     ┌────────────────────────────────────────────────┐
   8. confirm        │  Acumulador: N=2 secuencias consecutivas con   │
                     │  mismo top-1 y sim>τ → CONFIRMADO              │
                     └───────────────────────┬────────────────────────┘
                                             ▼
                     ┌────────────────────────────────────────────────┐
   9. log + UI       │  CSV append + overlay en ventana OpenCV        │
                     └────────────────────────────────────────────────┘
```

## 2. Contratos por módulo

Cada módulo es un archivo en `src/pipeline/` con una clase y método principal.
Type hints obligatorios. Sin lógica fuera de su responsabilidad.

### 2.1 `capture.py`
```python
class Capture:
    def __init__(self, source: int | str, target_fps: int = 15) -> None: ...
    def read(self) -> tuple[np.ndarray, float] | None:
        """Devuelve (frame_bgr, timestamp) o None si EOF/error."""
```
- `source=0` para C920, `str` para .mp4 en modo offline.
- Decimación interna: si cámara da 30 fps, devuelve 1 de cada 2 frames.
- `target_fps=15` fijo en producción.

### 2.2 `detect.py`
```python
class PersonDetector:
    def __init__(self, weights: Path, conf: float = 0.5, device: str = "cuda") -> None: ...
    def detect(self, frame: np.ndarray) -> BBox | None:
        """Detecta persona de mayor área. None si no hay."""
```
- Modelo: `yolo11n.pt` (ya en repo raíz). class=0 (person).
- Si hay >1 persona: tomar bbox con mayor `(x2-x1)*(y2-y1)`. Loguear warning.
- `BBox = NamedTuple(x1,y1,x2,y2,conf)` — coords en píxeles enteros.

### 2.3 `track.py`
```python
class TrackerFSM:
    """FSM simple para una persona a la vez."""
    state: Literal["IDLE", "WARMING", "ACTIVE", "RESET"]
    def step(self, bbox: BBox | None) -> Literal["IDLE","WARMING","ACTIVE","RESET"]: ...
```
- Transiciones:
  - `IDLE → WARMING`: bbox aparece.
  - `WARMING → ACTIVE`: 15 frames consecutivos con bbox válido (IOU>0.3 con anterior).
  - `ACTIVE → RESET`: bbox ausente >15 frames consecutivos.
  - `RESET → IDLE`: limpia buffer y vuelve.
- Solo en `ACTIVE` se acumula al `seq_buffer`.

### 2.4 `segment.py`
```python
class Segmenter:
    def __init__(self, model: str = "rvm_mobilenetv3", device: str = "cuda") -> None: ...
    def segment(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Devuelve silueta uint8 (H=64, W=44), valores {0,255}."""
```
- Crop por bbox con padding 10 % → RVM → máscara α → threshold 0.5 → resize 64×44 manteniendo aspect (pad 0).
- Mismo pipeline de pre-proceso que el dataset de entrenamiento (`scripts/pack_to_pkl.py`) — verificar paridad antes de embebér.

### 2.5 `seq_buffer.py`
```python
class SequenceBuffer:
    """FIFO de siluetas; emite (60,64,44) cada 30 frames nuevos."""
    def push(self, sil: np.ndarray) -> np.ndarray | None: ...
    def reset(self) -> None: ...
```
- Tamaño: 60. Stride de emisión: 30. Solapamiento: 50 %.
- Primera emisión: cuando se llena (60 frames). Siguientes: cada 30 frames.
- `reset()` limpia (cuando FSM dispara RESET).

### 2.6 `embed.py`
```python
class GaitEmbedder:
    def __init__(self, ckpt: Path, device: str = "cuda") -> None: ...
    def embed(self, seq: np.ndarray) -> np.ndarray:
        """Entrada (60,64,44) uint8. Salida (256,16) float32 L2-norm por feat."""
```
- Carga el state_dict de `gaitbase_ft_multisession_best_iter1200.pt` con `class_num=13` (igual que entrenamiento).
- Forward en eval, sin grad. Mismo pre-proceso que `cross_session_eval.py`.

### 2.7 `match.py`
```python
class GalleryMatcher:
    def __init__(self, gallery_npy: Path, index_json: Path, tau: float = 0.9807) -> None: ...
    def match(self, emb: np.ndarray) -> MatchResult:
        """Devuelve top-1 sujeto y similitud; flag unknown si sim<=tau."""
```
- Similitud: cosine sobre embedding aplanado a (256·16,) normalizado.
- `MatchResult = NamedTuple(subject:str|None, sim:float, unknown:bool, top5:list[(str,float)])`.

### 2.8 `confirm.py`
```python
class ConfirmationAccumulator:
    """N=2 secuencias consecutivas con mismo top-1 y sim>tau → CONFIRMADO."""
    def push(self, m: MatchResult) -> str | None:
        """Devuelve nombre confirmado o None."""
    def reset(self) -> None: ...
```
- Mantiene buffer de últimas N MatchResults.
- Si las N tienen mismo `subject` y todas con `unknown=False` → emite `subject` (1 sola vez por sesión activa hasta `reset`).
- Si una rompe la racha → resetea contador (no el FSM).

### 2.9 `log.py` + UI
- **CSV**: `logs/attendance_YYYY-MM-DD.csv`, columnas:
  `timestamp_iso, subject, mean_sim, n_sequences, frames_used, session_id`.
- Append-only. Una fila por confirmación; `session_id` es UUID corto generado en `WARMING→ACTIVE`.
- **UI** (solo `infer_live.py`): ventana OpenCV con bbox, nombre top-1, sim, estado FSM, contador de confirmación, banner "CONFIRMADO: <nombre>" 2 s al disparar.

## 3. Formato de la gallery (entrega 6.2)

- `gallery/embeddings.npy`: float32 array `(19, 256, 16)` — un embedding promedio por sujeto.
- `gallery/index.json`:
  ```json
  {
    "subjects": ["alexbojorquez", "carloscarrillo", ...],
    "ckpt": "gaitbase_ft_multisession_best_iter1200.pt",
    "tau": 0.9807,
    "feat_dim": [256, 16],
    "n_sequences_used": {"alexbojorquez": 4, ...}
  }
  ```
- **Estrategia de promedio**: por sujeto, embebér todas las secuencias de
  `data/pkl_multisession/<subj>/` (4 condiciones × 2 sesiones = hasta 8), L2-norm
  cada una, promediar, L2-norm el resultado. Misma técnica que en
  `openset_eval.py`.

## 4. Estados, latencias y FPS objetivo

| Etapa | Tiempo esperado en GTX 1650 | Notas |
|---|---|---|
| capture+decimate | ~2 ms | OpenCV |
| YOLOv11n | ~12 ms | TensorRT no necesario |
| RVM | ~25 ms | mobilenetv3 |
| seq_buffer | <1 ms | numpy roll |
| GaitBase forward | ~15 ms | secuencia de 60, batch 1 |
| match cosine | <1 ms | dot product 19×4096 |
| **Total por frame** | ~40 ms | margen para 15 fps (66 ms) |
| **Total por confirmación** | ~6 s | 2 secuencias × 4 s con stride |

## 5. Estructura de carpetas a crear

```
src/pipeline/
  __init__.py
  capture.py
  detect.py
  track.py
  segment.py
  seq_buffer.py
  embed.py
  match.py
  confirm.py
  log.py
  types.py        # BBox, MatchResult, etc.

scripts/
  build_gallery.py    # 6.2
  infer_video.py      # 6.3
  infer_live.py       # 6.4

configs/
  pipeline.yaml       # paths, τ, fps, ventana, stride, N, conf yolo, etc.

gallery/
  embeddings.npy
  index.json

logs/                 # se crea en runtime
```

## 6. Config `configs/pipeline.yaml` propuesto

```yaml
camera:
  source: 0
  target_fps: 15
  resize_to: [640, 360]

detector:
  weights: yolo11n.pt
  conf: 0.5
  class_id: 0

tracker:
  warmup_frames: 15
  reset_frames: 15
  iou_min: 0.3

segmenter:
  model: rvm_mobilenetv3
  out_size: [64, 44]
  bbox_pad: 0.10

sequence:
  window: 60
  stride: 30

embedder:
  ckpt: checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt
  class_num: 13

matcher:
  gallery_npy: gallery/embeddings.npy
  index_json: gallery/index.json
  tau: 0.9807

confirmation:
  n_consecutive: 2

logging:
  csv_dir: logs
```

## 7. Modos de uso

- **Offline** (6.3): `python scripts/infer_video.py --video path.mp4 --config configs/pipeline.yaml`. No abre ventana, solo logs.
- **En vivo** (6.4): `python scripts/infer_live.py --config configs/pipeline.yaml`. Abre ventana OpenCV, escribe CSV.

## 8. Casos límite y políticas

| Caso | Política |
|---|---|
| >1 persona en cuadro | Tomar bbox de mayor área. Loguear warning. (FSM no se altera). |
| Persona muy cerca/lejos (bbox <50 px o >80 % del frame) | Descartar frame del buffer pero no resetear FSM. |
| RVM produce silueta vacía (<5 % de píxeles) | Descartar frame del buffer. |
| Top-1 unknown durante secuencia | No incrementa confirmación; FSM sigue ACTIVE. |
| Confirmación duplicada del mismo sujeto | Loguear solo si pasaron >30 s desde la última. |
| Pérdida de cámara | Capture devuelve None 5 veces seguidas → exit graceful. |

## 9. Tests mínimos por módulo (pytest, en `tests/`)

- `test_track_fsm.py`: transiciones IDLE→WARMING→ACTIVE→RESET con secuencias mock.
- `test_seq_buffer.py`: emite cada 30 después de los primeros 60.
- `test_match.py`: gallery sintética 3 sujetos, distancias conocidas → top-1 y τ.
- `test_confirm.py`: N=2, alterna sujetos → no confirma; mismo sujeto 2× → confirma.

## 10. Lo que NO se hace en Fase 6

- Optimización ONNX/TensorRT (Fase 7).
- Quantización del checkpoint (Fase 7).
- Enrolamiento en vivo (decisión del usuario: gallery fija).
- Fusión de pose en score (cerrado en Fase 4).
- Soporte multicamera/multiángulo (out of scope).

## 11. Criterio de éxito de Fase 6

- 6.5 reporta sobre el usuario caminando frente a la C920:
  - **TPR ≥ 85 %** (16/19 sujetos confirmados al primer pase).
  - **FPR ≤ 5 %** sobre desconocidos (probarlo con 2–3 personas no enroladas).
  - **Latencia confirmación ≤ 8 s** desde entrada en cuadro.
  - **FPS sostenido ≥ 12** sin saturar VRAM (4 GB).
- Si no se cumple TPR, abrir 6.5b para diagnosticar (preproceso, τ, ruido de bbox).

---

## 12. Decisiones que necesito de ti antes de cerrar 6.1

1. **Detector**: ¿uso `yolo11n.pt` que ya está en repo, o prefieres validar con
   YOLOv8n? (Ambos caben, yolo11n es más rápido.)
2. **Segmentador**: ¿RVM como propongo, o quieres explorar SAM2 antes? RVM ya
   está validado para humanos en video, SAM2 es más pesado y necesita prompt.
3. **Logs**: ¿CSV plano o SQLite? Para 19 sujetos × asistencias diarias, CSV
   sobra. SQLite si quieres consultas históricas.
4. **UI**: ¿OpenCV puro (mínimo) o algo con `streamlit`/`gradio` para
   demos? Recomiendo OpenCV puro en 6.4; un wrapper web para 6.5 si quieres.

Con tus respuestas a estas 4 cierro 6.1 y arranco 6.2 (build_gallery.py).
