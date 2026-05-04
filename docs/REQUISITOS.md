# Requisitos y Decisiones — Fase 0

> Documento de cierre de Fase 0 del proyecto de Gait Recognition (pase de lista).
> Última actualización: 2026-04-20.

---

## 1. Dataset

| Propiedad | Valor |
|---|---|
| Sujetos | **19** |
| Videos por sujeto | 2 (`normal_*.mp4` + `rapido_*.mp4`) |
| Total videos | 38 |
| Frame rate | ~30 fps |
| Cámara | Logitech C920 fija |
| Ángulo | 90° lateral, único |
| Iluminación | Indoor controlada |
| Vestimenta | Misma entre los 2 videos del mismo sujeto; **distinta entre sujetos** (corregido en Fase 1) |
| Codec real | `mpeg4` part 2, 5.000 s exactos × 150 frames — re-encoding previo, no es salida directa de la C920 |
| Ubicación raw | `C:\Proyecto3\Proyecto-Reconocedor-IA\Dataset Crudo\<sujeto>\<condicion>.mp4` |
| Almacenamiento | Local (no se sube a storage externo) |

### Limitaciones a aceptar explícitamente

- **2 videos por sujeto es muy poco.** Las ventanas del mismo video no son independientes; cualquier split por ventana infla resultados. **Splits subject-disjoint para evaluación de calidad del modelo, video-disjoint para evaluación del sistema funcional.**
- **Dataset cerrado en condiciones (misma ropa, mismo ángulo, misma cámara).** El modelo NO generalizará a otra ropa, ángulo o iluminación. Esto es un sistema de pase de lista para *este aula*, no un sistema general de gait recognition.
- **Sin pretraining → fallo casi garantizado** (CLAUDE.md sec. 3). Es **obligatorio** usar pesos preentrenados de OpenGait y fine-tunear.
- **Riesgo apariencia→identidad** (Fase 1): cada sujeto tiene 2 videos con la **misma ropa** y ropa **distinta entre sujetos**. Un modelo silueta-only puede aprender la silueta-de-la-ropa como atajo de identidad. **Sumar rama de pose es prácticamente obligatorio** (no opcional como en Fase 0).
- **Frames útiles reales ≈ 100/video** (no 150): los videos arrancan con el sujeto parado ~1–1.5 s. En Fase 2 hay que recortar la secuencia útil con base en el bbox del tracker.
- **Sesión única (BRECHA ABIERTA — bloqueante antes de Fase 5).** Los 38 videos se grabaron en una sola sesión: mismo día, misma iluminación, misma cámara, mismo fondo, misma ropa por sujeto. Cualquier rank-1 que reportemos en Fase 3/4 está **inflado** porque gallery y probe comparten todas estas variables nuisance. Evidencia ya observada: zero-shot Fase 3 dio rank-1=100% con margen medio cos=+0.0036 — el modelo "acierta" pero no discrimina realmente, todos los embeddings caen en cos~0.99. Antes de cerrar Fase 5 hay que decidir explícitamente: **(a)** grabar una segunda sesión de ≥4 sujetos en otro día con ropa distinta (test set genuino para medir generalización), o **(b)** aceptar y documentar que el sistema solo funcionará en condiciones idénticas a las de captura. Mientras tanto, todas las métricas de Fase 3/4 deben leerse como "techo optimista", no como rendimiento esperado en producción.

---

## 2. Hardware

### Equipo único (entrenamiento Y despliegue)

| Componente | Detalle |
|---|---|
| GPU | NVIDIA GTX 1650, **4 GB VRAM** (Turing, sin Tensor Cores) |
| CPU | Intel Core i5-12450H (12.ª gen) |
| RAM | 16 GB DDR4 |
| OS | Windows 11 Home (10.0.26200) |
| Cámara despliegue | Logitech C920 (1080p@30 nativo) |

### VRAM medida con GaitBase (smoke test, AMP fp16, S=30 frames, 64×44)

| N identidades | Peak VRAM | Latencia inferencia |
|---|---|---|
| 1 | 317 MB | ~130 ms / sample |
| 8 | 649 MB | ~130 ms / sample |
| 32 | 1.9 GB | ~130 ms / sample |

**Conclusión:** GaitBase cabe holgado. Para entrenamiento dejaremos batch ~16 identidades × 8 muestras o ajustaremos según gradientes.

---

## 3. Targets de funcionamiento (definidos Fase 0)

| Métrica | Target | Justificación |
|---|---|---|
| Resolución captura | 720p @ 30 fps | C920 nativa |
| Resolución procesamiento | 480p (downscale) | Suficiente con persona ocupando >50% del alto |
| FPS pipeline end-to-end | **≥15 FPS** sostenido | Ciclo de marcha (~1 s) deja ~15 frames útiles |
| Latencia "marcar presente" | **≤2.5 s** desde que entra al cuadro | 1 ciclo + inferencia + votación 3 ventanas |
| Tamaño ventana gait | 30 frames @ 15 FPS efectivos = 2 s | Cubre ≥1 ciclo de marcha |
| Rank-1 baseline (Fase 3) | **≥70%** en split subject-disjoint honesto | Punto de partida realista con 19 sujetos |
| TAR@FAR=1% open-set (Fase 5) | **≥60%** | Conservador con misma ropa |

Targets **revisables** tras Fase 1 (auditoría del dataset).

---

## 4. Stack técnico

### Entorno de ejecución

- **Venv compartido:** `C:\Proyecto3\venv\` (Python 3.11.9).
- **PyTorch:** 2.5.1+cu121 con CUDA 12.1, GTX 1650 detectada y operativa.
- **Driver NVIDIA:** 581.83.

### Dependencias añadidas en Fase 0

```
kornia 0.8.2
einops 0.8.2
easydict 1.13
tabulate 0.10.0
```

(Las siguientes ya estaban: torch, torchvision, opencv, numpy, scipy, scikit-learn, matplotlib, mediapipe, ultralytics, pandas, tqdm.)

### Framework principal

- **OpenGait** clonado en `third_party/OpenGait/` (depth=1).
- Repo: https://github.com/ShiqiYu/OpenGait
- Modelo objetivo: **GaitBase (CVPR 2023)** — `configs/gaitbase/`.

### Pesos preentrenados disponibles

Descargado en `checkpoints/pretrained/GaitBase_Gait3D_120000.pt` (148 MB, 19.29 M params).
- **Origen:** GaitBase entrenado en **Gait3D-Parsing** por el equipo OpenGait.
- **Validación smoke test:** state_dict carga con `Missing keys: 0   Unexpected: 0`.
- **Limitación:** este checkpoint fue entrenado para Gait3D (3D body parsing), no exactamente CASIA-B. Para fine-tuning de Fase 3 evaluaremos:
  1. Reentrenar GaitBase en CASIA-B local (solo 5/124 sujetos disponibles → no es viable como pretraining real).
  2. Usar este checkpoint Gait3D como punto de partida.
  3. Pedir el checkpoint CASIA-B al autor (no está publicado en HuggingFace).

### CASIA-B local

- Ubicación: `C:\Proyecto3\CASIA-B\` (115 MB, **solo sujetos 001–005**).
- Formato: silueta binaria PNG por frame, estructura `<sujeto>/<condición>/<ángulo>/<frame>.png`.
- **No sirve como pretraining** (faltan 119 sujetos). Sí sirve como sanity check del pipeline OpenGait.

---

## 5. Estructura del proyecto

```
ProyectoChino/
├── CLAUDE.md                       # Reglas del proyecto (NO modificar)
├── README.md                       # Quickstart y comandos clave
├── configs/                        # YAMLs de proyecto (vacío hasta Fase 1)
├── data/
│   ├── raw/                        # Pointer a Dataset Crudo (no se duplica)
│   ├── silhouettes/                # PNG generadas en Fase 2
│   ├── poses/                      # NPY/JSON generadas en Fase 2
│   └── pkl/                        # Formato OpenGait, Fase 2
├── third_party/
│   └── OpenGait/                   # Framework clonado
├── src/
│   ├── preprocess/                 # Scripts de extracción Fase 2
│   └── inference/                  # Pipeline en vivo Fase 6
├── scripts/
│   └── smoke_test_gaitbase.py      # Validación Fase 0 ✓
├── checkpoints/
│   └── pretrained/
│       └── GaitBase_Gait3D_120000.pt  # 148 MB, validado
├── reports/                        # Auditorías y métricas (Fase 1+)
├── docs/
│   └── REQUISITOS.md               # Este archivo
└── legacy/                         # Código del intento previo (HybridGaitNet)
    ├── model.py                    # CNN-GEI + BiLSTM custom
    ├── train_v3.py                 # ⚠️ con bugs detectados (ver §7)
    ├── preprocess.py               # MediaPipe + MOG2 (no apto producción)
    ├── inference.py
    ├── validate.py
    ├── checkpoints/                # Pesos del baseline débil
    └── features/                   # NPY procesados con preprocess legacy
```

`data/raw/` queda vacío — el dataset crudo se referencia desde su ubicación original (`C:\Proyecto3\Proyecto-Reconocedor-IA\Dataset Crudo`) vía config en Fase 1, evitando duplicar 38 videos en disco.

---

## 6. Decisiones arquitectónicas tomadas

| Decisión | Elección | Alternativa descartada | Razón |
|---|---|---|---|
| Framework gait | **OpenGait** | Reescribir HybridGaitNet | SOTA, transfer learning, evaluación estandarizada |
| Modelo principal | **GaitBase** (Fase 3) | DeepGaitV2, SkeletonGait++ | DeepGaitV2 es más pesado para 4GB VRAM; GaitBase es el baseline canónico |
| Modelo multimodal | **SkeletonGait++** (Fase 4, candidato) | Fusión late propia | Modelo nativo silueta+esqueleto, pesos disponibles |
| Pose extractor | **RTMPose** (Fase 2) | MediaPipe (legacy) | Más estable y preciso para gait |
| Segmentación | **RVM o YOLO11-seg** (Fase 2) | MOG2 (legacy) | MOG2 falla en producción (warmup, sombras, ghosts) |
| Detector + tracker | **YOLO11n + ByteTrack** (Fase 6) | YOLOv8n | YOLO11 más reciente, mismo VRAM |
| Idioma de configs | **YAML + OmegaConf** | hydra puro | OmegaConf es más liviano; OpenGait ya usa YAML plano |
| Splits | **subject-disjoint** estricto para validar modelo | Ventanas mezcladas | Único split honesto |
| Almacenamiento | **Local en `C:\Proyecto3\`** | DVC, S3 | Solicitud explícita del usuario |

---

## 7. Bugs detectados en el código legacy (NO usar tal cual)

### `legacy/train_v3.py` — `batch_all_triplet_loss`

- Bucle Python triple O(N³) sobre el batch en CPU bloqueando GPU.
- Cuando todas las triplets son fáciles, `loss_vals` queda vacío y devuelve `tensor(0.0, requires_grad=True)`.
- El check `if loss.requires_grad:` en línea 216 omite silenciosamente épocas sin actualización.

### `legacy/train_v3.py` — split no subject-disjoint

- Línea 168–171: divide sujetos en `train_subj` / `val_subj`.
- Línea 231: galería se construye con **`all_subjects`** (incluye val).
- Eval mide identificación de ventanas no vistas de sujetos vistos, no discriminación abierta.
- Consecuencia: `val_acc` reportado **está inflado**.

### `legacy/preprocess.py` — siluetas con MOG2

- Necesita warmup (~50 primeros frames son basura → contaminan el GEI).
- Sensible a iluminación y sombras.
- Si el sujeto se queda quieto, se absorbe en el background.
- **Reemplazar por RVM o segmentador humano dedicado en Fase 2.**

### `legacy/preprocess.py` — keypoints faltantes

- Líneas 51 y 89: cuando MediaPipe/YOLO no detecta, copia el último frame.
- Si el sujeto sale del cuadro, se propaga el último kp por toda la secuencia.

### Conclusión

Los `best_model.pth` y `gallery_embeddings.npy` en `legacy/checkpoints/` deben tratarse como **referencia histórica**, no como baseline numérico válido.

---

## 8. Qué queda para Fase 1 (auditoría del dataset)

> No avanzar hasta aprobación explícita del usuario.

1. Inventariar los 38 videos crudos: duración, resolución, FPS real, formato.
2. Detectar videos corruptos o defectuosos.
3. Visualizar muestras (1 frame inicial + 1 medio + 1 final por video).
4. Verificar que el sujeto camina en cuadro completo en los 38 videos.
5. Validar que el ángulo realmente es 90° en todos.
6. Estimar cantidad de frames por sujeto realmente útil.
7. Definir splits **subject-disjoint** para evaluación del modelo y **video-disjoint** para evaluación del sistema funcional (con los 19 en gallery).
8. Reporte en `reports/01_dataset_audit.md`.

---

## 9. Reproducción del entorno

```bash
# Activar venv compartido
C:\Proyecto3\venv\Scripts\activate

# Validar PyTorch + GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Esperado: 2.5.1+cu121  True  NVIDIA GeForce GTX 1650

# Re-correr smoke test
python C:\Proyecto3\ProyectoChino\scripts\smoke_test_gaitbase.py
# Esperado: Missing keys: 0   Unexpected: 0   y forwards N=1..32 sin OOM
```
