# Sistema de Pase de Lista por Reconocimiento de Marcha

Identificación biométrica de personas a partir de su forma de caminar, en tiempo real, desde una cámara fija. Pensado como sustituto del pase de lista manual: cuando un sujeto enrolado pasa frente a la cámara, el sistema lo reconoce, lo marca presente y registra timestamp + confianza. Si la persona no está en la lista, la rechaza (open-set).

> **Estado:** Fase 6.4 cerrada — pipeline en vivo funcionando con cámara C920. Pendiente Fase 6.5 (validación end-to-end formal).
> **Última actualización:** 2026-05-03

---

## ¿Por qué reconocimiento de marcha y no rostro u otra biometría?

La marcha es difícil de falsificar, no requiere cooperación del sujeto y funciona a distancia con cámaras de bajo costo. El proyecto explora si, con un dataset modesto (19 sujetos, una sola cámara, 90° lateral, 2 sesiones), se puede llegar a un sistema operativo. La respuesta corta: **sí, con limitaciones documentadas**.

---

## Arquitectura del pipeline

```
┌────────┐   ┌──────────┐   ┌─────────┐   ┌────────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐
│ Cámara │──▶│ Detector │──▶│ Tracker │──▶│ Segmenter  │──▶│ Sequence │──▶│Embedder │──▶│ Matcher  │
│ (C920) │   │ (YOLOv11)│   │  IoU    │   │ (RVM)      │   │ buffer   │   │(GaitBase)│  │ + open-  │
│ 15 fps │   │ bbox     │   │ + reset │   │ silueta    │   │ 60×30    │   │ embed   │   │ set τ    │
└────────┘   └──────────┘   └─────────┘   └────────────┘   └──────────┘   └─────────┘   └─────┬────┘
                                                                                              │
                                                                                              ▼
                                                                                       ┌──────────────┐
                                                                                       │ Confirmación │
                                                                                       │ N=2 consec.  │
                                                                                       │ → CSV log    │
                                                                                       └──────────────┘
```

**Decisiones clave:**

| Componente | Elección | Por qué |
|---|---|---|
| Detector | YOLOv11n | Pequeño (6 MB), rápido en GTX 1650, suficiente para 1 persona a la vez |
| Tracker | IoU + warmup/reset frames | No necesitamos ReID — solo persistencia de bbox para acumular secuencia |
| Segmenter | Robust Video Matting (RVM) | Siluetas estables sin GrabCut clásico; downsample 0.25 para velocidad |
| Embedder | **GaitBase fine-tuned multisesión** | Baseline sólido de OpenGait; comprobado mejor que SkeletonGait++ en este dataset |
| Matching | Cosine similarity vs galería de prototipos | Multi-template por sujeto (varias secuencias promediadas) |
| Open-set | Umbral τ=0.9807 calibrado en val | Se rechaza si `max_sim < τ` |
| Confirmación | N=2 secuencias consecutivas | Reduce falsos positivos a costa de ~4 s extra de latencia |

Por qué **no** multimodal (silueta + pose): se probaron Ruta A (SkeletonGait++ con init GaitBase) y Ruta B (ST-GCN lite con fusión tardía). Ambas convergen a α*=1.0 en open-set — la pose no aporta información adicional con este dataset/ángulo. Se reasignó pose como *quality gate* (descartar siluetas mal extraídas) en runtime, no como rama de score. Detalle completo en `reports/PHASE4_CIERRE_2026-04-28.md`.

---

## Métricas

Evaluación con split **subject-disjoint** (13 train / 3 val / 3 test) sobre el modelo en producción `gaitbase_ft_multisession_best_iter1200.pt`:

| Métrica | Valor | Significado |
|---|---|---|
| EER (Equal Error Rate) | **8.11 %** | Punto donde FAR = FRR |
| TAR @ FAR = 0 % | **86.49 %** | De cada 100 enrolados, 86 son aceptados sin admitir un solo desconocido |
| Umbral operativo τ | **0.9807** | Calibrado en val con FAR=0% |
| FPS de procesamiento | 15 | Limitado por segmentación + embedding en GTX 1650 |
| Latencia identificación | ~4 s | 2 secuencias × 2 s (60 frames @ 15 fps cada una) |

**Limitaciones aceptadas (documentadas):**
- El par `ricardomora ↔ hectorsanchez` se confunde sistemáticamente en NN y NR. Causa probable: caminar muy similar + mismo rango de altura. Aceptado.
- Modelo entrenado solo con vista lateral 90°. Cambios fuertes de ángulo degradan el desempeño.
- Solo 2 sesiones por sujeto: el modelo no ha visto suficiente variación intra-sujeto (ropa, calzado, mochila).

---

## Instalación

### Requisitos
- Python 3.10+
- GPU NVIDIA con CUDA 11.8+ (probado en GTX 1650 4GB y RTX 2060 6GB)
- 16 GB RAM
- Cámara USB (probado con Logitech C920)

### Setup

```bash
git clone https://github.com/Felix2420/ProyectoRait.git
cd ProyectoRait

# Asegúrate de tener git-lfs instalado para descargar los modelos (.pt)
git lfs install
git lfs pull

# Entorno virtual
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

> **Importante:** los checkpoints (`*.pt`, ~712 MB) viven en Git LFS. Sin `git lfs pull` solo descargarás punteros.

---

## Uso

### Inferencia en vivo (modo producción)

```bash
python scripts/infer_live.py --config configs/pipeline.yaml
```

Lo que hace:
1. Abre la cámara (índice 0 por defecto), captura a 15 fps efectivos.
2. Detecta una persona a la vez (la de mayor área de bbox).
3. Acumula 60 frames de silueta, calcula embedding.
4. Compara vs galería de 19 sujetos.
5. Si dos secuencias consecutivas dan la misma identidad con sim > τ, escribe `logs/attendance_<fecha>.csv`.
6. UI muestra el bbox, la silueta segmentada y el match actual.

### Inferencia offline (sobre un video grabado)

```bash
python scripts/infer_video.py \
  --video path/al/video.mp4 \
  --config configs/pipeline.yaml \
  --output results.csv
```

### Listar cámaras disponibles

```bash
python scripts/list_cameras.py
```

### Reconstruir la galería (cuando enrolas o cambias sujetos)

```bash
python scripts/build_gallery.py \
  --pkl-dir data/pkl_multisession/ \
  --split configs/splits.yaml \
  --ckpt checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt \
  --output gallery/
```

---

## Configuración (configs/pipeline.yaml)

Todos los parámetros del runtime están en un solo archivo YAML. Ejemplo de los más relevantes:

```yaml
camera:
  source: 0              # índice de cámara
  target_fps: 15

sequence:
  window: 60             # frames por secuencia (4 s @ 15 fps)
  stride: 30             # 50 % de solapamiento

embedder:
  ckpt: checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt

matcher:
  tau: 0.9807            # umbral open-set; bajarlo = más permisivo

confirmation:
  n_consecutive: 2       # N secuencias para confirmar identidad
```

Para producción con condiciones distintas (cámara, distancia, iluminación), recalibra τ siguiendo `reports/06_openset_eval.md`.

---

## Estructura del repositorio

```
ProyectoRait/
├── README.md                      # Este archivo
├── CLAUDE.md                      # Roadmap completo + reglas de trabajo
├── SETUP_NEW_PC.md                # Guía para configurar otra máquina
├── CLAUDE_INSTRUCTIONS_2060.md    # Instrucciones específicas para la 2060
├── requirements.txt               # Dependencias fijadas
├── requirements.lock.txt          # Lock completo (pip freeze)
│
├── configs/                       # YAMLs de pipeline y splits
│   ├── pipeline.yaml
│   ├── splits.yaml
│   ├── partition_finetune.json
│   └── partition_zeroshot.json
│
├── scripts/                       # Entrenamiento, evaluación, inferencia
│   ├── infer_live.py              # ⭐ Pipeline en vivo (Fase 6.4)
│   ├── infer_video.py             # Inferencia offline
│   ├── build_gallery.py           # Construye galería de embeddings
│   ├── finetune_gaitbase_multisession.py
│   ├── openset_eval.py            # Evaluación open-set
│   └── ... (entrenamientos, audits, packs)
│
├── src/
│   ├── pipeline/                  # Módulos del pipeline en vivo
│   │   ├── capture.py             # Cámara
│   │   ├── detect.py              # YOLO
│   │   ├── track.py               # IoU tracker
│   │   ├── segment.py             # RVM
│   │   ├── seq_buffer.py          # Acumula secuencias
│   │   ├── embed.py               # GaitBase
│   │   ├── match.py               # Matching + open-set
│   │   ├── confirm.py             # Confirmación N=2
│   │   └── log.py                 # Salida CSV
│   ├── models/stgcn_lite.py       # Modelo de pose (Fase 4, descartado)
│   └── preprocess/                # Extracción de secuencias
│
├── checkpoints/                   # Modelos (Git LFS, ~712 MB)
│   ├── pretrained/
│   │   └── GaitBase_Gait3D_120000.pt
│   └── finetune/
│       └── gaitbase_ft_multisession_best_iter1200.pt   ⭐ Producción
│
├── gallery/                       # Embeddings prototipo (no .pt, son .npy)
│   └── index.json
│
├── reports/                       # 16 reportes de fases + auditorías
│   ├── 01_dataset_audit.md
│   ├── 02_dataset_preparation.md
│   ├── 06_openset_eval.md
│   ├── 14_pipeline_design.md
│   ├── PHASE4_CIERRE_2026-04-28.md
│   └── CHECKPOINT_2026-05-03.md
│
├── data/                          # Dataset (NO versionado, privado)
└── third_party/OpenGait           # Submodule del framework base
```

---

## Roadmap

| Fase | Descripción | Estado |
|---|---|---|
| 0 | Setup y descubrimiento | ✅ Cerrada |
| 1 | Auditoría del dataset | ✅ Cerrada |
| 2 | Preparación + splits subject-disjoint | ✅ Cerrada |
| 3 | Baseline GaitBase fine-tuned | ✅ Cerrada |
| 4 | Multimodal (silueta + pose) | ✅ Cerrada como Caso C — fusión descartada |
| 5 | Open-set + calibración de τ | ✅ Cerrada |
| 6.0–6.4 | Diseño + pipeline en vivo + UI + log | ✅ Cerrada |
| **6.5** | **Validación end-to-end formal (TPR/FPR/latencia en sesiones reales)** | ⏳ **Próxima** |
| 7 | Optimización (TensorRT/ONNX) y despliegue | ⬜ Pendiente |

Ver `CLAUDE.md` para el roadmap completo con criterios de aceptación por fase.

---

## Documentación clave

| Archivo | Contenido |
|---|---|
| `CLAUDE.md` | Especificación del proyecto, reglas de trabajo, roadmap |
| `reports/CHECKPOINT_2026-05-03.md` | Estado actual y próximos pasos |
| `reports/14_pipeline_design.md` | Diagrama de bloques + contratos de módulos |
| `reports/06_openset_eval.md` | Calibración del umbral τ |
| `reports/PHASE4_CIERRE_2026-04-28.md` | Por qué se descartó multimodal |
| `SETUP_NEW_PC.md` | Pasos para clonar el entorno en otra máquina |

---

## Dataset

**No está versionado en este repo** (privacidad de los sujetos).

- **19 sujetos**, 2 sesiones cada uno, ~30 fps, 90° lateral, indoor.
- Formatos: PNG de siluetas + keypoints en NPY (RTMPose).
- Empaquetado a `.pkl` por secuencia (formato OpenGait).
- Acceso bajo solicitud: jf990015@gmail.com.

---

## Troubleshooting

**No reconoce a nadie / todo es "Desconocido":**
- Verifica que la galería existe: `gallery/embeddings.npy` y `gallery/index.json`.
- Si recolectaste data nueva, reconstruye galería con `scripts/build_gallery.py`.
- Considera bajar `tau` en `configs/pipeline.yaml` para diagnosticar (no en producción).

**Cámara no abre:**
```bash
python scripts/list_cameras.py
```
Cambia `camera.source` en el YAML al índice correcto.

**Lento en GPU pequeña (GTX 1650):**
- Baja `target_fps` a 10.
- Sube `segmenter.downsample_ratio` a 0.4.
- Más opciones en Fase 7 (TensorRT, INT8).

**`git lfs pull` falla / modelos faltan:**
```bash
git lfs install
git lfs fetch --all
git lfs checkout
```

---

## Contacto

- **Autor:** Jesús Félix
- **Email:** jf990015@gmail.com
- **Repositorio:** https://github.com/Felix2420/ProyectoRait

---

## Reconocimientos

- [OpenGait](https://github.com/ShiqiYu/OpenGait) — framework base de gait recognition.
- GaitBase (Fan et al.) — modelo baseline.
- [RTMPose](https://github.com/open-mmlab/mmpose) — pose estimation.
- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) — segmentación.
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) — detección.
