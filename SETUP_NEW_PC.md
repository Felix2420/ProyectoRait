# SETUP en PC nueva — ProyectoChino

Guía paso a paso para levantar el proyecto en otra máquina (target real:
**RTX 2060 6GB**) y dejarlo corriendo el **pipeline en vivo** o la **evaluación
end-to-end**.

**Estado actual (snapshot 2026-04-29):**
- Fases 0–5 + 6.0–6.4 cerradas. Pipeline en vivo con C920 ya probado funcionando.
- Pendiente: **6.5** (validación end-to-end con TPR/FPR/latencia/FPS) y **7**
  (optimización ONNX/TensorRT).
- Modelo de producción: `gaitbase_ft_multisession_best_iter1200.pt` (silueta sola).
- τ producción = **0.9847** (multi-template, gallery 4×/sujeto).
- Pipeline solo-silueta; los keypoints/heatmaps/SkeletonGait++ están archivados
  (ver `reports/PHASE4_CIERRE_2026-04-28.md` y `CHECKPOINT_2026-04-28_2310.md`).

---

## 0. Antes de empezar (en la PC vieja)

```bash
# Activar el venv actual
C:\Proyecto3\venv\Scripts\activate

# 1. Refrescar requirements.lock.txt (por si cambió algo desde el último export)
pip freeze > C:\Proyecto3\ProyectoChino\requirements.lock.txt

# 2. Empaquetar el proyecto en 4 zips en C:/Proyecto3/export/
python C:\Proyecto3\ProyectoChino\scripts\export_project.py
```

Produce (tamaños reales del export 2026-04-29):

| Zip | Tamaño | Contenido |
|---|---|---|
| `ProyectoChino_code.zip`        |  33 MB | código completo: src/, scripts/, configs/, **gallery/**, reports/, docs/, third_party/, yolo11n.pt, requirements.lock.txt, CLAUDE.md, este SETUP |
| `ProyectoChino_checkpoints.zip` | 188 MB | `GaitBase_Gait3D_120000.pt` (pretrained) + `gaitbase_ft_multisession_best_iter1200.pt` (**modelo prod**) + `gaitbase_ft_best_iter400.pt` (Fase 3 por si regresamos) |
| `ProyectoChino_pkls.zip`        |  20 MB | pkl/, pkl_s2/, pkl_multisession/, pkl_multimodal/ |
| `ProyectoChino_data_raw.zip`    | 187 MB | videos crudos s1 + s2 (75 archivos) |

Total ≈ **428 MB**. Cabe en USB chico o se sube directo a Drive sin partir.

**No incluidos a propósito** (palanca de keypoints archivada):
- `checkpoints/finetune/skeletongaitpp_*.pt` (~70 MB)
- `checkpoints/finetune/stgcn_lite_best.pt`
- `data/heatmaps/`, `data/poses/`, `data/poses_raw/`

Si en la PC nueva se quiere retomar Ruta A/B (palancas 6/7 del CHECKPOINT),
copiarlos aparte. El pipeline **NO los necesita**.

Copia los 4 zips a la PC nueva (USB, disco externo, cloud).

---

## 1. Hardware recomendado en PC nueva

- **GPU**: ≥ 4 GB VRAM. La GTX 1650 4GB ya corre el pipeline a 18–24 FPS; la
  **RTX 2060 6GB** debe rendir parecido o mejor (mismas TFLOPs FP32, mejor
  caché). Para reentrenar Fase 4 multimodal en serio se necesitan ≥ 8 GB.
- **RAM**: 16 GB mínimo (32 GB ideal si vas a regenerar pkls).
- **Disco**: 30 GB libres (export descomprimido + venv + logs).
- **Cámara**: Logitech C920 (la usada en validación). Cualquier USB UVC sirve,
  pero hay que recalibrar τ si cambia mucho la óptica.
- **OS**: Windows 10/11 o Linux. Instrucciones abajo son para Windows.

---

## 2. Software base (una sola vez)

### 2.1 Python 3.11.9

Descarga de https://www.python.org/downloads/release/python-3119/.
Importante: mismo minor (3.11) que la PC vieja para que las wheels compiladas sirvan.

### 2.2 Git

Descarga e instala Git para Windows: https://git-scm.com/download/win.

### 2.3 CUDA Toolkit

- RTX 2060 / 30xx / 40xx → **CUDA 12.1 o 12.4** (coincide con `torch==2.5.1+cu121`).
- Driver NVIDIA reciente (≥ 552.x recomendado).
- Descarga: https://developer.nvidia.com/cuda-downloads.

Verifica con:

```bash
nvidia-smi     # debe listar tu GPU y CUDA version
nvcc --version # debe responder
```

### 2.4 Build Tools (por si acaso)

Visual Studio 2022 Build Tools con "Desktop development with C++" — algunos
paquetes (mmcv, ciertos wheels) necesitan compilar. Si todo instala por wheel,
no hace falta, pero es bueno tenerlo listo.

---

## 3. Descompresión y estructura

Decide dónde vivirá el proyecto. **Recomendación firme: `C:\Proyecto3\`** (igual
a la PC actual) porque hay paths absolutos en `configs/pipeline.yaml` y en
varios scripts que asumen esa raíz.

```bash
# Crear estructura base
mkdir C:\Proyecto3
cd C:\Proyecto3

# Descomprimir los 4 zips (todos expanden en ProyectoChino/)
tar -xf ProyectoChino_code.zip         -C C:\Proyecto3\
tar -xf ProyectoChino_checkpoints.zip  -C C:\Proyecto3\
tar -xf ProyectoChino_pkls.zip         -C C:\Proyecto3\
tar -xf ProyectoChino_data_raw.zip     -C C:\Proyecto3\
```

(En Windows moderno `tar` viene incluido y entiende zip. Si no, usa 7-Zip.)

Estructura esperada después:

```
C:\Proyecto3\ProyectoChino\
  CLAUDE.md
  README.md
  SETUP_NEW_PC.md
  CLAUDE_INSTRUCTIONS_2060.md
  requirements.lock.txt
  yolo11n.pt
  configs/
    pipeline.yaml          # parámetros runtime del pipeline
    splits.yaml
  gallery/                  # CRÍTICO para producción
    embeddings.npy          # (19, 4, 256, 16)
    valid_mask.npy          # (19, 4)
    index.json              # metadata + caveat de calibración
  data/
    pkl/                    # s1 procesado (19 sujetos)
    pkl_s2/                 # s2 procesado
    pkl_multisession/
    pkl_multimodal/         # heatmaps duales (no usados en prod)
  data_raw/                 # del zip raw
    s1/                     # videos sesión 1
    s2/                     # videos sesión 2 (ex dataset2/)
  checkpoints/
    pretrained/GaitBase_Gait3D_120000.pt
    finetune/gaitbase_ft_multisession_best_iter1200.pt   # MODELO PROD
    finetune/gaitbase_ft_best_iter400.pt
  src/
    pipeline/               # 10 módulos: capture, detect, track, segment,
                            # seq_buffer, embed, match, confirm, log, types
  scripts/
    smoke_test_gaitbase.py
    cross_session_eval.py
    openset_eval.py
    openset_eval_multitemplate.py   # 6.2b
    build_gallery.py                # 6.2
    infer_video.py                  # 6.3 — un video offline
    eval_pipeline_offline.py        # 6.3 — batch
    infer_live.py                   # 6.4 — cámara C920
    export_project.py
  reports/
    CHECKPOINT_2026-04-28_2310.md   # resumen actual del estado
    PHASE4_CIERRE_2026-04-28.md
    14_pipeline_design.md
    15_openset_multitemplate.{json,csv,png}
    16_pipeline_offline_eval.{json,csv}
    ...
  docs/
  third_party/              # OpenGait (puedes re-clonar para limpiar)
  legacy/                   # NO va en el zip
```

Re-clonar OpenGait (opcional, más limpio):

```bash
cd C:\Proyecto3\ProyectoChino\third_party
rmdir /s /q OpenGait
git clone --depth 1 https://github.com/ShiqiYu/OpenGait
```

---

## 4. Crear venv y dependencias

```bash
cd C:\Proyecto3
python -m venv venv
venv\Scripts\activate

python -m pip install --upgrade pip

# PyTorch con CUDA (ajusta cu121 o cu124 según tu CUDA toolkit)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Resto del stack exacto como la PC vieja
pip install -r C:\Proyecto3\ProyectoChino\requirements.lock.txt
```

Si `requirements.lock.txt` choca con tu CUDA, filtra torch/nvidia/triton:

```bash
findstr /v /i "torch nvidia triton" C:\Proyecto3\ProyectoChino\requirements.lock.txt > requirements.clean.txt
pip install -r requirements.clean.txt
```

---

## 5. Verificación (sanity checks)

### 5.1 Torch ve la GPU

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Esperado: `2.5.1+cu121 True NVIDIA GeForce RTX 2060`.

### 5.2 Smoke test GaitBase

```bash
cd C:\Proyecto3\ProyectoChino
python scripts\smoke_test_gaitbase.py
```

Esperado: `Missing keys: 0  Unexpected: 0`.

### 5.3 Reproducir multi-template open-set (Fase 6.2b — debe dar EER 0% / τ=0.9847)

```bash
python scripts\openset_eval_multitemplate.py
```

Esperado (mismos números que `reports\15_openset_multitemplate.json`):

```
ALL    EER=0.00%   τ@FAR=0%=0.9847   TAR@FAR=0%=100%   rank-1=100%
```

### 5.4 Reproducir pipeline offline end-to-end (Fase 6.3)

```bash
python scripts\eval_pipeline_offline.py
```

Esperado (mismos números que `reports\16_pipeline_offline_eval.json`):

```
Videos procesados: 37
Videos con secuencia: 29 (78%)
Rank-1 sobre con-secuencia: 29/29 = 100%
Aceptados bajo τ=0.9847: 29/29 = 100%
```

Si los números coinciden (±0.1%), el ambiente está OK.

### 5.5 Pipeline en vivo con C920 (Fase 6.4)

Conecta la cámara y corre:

```bash
python scripts\infer_live.py
```

Debe abrir la ventana OpenCV con bbox + nombre + sim + estado FSM. Ajusta los
parámetros en `configs/pipeline.yaml` si la C920 no es `index 0`.

---

## 6. Próximos pasos del proyecto en la PC nueva

Lee **`reports/CHECKPOINT_2026-04-28_2310.md`** (snapshot completo del estado).
Lo que queda:

1. **Fase 6.5 — Validación end-to-end formal** caminando frente a la cámara:
   TPR (sujetos enrolados), FPR (sujetos no enrolados), latencia de
   confirmación, FPS sostenido. Pone a prueba la política N=2 confirmaciones
   y τ=0.9847 en condiciones reales.
2. **Fase 7 — Optimización**: ONNX/TensorRT, cuantización INT8 si el FPS lo pide.
3. **(Opcional) Palanca 9 del CHECKPOINT**: pose como quality gating en runtime
   (RTMPose por frame, descartar si <50% keypoints con conf>0.5). Cierra el
   requisito multimodal sin tocar el modelo.

---

## 7. Claude Code en la PC nueva

Cuando arranques Claude Code en la PC nueva, di:

> "Retomamos ProyectoChino. Ya hice el setup siguiendo SETUP_NEW_PC.md.
> Lee `reports/CHECKPOINT_2026-04-28_2310.md` para ver el estado actual.
> Quiero arrancar Fase 6.5 / Fase 7 / [lo que sea]."

Si la memoria no migró (`~/.claude/`), con leer el CHECKPOINT y CLAUDE.md
sec.10 tiene contexto suficiente. Hay también un `CLAUDE_INSTRUCTIONS_2060.md`
con notas específicas de esa máquina.

---

## 8. Troubleshooting conocido

- **`UnicodeEncodeError: 'charmap' codec can't encode character 'τ'`**: símbolo
  τ en prints con cp1252 (Windows). Ya reemplazado por `tau` en
  `openset_eval.py` y `cross_session_eval.py`.
- **Stdout en jobs largos**: usa `python -u script.py` para evitar buffering
  de 64KB cuando rediriges a archivo.
- **`ImportError: DLL load failed` en torch**: desajuste CUDA toolkit vs torch.
  Reinstala torch con el index URL correcto a tu CUDA.
- **`FileNotFoundError: configs/splits.yaml`** o `configs/pipeline.yaml`:
  corre los scripts desde `C:\Proyecto3\ProyectoChino\`, no desde `scripts/`.
- **`gallery/embeddings.npy` no encontrado**: el code.zip lo trae. Si por
  alguna razón faltara, regenéralo con `python scripts\build_gallery.py`.
- **C920 no abre en `infer_live.py`**: en `configs/pipeline.yaml` cambia
  `capture.source` (default `0`) al índice correcto. Verifica con
  `python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"`.
- **VRAM insuficiente en RTX 2060 6GB**: el pipeline solo-silueta cabe en
  ~2 GB. Si entrenas algo, baja `batch_size` en el config correspondiente.

---

## 9. Checklist rápido

- [ ] Python 3.11.9 instalado
- [ ] Git instalado
- [ ] CUDA toolkit 12.1+ compatible con la RTX 2060
- [ ] 4 zips copiados y descomprimidos en `C:\Proyecto3\`
- [ ] venv creado en `C:\Proyecto3\venv` (sin punto)
- [ ] `torch.cuda.is_available()` devuelve `True`
- [ ] `scripts\smoke_test_gaitbase.py` pasa
- [ ] `openset_eval_multitemplate.py` reproduce EER 0% / τ=0.9847
- [ ] `eval_pipeline_offline.py` reproduce 29/29 = 100%
- [ ] `infer_live.py` abre la C920 y muestra overlay
- [ ] Leíste `reports/CHECKPOINT_2026-04-28_2310.md`

Si todo eso está OK → listo para Fase 6.5 / Fase 7.
