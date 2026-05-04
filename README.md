# Gait Recognition — Sistema de Pase de Lista Automático

**Estado actual:** Fase 6.4 (Pipeline en vivo probado)  
**Última actualización:** 2026-05-03

---

## 📋 Descripción General

Sistema de identificación automática de personas basado en reconocimiento de marcha (gait recognition). Utiliza video de una cámara fija para:

1. **Identificar sujetos enrolados** contra una galería de 19 individuos
2. **Rechazar desconocidos** (open-set recognition) con umbral calibrado
3. **Registrar asistencia** en tiempo real con timestamp y nivel de confianza

**Modelo base:** GaitBase (siluetas binarias) fine-tuned en dataset multisesión.

---

## 🎯 Métricas de Desempeño

| Métrica | Valor |
|---------|-------|
| **EER (Equal Error Rate)** | 8.11 % |
| **TAR @ FAR=0%** | 86.49 % |
| **Umbral open-set (τ)** | 0.9807 |
| **FPS procesamiento** | 15 fps |
| **Latencia identificación** | ~4 s (2 secuencias) |

### Limitaciones conocidas
- **Par problemático:** `ricardomora ↔ hectorsanchez` falla (documentado)
- **Multimodal descartado:** pose + silueta no aportó mejora en este escenario
- **Generalización limitada:** 19 sujetos, 90° lateral, 2 sesiones

---

## 🚀 Instalación

### Requisitos
- Python 3.9+
- GPU NVIDIA (CUDA 11.8+) recomendada
- 8 GB RAM mínimo

### Setup

```bash
git clone https://github.com/TU_USUARIO/ProyectoChino.git
cd ProyectoChino
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## ⚡ Quickstart

### Inferencia en vivo

```bash
python scripts/infer_live.py \
  --checkpoint checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt \
  --gallery gallery/index.json \
  --output logs/attendance.csv
```

### Inferencia offline (video)

```bash
python scripts/infer_video.py \
  --video test_video.mp4 \
  --checkpoint checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt \
  --gallery gallery/index.json \
  --output results.csv
```

---

## 📁 Estructura

```
ProyectoChino/
├── README.md
├── CLAUDE.md                    # Especificación técnica completa
├── requirements.txt
├── scripts/                     # Scripts de entrenamiento e inferencia
├── checkpoints/                 # Modelos pre-entrenados
├── gallery/                     # Galería de embeddings
├── data/                        # Dataset (no versionado)
└── reports/                     # Documentación de análisis
```

---

## 📊 Dataset

- **19 sujetos**, 2 sesiones, 30 fps
- **Formato:** Siluetas + keypoints
- **Privado:** Contactar jf990015@gmail.com para acceso

---

## 🔧 Parámetros Configurables

```yaml
fps_process: 15              # Procesamiento 15 fps
window_size: 60             # 4 segundos de ventana
stride: 30                  # 50% overlapping
sequences_confirm: 2        # N=2 para confirmar
threshold: 0.9807           # Umbral open-set
```

---

## 📚 Documentación

- **CLAUDE.md** — Especificación técnica y roadmap (7 fases)
- **CHECKPOINT_2026-05-03.md** — Estado actual del proyecto
- **PHASE4_CIERRE_2026-04-28.md** — Análisis multimodal

---

## 🐛 Troubleshooting

**Cámara no detectada:**
```bash
python scripts/list_cameras.py
```

**Reconstruir galería:**
```bash
python scripts/build_gallery.py \
  --dataset data/pkl_multisession/ \
  --split_json splits/split_13_3_3.json \
  --output gallery/
```

---

## 👤 Contacto

Email: jf990015@gmail.com  
Fase actual: 6 (validación end-to-end próxima)

---

**Ver CLAUDE.md para documentación técnica completa.**
