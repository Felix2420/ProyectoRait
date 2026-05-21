# Checkpoint: Gait Recognition — 2026-05-08

**Responsable:** Usuario + Claude
**Fase:** Fase 6 — **6.4 completada, 6.5 bloqueada por bug en runtime**
**Sesión:** Intento de validación end-to-end → bug CUBLAS en `infer_live.py`

---

## Estado Actual

### ✅ Completado (sin cambios desde 2026-05-06)
- **Fase 0–5:** Auditoría, dataset, extracción, empaquetado, GaitBase fine-tuning, multimodal descartado, calibración open-set
- **Fase 6.0–6.4:** Pipeline en vivo con C920, UI, FSM tracker, CSV log
- **Validación open-set en vivo:** persona NO enrolada → sim=0.9403 < τ=0.9847 (gap 0.0444), confirmada el 2026-05-06
- **Documentación de presentación:** `GUION_PRESENTACION.md`, `PRESENTACION_CLASE.md`, `Modelos_Scripts_Arquitectura.pdf`, `Fases_Proyecto_Explicacion.pdf`

### 📊 Métricas en producción (sin cambios desde 2026-05-03)
- Modelo: `checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt`
- Umbral: τ=0.9847 (FAR=0% absoluto)
- EER ALL: 8.11%
- TAR@FAR=0%: 86.49% (32/37 genuinos)
- Gallery: 19 sujetos, multi-template (K≤4)
- Procesamiento: ~6-8 fps reales (GTX 1650)
- Confirmación: N=2 secuencias consecutivas con sim > τ

### 🆕 Nuevo en esta sesión (2026-05-08)
- **Bug detectado en `infer_live.py`:** `RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED`
  - Stack: aflora en `modeling/modules.py:106` → `x.matmul(self.fc_bin)` (capa `SeparateFCs`)
  - Origen del call: `src/pipeline/embed.py:79` dentro de `torch.amp.autocast("cuda")`
  - Kernel reportado: `cublasGemmStridedBatchedEx(... CUDA_R_16F ... CUBLAS_GEMM_DEFAULT_TENSOR_OP)` (FP16 + tensor-op)
  - **Hardware relevante:** GTX 1650 (TU117) **no tiene Tensor Cores** → autocast FP16 puede fallar en cuBLAS según driver/versión
- **Hipótesis ordenadas (sin validar todavía):**
  1. OOM enmascarado por error asíncrono CUDA (4 GB es muy ajustado para YOLO + RVM + GaitBase concurrentes)
  2. Error asíncrono de una op previa (RVM o YOLO) reportado tarde en la GEMM
  3. Autocast FP16 incompatible con cuBLAS en GTX 1650 sin tensor cores
- **Acciones de diagnóstico definidas (pendientes):**
  1. `nvidia-smi` — descartar OOM y procesos competidores
  2. `$env:CUDA_LAUNCH_BLOCKING="1"; python scripts\infer_live.py` — localizar el error verdadero
  3. Desactivar autocast en `src/pipeline/embed.py:78-79` (forzar FP32) si 1–2 no aclaran

### 🚩 Limitaciones conocidas y aceptadas (sin cambios)
- Par `ricardomora ↔ hectorsanchez`: falla NN y NR (mismo build corporal)
- Dataset pequeño (19 sujetos, 90° lateral, 2 sesiones) limita generalización
- Multimodal descartado con evidencia
- Cambio de cámara requiere recalibración de τ

### 🔧 Configuración acordada (sin cambios)
- 1 persona a la vez (bbox mayor área)
- Ventana 60 frames @ 15 fps = 4 segundos
- Stride 30 frames (50% solapamiento)
- Trigger: bbox válido ≥15 frames antes de acumular
- Reset: bbox ausente >15 frames
- Log CSV con timestamp + identidad + confianza

---

## Artefactos del Proyecto (sin cambios)

### Checkpoints
| Archivo | Estado | Descripción |
|---|---|---|
| `checkpoints/pretrained/GaitBase_Gait3D_120000.pt` | Referencia | Preentrenado Gait3D 4,000 sujetos |
| `checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt` | ✅ **Producción** | Fine-tune s1+s2, iter 1,200 |
| `checkpoints/finetune/skeletongaitpp_best_iter1000.pt` | ❌ Archivado | Multimodal descartado |

### Galería
| Archivo | Contenido |
|---|---|
| `gallery/embeddings.npy` | (19, K_max, 256, 16) float32 |
| `gallery/valid_mask.npy` | (19, K_max) bool |
| `gallery/index.json` | Metadata de sujetos y templates |

### Redes neuronales en producción
| # | Red | Rol | Archivo |
|---|---|---|---|
| 1 | YOLOv11n (Ultralytics) | Detector → bbox | `src/pipeline/detect.py` (`yolo11n.pt`) |
| 2 | RVM MobileNetV3 | Segmentación → silueta 64×44 | `src/pipeline/segment.py` (`PeterL1n/RobustVideoMatting`) |
| 3 | GaitBase / ResNet9 | Embedder → 256×16 → flat 4096 | `src/pipeline/embed.py` |

---

## ¿Qué falta?

### **Bloqueador inmediato: bug CUBLAS en `infer_live.py`**
1. Ejecutar plan de diagnóstico (nvidia-smi → CUDA_LAUNCH_BLOCKING=1 → desactivar autocast)
2. Documentar la causa raíz cuando se confirme
3. Aplicar fix (probable: forzar FP32 en `embed.py` si la GTX 1650 no tolera FP16 cuBLAS)

### **Fase 6.5 — Validación End-to-End** (PENDIENTE, depende de fix)
1. Pruebas formales con sujetos enrolados → TPR/FPR/latencia reales
2. Robustez (ropa, velocidad, iluminación, oclusión)
3. Open-set extendido (más desconocidos en distintas condiciones)

### **Fase 7 — Optimización y Despliegue** (después de 6.5)
1. Quantización INT8 / TensorRT (objetivo: embedding ~30 ms vs ~100 ms actual)
2. SQLite en lugar de CSV
3. Documentación final (manual, guía de calibración de τ)

---

## ¿Qué sigue?

**Siguiente sesión empieza con:**
1. Correr `nvidia-smi` con `infer_live.py` lanzado para ver consumo real de VRAM
2. Si VRAM < 3.5 GB, no es OOM → probar `CUDA_LAUNCH_BLOCKING=1`
3. Si el error sigue apuntando a la GEMM FP16, desactivar `torch.amp.autocast` en `src/pipeline/embed.py:78-79`
4. Una vez resuelto, retomar Fase 6.5 con `infer_live.py`

**Empieza próxima sesión con:**
`"Estoy en Fase 6.5 bloqueada por CUBLAS_STATUS_EXECUTION_FAILED. Resumen del diagnóstico: [...]. ¿Continuamos?"`
