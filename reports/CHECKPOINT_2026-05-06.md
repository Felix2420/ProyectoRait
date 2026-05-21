# Checkpoint: Gait Recognition — 2026-05-06

**Responsable:** Usuario + Claude  
**Fase:** Fase 6 (Pipeline en vivo) — **6.4 completada, 6.5 pendiente**  
**Sesión:** Preparación de presentación de clase + validación open-set en vivo

---

## Estado Actual

### ✅ Completado
- **Fase 0–5:** Auditoría, dataset, extracción, empaquetado, GaitBase fine-tuning, multimodal descartado, calibración open-set
- **Fase 6.0–6.4:** Pipeline en vivo con C920, UI, FSM tracker, CSV log → **probado en vivo**
- **Presentación de clase:** Generados guión, estructura de slides y PDF técnico

### 📊 Métricas en producción (sin cambios desde 2026-05-03)
- Modelo: `checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt`
- Umbral: τ=0.9847 (recalibrado de 0.9807 para FAR=0% absoluto)
- EER ALL: 8.11%
- TAR@FAR=0%: 86.49% (32/37 genuinos)
- Gallery: 19 sujetos, multi-template (K≤4)
- Procesamiento: ~6-8 fps reales (GTX 1650)
- Confirmación: N=2 secuencias consecutivas con sim > τ

### ✅ Nuevo en esta sesión (2026-05-06)
- **Validación open-set en vivo confirmada:** persona NO enrolada genera sim=0.9403 < τ=0.9847 → UNK correcto
  - Top1 asignado: jorgeespinoza (vecino más cercano en galería)
  - Gap desconocido vs umbral: 0.9847 − 0.9403 = **0.0444** — margen holgado
  - Comportamiento open-set funciona correctamente en condiciones reales
- **Documentación de presentación generada:**
  - `reports/PRESENTACION_CLASE.md` — estructura de 18 slides con guión
  - `reports/GUION_PRESENTACION.md` — guión ajustado al PPTX real (18 slides, 4 personas)
  - `reports/Modelos_Scripts_Arquitectura.pdf` — PDF técnico de modelos y scripts
  - `reports/Fases_Proyecto_Explicacion.pdf` — PDF con cada fase explicada explícitamente

### 🚩 Limitaciones conocidas y aceptadas (sin cambios)
- Par `ricardomora ↔ hectorsanchez`: falla NN y NR (documentado, mismo build corporal)
- Dataset pequeño (19 sujetos, 90° lateral, 2 sesiones) limita generalización
- Multimodal (pose+silueta) no aporta a este ángulo/volumen → descartado con evidencia
- Cambio de cámara requiere recalibración de τ

### 🔧 Configuración acordada (sin cambios desde 2026-04-28)
- 1 persona a la vez (bbox mayor área)
- Ventana 60 frames @ 15 fps = 4 segundos
- Stride 30 frames (50% solapamiento)
- Trigger: bbox válido ≥15 frames antes de acumular
- Reset: bbox ausente >15 frames
- Log CSV con timestamp + identidad + confianza

---

## Artefactos del Proyecto

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

### Reportes generados
| Archivo | Descripción |
|---|---|
| `reports/GUION_PRESENTACION.md` | Guión de presentación oral (18 slides, 4 personas) |
| `reports/PRESENTACION_CLASE.md` | Estructura completa con estado del arte |
| `reports/Modelos_Scripts_Arquitectura.pdf` | PDF técnico: modelos, arquitecturas, scripts |
| `reports/Fases_Proyecto_Explicacion.pdf` | PDF: cada fase explicada explícitamente con código |

---

## ¿Qué falta?

### **Fase 6.5 — Validación End-to-End** (PENDIENTE)

1. **Pruebas formales con sujetos enrolados** — múltiples sesiones
   - Registrar TPR real en producción (¿coincide con 86.49% offline?)
   - Registrar FPR real (¿hay falsos positivos en condiciones reales?)
   - Latencia real por identificación en escenario de clase

2. **Validación de robustez**
   - Ropa diferente a s1 y s2 de entrenamiento
   - Velocidad variable
   - Iluminación distinta a la del dataset
   - ¿Qué pasa con oclusión parcial (mochila, abrigo)?

3. **Validación open-set extendida** ✓ parcial
   - ~~Persona desconocida rechazada~~ → **confirmado 2026-05-06** (sim=0.9403 < τ)
   - Pendiente: múltiples personas desconocidas en distintas condiciones

### **Fase 7 — Optimización y Despliegue** (SIGUIENTE a Fase 6.5)
1. Quantización INT8 / TensorRT (objetivo: embedding en ~30 ms vs ~100 ms actual)
2. SQLite en lugar de CSV para log persistente
3. Documentación final (manual de instalación, guía de calibración de τ)

---

## ¿Qué sigue?

**Opción A (recomendado):**  
Continúa a **Fase 6.5 — Validación End-to-End formal**
- Ejecuta `infer_live.py` con los 19 sujetos enrolados
- Documenta TPR/FPR/latencia real en `reports/15_validation_endtoend.md`
- Cuando esté validado, avanza a Fase 7

**Opción B:**  
Si hay presentación próxima inminente:
- El sistema está listo para demo en vivo o video grabado
- La validación open-set ya está confirmada (screenshot 2026-05-06)
- Se puede presentar con las métricas offline como referencia

**Opción C:**  
Si se quieren mejorar las métricas antes de Fase 7:
- Explorar umbral dinámico por sujeto (algunos tienen similitud media más baja)
- Revisar si hay sujetos con pocos templates en galería (valid_mask)

---

**Empieza próxima sesión con:**  
`"Estoy en Fase [6.5 / 7]. Resumen: [hallazgos]. ¿Continuamos?"`
