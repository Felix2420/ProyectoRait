# Checkpoint: Gait Recognition — 2026-05-03

**Responsable:** Usuario + Claude  
**Fase:** Fase 6 (Pipeline en vivo) — **6.4 completada, 6.5 pendiente**

---

## Estado Actual

### ✅ Completado
- **Fase 0–5:** Auditoría, dataset, GaitBase baseline, open-set calibration
- **Fase 6.0:** Cierre formal de Fase 4 (decisión de silueta sola, pose→quality gating)
- **Fase 6.1–6.3:** Diseño pipeline, build_gallery.py, infer_video.py
- **Fase 6.4:** infer_live.py con C920 + UI + CSV log → **probado en vivo**

### 📊 Métricas en producción
- Modelo: `checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt`
- Umbral: τ=0.9807 (FAR=0%)
- EER ALL: 8.11 %
- TAR@FAR=0%: 86.49 %
- Gallery: 19 sujetos fijos
- Procesamiento: 15 fps
- Confirmación: N=2 secuencias consecutivas con sim>τ

### 🚩 Limitaciones conocidas y aceptadas
- Par `ricardomora ↔ hectorsanchez`: falla NN y NR (documentado)
- Dataset pequeño (19 sujetos, 90° lateral, 2 sesiones) limita generalización
- Multimodal (pose+silueta) no aporta mejora en este dataset/ángulo → descartado

### 🔧 Configuración acordada con usuario (2026-04-28)
- 1 persona a la vez (bbox mayor área)
- Ventana 60 frames @ 15 fps = 4 segundos
- Stride 30 frames (50% solapamiento)
- Trigger: bbox válido ≥15 frames antes de acumular
- Reset: bbox ausente >15 frames
- Log CSV con timestamp + identidad + confianza

---

## ¿Qué falta?

### **Fase 6.5 — Validación End-to-End** (PENDIENTE)
Antes de despliegue final, falta hacer:

1. **Pruebas formales en vivo** — múltiples sesiones, múltiples sujetos
   - Registrar TPR (% identificados correctamente)
   - Registrar FPR (% falsos positivos)
   - Registrar latencia por frame y por identidad
   - Variabilidad según ropa, iluminación, velocidad de marcha

2. **Validación de robustez**
   - ¿Qué pasa si el sujeto cambia velocidad?
   - ¿Qué pasa con oclusión parcial?
   - ¿Qué pasa si dos personas pasan simultáneamente?
   - ¿Qué pasa con cambio de ropa (esperado en producción)?

3. **Métricas de desempeño en tiempo real**
   - Tiempo por frame (ms)
   - Tiempo por identificación (ms)
   - Uso de GPU/CPU
   - Estabilidad durante sesión larga

### **Fase 7 — Optimización y Despliegue** (SIGUIENTE)
Si Fase 6.5 valida correctamente:

1. **Optimización de velocidad** (si latencia > umbral aceptable)
   - Quantización INT8 (PyTorch → TensorRT)
   - ONNX export + optimización
   - Perfilado bottleneck (¿dónde se gasta más tiempo?)

2. **Robustez en producción**
   - Manejo de errores (GPU crash, desconexión C920, etc.)
   - Log persistente (SQLite en lugar de CSV)
   - UI mejorada (dashboard de estadísticas, alertas)

3. **Documentación final**
   - Manual de instalación
   - Guía de calibración de τ
   - Runbook de troubleshooting

---

## ¿Qué sigue?

**Opción A (recomendado):**  
Continúa a **Fase 6.5 — Validación End-to-End**
- Ejecuta infer_live.py con múltiples sujetos y sesiones reales
- Registra TPR/FPR/latencia
- Documenta hallazgos en `reports/15_validation_endtoend.md`
- Cuando esté validado, avanza a Fase 7

**Opción B (si hay problemas con inferencia en vivo):**
- Describe qué no funciona bien (falsos positivos, latencia, inestabilidad)
- Diagnosticamos y ajustamos infer_live.py antes de validar

**¿Cuál prefieres?** Y si es Opción A, ¿cuándo puedes hacer las pruebas formales?

---

**Empieza próxima sesión con:** "Listo. Estoy en Fase 6.5. Resumen: [hallazgos de validación]. ¿Continuamos a Fase 7 o ajustamos algo?"
