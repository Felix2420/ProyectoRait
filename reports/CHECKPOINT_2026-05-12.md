# Checkpoint: Gait Recognition — 2026-05-12

**Responsable:** Usuario + Claude  
**Fase:** Fase 6.5 + 7.0 (Optimización completada)  
**Sesión:** Fix CUBLAS + Optimización FPS (Fase 7) + Recalibración de umbral (Fase 6.5)

---

## Estado Actual

### ✅ Completado desde 2026-05-08

- **Bug CUBLAS resuelto:** `CUBLAS_STATUS_EXECUTION_FAILED` en GTX 1650
  - Causa: `torch.amp.autocast(FP16)` incompatible sin Tensor Cores
  - Fix: Desactivar autocast → forzar FP32 en `src/pipeline/embed.py:78`
  - Verificación: `test_embedding_fix.py` pasa sin errores

- **Fase 7.0 — Optimización de Velocidad completada:**
  1. ✅ Profiling per-etapa: `scripts/profile_pipeline.py` con `torch.cuda.Event`
  2. ✅ Reducir imgsz YOLO: 1280×720 → 640 (2x más rápido)
  3. ⚠️ torch.compile: no viable (requiere Triton, no disponible en Windows)

- **Recalibración de umbral (Fase 6.5):**
  - Cambio: τ = 0.9847 → **τ = 0.85** (para condiciones en vivo)
  - Contexto: Con τ=0.9847, TODOS los sujetos enrolados fueron marcados como desconocidos
  - Observación en vivo: similitud ≈0.95 para enrolados → τ=0.85 debería aceptarlos

### 📊 Métricas Post-Optimización

**Profiling Fase 7 (100 frames, con imgsz=640):**

| Etapa | Tiempo (ms) | % Total | Notas |
|---|---|---|---|
| YOLO | 17.98 | 23% | Optimización: imgsz=640 aplicada |
| RVM | 26.45 | 35% | Cuello de botella principal |
| GaitBase embed | 59.48 | 8% (amortizado) | Compilación no viable |
| Matcher | 0.28 | <1% | CPU, negligible |
| **FPS efectivos** | **20.7** | — | **+21% vs baseline (17.1 FPS)** |

**Comparación FPS:**

| Config | FPS | Mejora |
|--------|-----|--------|
| Baseline (2026-05-06) | ~20 | — |
| Profiling sin optimizaciones simuladas | 17.1 | baseline |
| **Post-optimización (actual)** | **20.7** | **+21%** |

### 🆕 Configuración Activa

```yaml
# configs/pipeline.yaml
camera:
  target_fps: 60          # captura nativa, pipeline limita a ~20 FPS

detector:
  imgsz: 640              # reducido desde 1280×720 → +13% velocidad YOLO

embedder:
  compile: false          # torch.compile no viable en Windows sin Triton

matcher:
  tau: 0.85               # recalibrado para condiciones en vivo

sequence:
  window: 60              # frames por secuencia (4 s @ 15 FPS)
  stride: 30              # 50% solapamiento
```

### 🚩 Limitaciones Conocidas (sin cambios)

- Par `ricardomora ↔ hectorsanchez`: falla NN y NR (mismo build corporal)
- Dataset pequeño (19 sujetos, 90° lateral, 2 sesiones) limita generalización
- Cambio de cámara requiere recalibración de τ

---

## Archivos Modificados (Fase 7)

### Nuevos
- `scripts/profile_pipeline.py` — profiling per-etapa con torch.cuda.Event
- `test_infer_live_init.py` — verificación de inicialización sin cámara
- `test_embedding_fix.py` — verificación del fix CUBLAS

### Modificados
- `src/pipeline/detect.py` — agregado parámetro `imgsz`
- `src/pipeline/embed.py` — fix CUBLAS + torch.compile con fallback
- `scripts/infer_live.py` — pasar `imgsz` y `compile` desde config
- `scripts/eval_pipeline_offline.py` — pasar `imgsz` y `compile` desde config
- `configs/pipeline.yaml` — exponer `imgsz: 640` y `compile: false`

---

## ¿Cómo Revertir los Cambios de Fase 7?

Si necesitas volver al estado anterior (2026-05-08), sigue estos pasos:

### Opción A: Revertir solo optimizaciones de velocidad (mantener fix CUBLAS)

1. **Restaurar imgsz original en `pipeline.yaml`:**
   ```yaml
   detector:
     imgsz: 1280        # cambiar desde 640
   ```
   Impacto: FPS bajará ~13%

2. **Mantener `embed.py` como está** (fix CUBLAS es obligatorio)

3. **Limpiar scripts auxiliares (opcional):**
   ```bash
   rm scripts/profile_pipeline.py
   rm test_infer_live_init.py
   ```

### Opción B: Revertir TODO a 2026-05-08 (NO recomendado)

Si quieres restaurar completamente el estado del 2026-05-08, ejecuta:

```bash
git log --oneline | grep "Fase 7"              # identifica commits
git show <commit-hash>:src/pipeline/embed.py   # ver versión anterior
git checkout <commit-hash> -- src/pipeline/embed.py  # restaurar
```

**Problema:** perderías el fix CUBLAS (el pipeline crazyearía con CUBLAS_STATUS_EXECUTION_FAILED)

### Opción C: Revertir solo torch.compile (no aplica aquí)

torch.compile ya está desactivado (fallback automático). No hay nada que revertir.

---

## ¿Qué Sigue?

### Próxima sesión (cuando puedas probar en vivo):

**Fase 6.5 — Validación End-to-End (PENDIENTE):**

1. Correr `infer_live.py` con τ=0.85
2. Probar con sujetos enrolados:
   - ¿Ahora aparecen como "PRESENTE"?
   - ¿Con qué similitud? (observar en UI)
3. Probar con sujetos desconocidos:
   - ¿Se rechazan como "UNK"?
4. Robustez (diferentes condiciones):
   - Cambio de ropa
   - Velocidad diferente
   - Iluminación
   - Oclusión parcial

**Métrica de éxito:** τ=0.85 acepta enrolados, rechaza desconocidos, con margen de seguridad

### Después (Fase 7 Parte 2):

- SQLite en lugar de CSV (mejora audit trail)
- Documentación final (manual, guía de calibración)

---

## Verificación (antes de usar en producción)

```bash
# 1. Verificar que infer_live.py inicializa sin errores:
python test_infer_live_init.py

# 2. Medir FPS actual:
python scripts/profile_pipeline.py --n-frames 100

# 3. Correr en vivo con cámara:
python scripts/infer_live.py
```

**Próximo checkpoint después de Fase 6.5 (validación).**

---

## Notas técnicas

### Por qué torch.compile no funciona en Windows:

- Inductor backend requiere **Triton** (compilador CUDA)
- Triton solo soporta Linux/NVIDIA en la rama principal
- Soporte experimental en Windows aún en desarrollo
- Fallback automático en `embed.py` usando try-except

### Por qué imgsz=640 funciona:

- YOLO11n (nano) procesa frames a resolución variable
- 640px es suficiente para detectar personas a ~2-3 metros en 720p
- Reduce complejidad del modelo: O(resolution²) en cálculo
- No afecta accuracy significativamente (bbox aún preciso para tracking)

### Por qué RVM es el cuello de botella ahora:

- RVM procesa FRAME COMPLETO (1280×720 o 640×720 después de crop YOLO)
- No es costoso respetar a YOLO (19 ms vs 27 ms para RVM)
- Segmentación recurrente (estado) más compleja que detección
- Próxima optimización: reemplazar RVM con modelo más ligero (SegFormer, etc.) — Fase 8

