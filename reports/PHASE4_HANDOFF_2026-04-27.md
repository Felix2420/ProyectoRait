# Checkpoint Fase 4 — handoff 2026-04-27

> Estado al final de la sesión del 27-abr para retomar después sin perder contexto.
> No es un reporte ejecutado todavía; es un snapshot operativo.

---

## 1. Requisito duro confirmado por el usuario

El proyecto **debe** usar los dos flujos en inferencia: **siluetas + keypoints**.
No es opcional. Cualquier ruta que cierre Fase 4 sin un canal de pose funcional
queda descartada.

## 2. Lo que se hizo hoy

1. Instalación de los datos multimodales que faltaban en el repo:
   - `data/heatmaps/` (75 secs)
   - `data/pkl_multimodal/` (150 archivos = 75 × `0_heatmap.pkl` + 75 × `1_sil.pkl`)
   - `data/poses_raw/` (75 secs)
   - Origen: `C:\Users\Jesus\Downloads\data\` (entrenado en otra PC).
   - `data/pkl/` no se tocó (md5 idénticos al repo).
2. `pip install imageio` en `C:\Proyecto3\venv\` (lo pedía el `__init__.py` de
   OpenGait al cargar `BigGait.py`).
3. Smoke test multimodal verde: `scripts/smoke_test_skeletongaitpp.py` →
   forward OK, embedding `(1, 256, 16)`, VRAM pico 635 MB en GTX 1650.
4. **Eval open-set SGPP solo** (`scripts/openset_eval_skeletongaitpp.py`):
   - Salidas: `reports/10_openset_skeletongaitpp_iter800.json` + `_pairs.csv`.
   - Resultado: **EER ALL = 18.92 %, TAR@FAR=0% = 54.0 %, gap +0.003**.
   - Veredicto: embedding casi degenerado (rama de pose colapsada).
5. **Eval fusión tardía** (`scripts/openset_eval_fusion.py`, barrido α∈[0,1]):
   - Salida: `reports/11_openset_fusion.json`.
   - Mejor fusión: α=0.8 (peso GaitBase) → EER 8.11 %, TAR@FAR=0% 86.5 %.
     Idéntico al baseline solo-silueta de Fase 5. La rama de pose **no aporta**.
   - `ricardomora → hectorsanchez` sigue mal en cualquier α.

## 3. Diagnóstico técnico de por qué `iter800` está mal

Mirando `scripts/finetune_skeletongaitpp.py`:

- Línea 388 imprime `"sin checkpoint previo, random init"`. **El plan §7.3
  prescribía inicializar la rama silueta desde
  `gaitbase_ft_multisession_best_iter1200.pt`; eso no se aplicó.**
- Checkpoint guardado:
  `iter=800, total_iter=1500, milestones=(750, 1250), patience=3`.
  Early-stop disparó tras el primer milestone — el modelo nunca llegó al
  refinamiento del segundo milestone.
- Causa raíz: dual-stream 13.3 M params + batch P=4 K=2 + random init
  total = triplet colapsa (riesgo "Alta" del plan §8).

## 4. Estado operativo

- Modelo en producción sigue siendo **Fase 3.5**:
  `checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt`,
  τ = 0.9807, EER 8.11 %, TAR@FAR=0% 86.49 %.
- Fase 4 **no cerrada** (requisito multimodal duro).
- Reportes 09 y 10 (md ejecutados) **pendientes**: solo existen los JSON/CSV.

## 5. Dos rutas para retomar — decisión pendiente del usuario

### Ruta A — Re-entrenar SkeletonGait++ con init correcto

- Modificar `scripts/finetune_skeletongaitpp.py` para:
  1. Cargar pesos compatibles desde `gaitbase_ft_multisession_best_iter1200.pt`
     al backbone de SGPP (ResNet9 interno post-stem; primera conv random por
     `in_channels` 1 → 3; transferir FCs y BNNecks tal cual).
  2. Warm-up 200 iter con rama silueta congelada para que la rama pose
     aprenda primero.
  3. 1500 iter completas, mismo schedule.
- Tiempo estimado GTX 1650: **4–8 h**.
- Antes de tocar código verificar compatibilidad de pesos
  `gaitbase` ↔ `SGPP.Backbone` (leer ambas clases).

### Ruta B — Modelo solo-pose ligero + late fusion (recomendada)

- Entrenar **GaitGraph2** (o GCN/MLP simple, ~0.3 M params) sobre
  `data/processed/<subj>/<cond>/keypoints.npy` (s1) y
  `data/processed_s2/...` (s2). Ya están extraídos en formato `(T,17,3)`.
- Late fusion silueta-GaitBase + pose-GaitGraph2 con `openset_eval_fusion.py`
  adaptado.
- Tiempo estimado GTX 1650: **30–60 min**.
- Cumple el requisito multimodal en inferencia (dos flujos).
- Si falla → señal de que la pose 90° lateral no aporta y procede A o cambiar
  extractor (RTMPose-l).

**Recomendación al retomar: empezar por B.** Si no resuelve el caso
`ricardomora → hectorsanchez` ni mejora TAR@FAR=0% por encima de 90 %,
escalar a A.

## 6. Comandos exactos para reproducir lo de hoy

```bash
cd C:\Proyecto3\ProyectoChino

# Smoke test (rápido, valida que todo carga)
C:/Proyecto3/venv/Scripts/python.exe scripts/smoke_test_skeletongaitpp.py

# Eval SGPP solo
C:/Proyecto3/venv/Scripts/python.exe scripts/openset_eval_skeletongaitpp.py \
  --checkpoint checkpoints/finetune/skeletongaitpp_best_iter800.pt \
  --tag skeletongaitpp_iter800

# Fusión tardía GaitBase + SGPP
C:/Proyecto3/venv/Scripts/python.exe scripts/openset_eval_fusion.py
```

## 7. Archivos generados/modificados hoy

- **Datos** (movidos desde Downloads):
  - `data/heatmaps/`, `data/pkl_multimodal/`, `data/poses_raw/`
- **Reportes nuevos**:
  - `reports/10_openset_skeletongaitpp_iter800.json` + `_pairs.csv`
  - `reports/11_openset_fusion.json`
  - `reports/PHASE4_HANDOFF_2026-04-27.md` (este archivo)
- **Venv**: `imageio` añadido.
- **Sin cambios**: scripts, configs, splits, checkpoints existentes.

## 8. Próximo arranque de sesión — qué decir

> "Estoy en Fase 4 reabierta. Leí `reports/PHASE4_HANDOFF_2026-04-27.md`.
> Decisión: ruta A / ruta B."
