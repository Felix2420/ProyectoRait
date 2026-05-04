# Checkpoint Fase 4 — handoff Ruta A (post Ruta B fallida) — 2026-04-27

> Snapshot operativo para retomar Fase 4 en la PC de entrenamiento.
> Ruta B descartada con datos. Próxima acción: Ruta A en otra PC.

---

## 1. Estado actual

- **Producción sigue en Fase 3.5:** `gaitbase_ft_multisession_best_iter1200.pt`,
  τ = 0.9807, EER 8.11 %, TAR@FAR=0% 86.49 %.
- **Fase 4 abierta** — requisito multimodal (siluetas + keypoints) sigue sin cumplirse.
- Probadas y descartadas:
  - SkeletonGait++ `iter800` (random init) → EER 18.92 % solo, fusión no aporta.
  - ST-GCN ligero (Ruta B) → EER 62 % solo, fusión degrada para todo α<1.

## 2. Por qué Ruta B falló (resultado, no especulación)

`reports/13_openset_fusion_pose.json` (barrido α GaitBase + ST-GCN lite):

| α (GaitBase) | EER_ALL | TAR@FAR=0%_ALL |
|---|---|---|
| 0.0 (solo pose)    | 62.16 % | 10.8 % |
| 0.5                 | 54.05 % | 21.6 % |
| 0.9                 | 24.32 % | 54.0 % |
| 1.0 (solo silueta)  |  8.11 % | 86.5 % |

- Val rank-1 fue 83 % en 13 sujetos cerrados → el modelo aprende, pero **no generaliza** a sujetos nuevos. Overfit con poca data + ángulo 90° donde el esqueleto contiene poca info identitaria.
- Caso `ricardomora → hectorsanchez` sigue siendo el error duro en todo α.

Conclusión técnica: **fusionar tardíamente un modelo solo-pose entrenado from-scratch con
13 sujetos no levanta el techo del 8.11 %** que ya alcanza GaitBase solo. Hay que
fusionar **a nivel de features dentro del backbone** (lo que SkeletonGait++ hace), pero
entrenarlo con init correcto.

## 3. Lo que se hizo hoy

1. Auditoría de keypoints: 75 secs, 0 NaN, conf media 0.77, T∈[73,129]. Limpios.
2. Implementado **ST-GCN ligero** (`src/models/stgcn_lite.py`, 0.26 M params).
3. Implementado **train_stgcn.py** (triplet BatchAll + 0.1·CE, P×K=4×4, win T=30,
   augmentation: flip LR, jitter σ=0.01, low-conf dropout).
4. Implementado **openset_eval_fusion_pose.py** (barrido α 0..1).
5. Entrenado: best val rank-1 83.33 % @ iter 200, early-stop iter 450.
6. Evaluado: ver tabla §2. Ruta B descartada con datos.

## 4. Archivos generados (ya en este repo, viajan a la otra PC)

- `src/models/stgcn_lite.py` *(modelo ligero, conservar — útil de referencia)*
- `scripts/train_stgcn.py` *(no relevante para Ruta A)*
- `scripts/openset_eval_fusion_pose.py` *(no relevante para Ruta A)*
- `checkpoints/finetune/stgcn_lite_best.pt` *(no se usa en Ruta A)*
- `reports/12_train_stgcn_lite.json`
- `reports/13_openset_fusion_pose.json`
- `reports/PHASE4_HANDOFF_RUTA_A_2026-04-27.md` *(este archivo)*

## 5. Ruta A — re-entrenar SkeletonGait++ con init correcto

### 5.1 Diagnóstico previo (recordatorio del handoff anterior)

`scripts/finetune_skeletongaitpp.py` línea 388 imprime
`"sin checkpoint previo, random init"`. El plan original §7.3 prescribía inicializar
la rama silueta desde `gaitbase_ft_multisession_best_iter1200.pt`. Eso **no se aplicó**.
Resultado: dual-stream 13.3 M params + batch P=4 K=2 + random init = triplet colapsa.

### 5.2 Cambios concretos a hacer en `scripts/finetune_skeletongaitpp.py`

**Tres modificaciones obligatorias.** No hagas más.

1. **Cargar pesos compatibles desde GaitBase iter1200** *antes* del bucle de entrenamiento:

   ```python
   # justo después de instanciar el modelo SGPP, antes del optimizer:
   GB_CKPT = REPO / "checkpoints" / "finetune" / "gaitbase_ft_multisession_best_iter1200.pt"
   gb_state = torch.load(GB_CKPT, map_location=device, weights_only=False)
   gb_state = gb_state.get("model", gb_state)

   sgpp_state = model.state_dict()
   transferred, skipped = [], []
   for k, v in gb_state.items():
       # SGPP usa nombres como "Backbone.0.X" si está envuelto. Mapear según corresponda
       # leyendo ambas state_dicts en debug primero. La regla general:
       #   - FCs.*  -> FCs.*           (igual)
       #   - BNNecks.* -> BNNecks.*    (igual)
       #   - Backbone.layer*.* (resnet9 interno post-stem)
       # La PRIMERA conv (in_channels 1 -> 3) NO se transfiere: queda random.
       sk = k  # ajustar mapping si los prefijos difieren
       if sk in sgpp_state and sgpp_state[sk].shape == v.shape:
           sgpp_state[sk] = v
           transferred.append(sk)
       else:
           skipped.append(k)
   model.load_state_dict(sgpp_state, strict=False)
   print(f"[init] transferidos={len(transferred)}  skipped={len(skipped)}")
   print(f"[init] ejemplos skipped (primeros 5): {skipped[:5]}")
   ```

   **ANTES de tocar código**: imprimir `model.state_dict().keys()` de SGPP y de GaitBase
   y verificar mapping. La primera conv del Backbone de SGPP es `in_channels=3`
   (heatmap×2 + sil×1) vs `in_channels=1` de GaitBase: esa **NO** se transfiere.

2. **Warm-up 200 iter con rama silueta congelada** para que la rama pose aprenda primero.
   Implementar gating con `requires_grad`:

   ```python
   WARMUP_ITERS = 200

   def freeze_silhouette_branch(model, freeze: bool):
       """Congela parámetros que vienen del init de GaitBase (rama silueta)."""
       for name, p in model.named_parameters():
           # Heurística: lo que se transfirió antes (los nombres en `transferred`).
           # Hacer un set explicito; pasarlo como argumento o cerrar sobre él.
           if name in transferred:
               p.requires_grad = not freeze

   freeze_silhouette_branch(model, freeze=True)
   for it in range(1, total_iter + 1):
       if it == WARMUP_ITERS + 1:
           freeze_silhouette_branch(model, freeze=False)
           print(f"[warmup-end] descongelando rama silueta @ iter {it}")
       ...
   ```

   El optimizer debe instanciarse después del freeze inicial o usar
   `params=filter(lambda p: p.requires_grad, model.parameters())` y rebuildearlo en
   iter 201.

3. **Schedule completo 1500 iter** sin tocar milestones (750, 1250) ni patience.
   Eliminar el early-stop temprano que cortó iter 800. Si quieres mantener early-stop,
   subir patience a 6 evals y NO permitir cortar antes del segundo milestone (iter 1250).

   ```python
   # cambia patience = 3 -> patience = 6
   # añade guardia:
   if no_improve >= PATIENCE and it > 1250:
       break
   ```

### 5.3 Comandos a correr en la otra PC

```bash
cd C:\Proyecto3\ProyectoChino  # o equivalente

# 0) sanity: GPU y deps
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1) ANTES de entrenar: imprimir keys para verificar mapping de pesos
python -c "
import torch, sys
from pathlib import Path
sys.path.insert(0, 'third_party/OpenGait/opengait')
sys.path.insert(0, 'third_party/OpenGait')
gb = torch.load('checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt',
                map_location='cpu', weights_only=False)
gb_state = gb.get('model', gb)
print('=== GaitBase keys (primeras 30) ===')
for k in list(gb_state.keys())[:30]: print(k, gb_state[k].shape)
"

# 2) entrenar con los cambios aplicados
python scripts/finetune_skeletongaitpp.py  # ajustar args si los hay

# 3) eval (mismo script de antes, sólo cambia el checkpoint)
python scripts/openset_eval_skeletongaitpp.py \
  --checkpoint checkpoints/finetune/skeletongaitpp_best_iter<NUEVO>.pt \
  --tag skeletongaitpp_routeA

# 4) fusión tardía (mismo script de antes)
python scripts/openset_eval_fusion.py
```

### 5.4 Tiempo estimado

- GTX 1650: 4–8 h.
- GPU mejor (RTX 30xx/40xx): 1–3 h.

### 5.5 Criterio de éxito Ruta A

- **OK:** EER_ALL ≤ 6 % **y** TAR@FAR=0%_ALL ≥ 90 % **y** caso
  `ricardomora → hectorsanchez` resuelto.
- **Si EER_ALL > 8 % con SGPP solo** (peor que Fase 3.5): probable colapso de la rama
  pose o mapping de pesos incorrecto. **Detenerse y revisar**, no insistir en más
  iteraciones.

### 5.6 Riesgos a vigilar durante el train

| Síntoma | Causa probable | Acción |
|---|---|---|
| triplet loss colapsa a 0 sin EER bajo | overfit batch + identidades pocas | bajar K a 2, subir margin a 0.3 |
| val rank-1 atascado en ~5 % | mapping de pesos rompe el backbone | imprimir `transferred` y revisar shapes |
| `loss_ce` crece tras iter 200 | rama silueta se descongeló mal | verificar que requires_grad volvió a True |
| OOM en GTX 1650 | batch demasiado grande | bajar P a 3 (no menos) |

## 6. Próximo arranque de sesión — qué decir

> "Estoy en Fase 4, Ruta A. Leí `reports/PHASE4_HANDOFF_RUTA_A_2026-04-27.md`.
> Voy a aplicar las 3 modificaciones del §5.2 a `scripts/finetune_skeletongaitpp.py`
> y entrenar."

Si el resultado de Ruta A falla (EER_ALL > 8 % o no se resuelve `ricardomora`),
escalar a:
- Cambiar extractor de pose a RTMPose-l (mayor confianza articular).
- Re-grabar dataset con ángulos no laterales (45°) — la pose lateral 90° tiene techo
  natural.
