# Checkpoint Fase 4 — handoff export + entrenamiento RTX 2060 — 2026-04-27

> Snapshot operativo. Cierra esta sesion en la GTX 1650 y abre en la PC con
> RTX 2060 para ejecutar Ruta A. Todo lo necesario esta exportado y los scripts
> ya tienen los 3 cambios aplicados.

---

## 1. Estado al cierre

- **Producción:** Fase 3.5 intacta — `gaitbase_ft_multisession_best_iter1200.pt`,
  τ = 0.9807, EER 8.11 %, TAR@FAR=0% 86.49 %.
- **Fase 4 abierta** (requisito multimodal duro pendiente).
- **Ruta B (ST-GCN ligero) descartada con datos.** Reportes 12, 13.
- **Ruta A pendiente de entrenar en RTX 2060.**

## 2. Lo que se hizo en esta sesión

1. Auditoría keypoints: 75 secs, 0 NaN, conf media 0.77.
2. Implementado ST-GCN ligero (`src/models/stgcn_lite.py`, 0.26 M params,
   val rank-1 83 % @ iter 200). Resultado fusión: EER_ALL 8.11 % en α=1.0
   (idéntico a baseline; pose no aporta). **Ruta B descartada.**
3. **Ruta A preparada:**
   - Auditoría mapping de pesos GaitBase → SGPP en
     `scripts/_verify_init_transfer.py`. Resultado: **61 keys transferibles,
     6.9M / 13.3M params (51.6%)**, sin shape mismatches.
   - **3 cambios aplicados a `scripts/finetune_skeletongaitpp.py`:**
     1. `init_from_gaitbase()` carga `gaitbase_ft_multisession_best_iter1200.pt`
        antes del optimizer (cobertura 51.6%).
     2. Warm-up 200 iter con la rama silueta congelada (gating por
        `requires_grad`); optimizer rebuildeado en iter 201.
     3. `early_stop_patience = 6` y `early_stop_min_iter = 1250` —
        prohibido cortar antes del segundo milestone.
4. Export generado en `C:\Proyecto3\export\` (3 zips, 250 MB total).

## 3. Lo que se exportó (250 MB)

| zip | tamaño | contenido relevante para Ruta A |
|---|---|---|
| `ProyectoChino_code.zip` | 33.2 MB | scripts (con `finetune_skeletongaitpp.py` modificado), `_verify_init_transfer.py`, configs, third_party/OpenGait, requirements.lock.txt (86 paquetes), todos los reportes y handoffs |
| `ProyectoChino_checkpoints.zip` | 196.5 MB | `GaitBase_Gait3D_120000.pt` (pretrained), **`gaitbase_ft_multisession_best_iter1200.pt`** (init de Ruta A), `gaitbase_ft_best_iter400.pt` (fallback) |
| `ProyectoChino_pkls.zip` | 20.3 MB | `pkl/`, `pkl_s2/`, `pkl_multisession/`, **`pkl_multimodal/`** (los datos que entrena SGPP) |

**Lo que NO va** (no se necesita para Ruta A): `data_raw/` (videos crudos),
`data/processed/` y `data/processed_s2/` (intermedios), checkpoints de ST-GCN
y SGPP iter800 (no útiles para Ruta A).

## 4. Setup en la PC con RTX 2060 (asumiendo libs ya instaladas)

Si el venv `C:\Proyecto3\venv\` ya existe con torch+CUDA+OpenGait dependencies:

```bat
cd C:\Proyecto3
tar -xf ProyectoChino_code.zip         -C C:\Proyecto3\
tar -xf ProyectoChino_checkpoints.zip  -C C:\Proyecto3\
tar -xf ProyectoChino_pkls.zip         -C C:\Proyecto3\

C:\Proyecto3\venv\Scripts\activate
cd C:\Proyecto3\ProyectoChino
```

Si **no** existe venv, ver `SETUP_NEW_PC.md` §2–§4 (Python 3.11.9, CUDA 12.1,
`pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`,
luego `pip install -r requirements.lock.txt`).

## 5. Sanity (10 min — obligatorio antes de entrenar)

```bat
:: 5.1 Torch ve la GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
:: esperado: 2.5.1+cu121 True NVIDIA GeForce RTX 2060

:: 5.2 Reproducir baseline EER 8.11%
python scripts\openset_eval.py ^
  --checkpoint checkpoints\finetune\gaitbase_ft_multisession_best_iter1200.pt ^
  --tag sanity_2060 --class-num 13
:: esperado:  ALL  EER= 8.11%  TAR@FAR<=1%=86.49%

:: 5.3 Verificar mapping de pesos
python scripts\_verify_init_transfer.py
:: esperado:
::   transferidos = 61 keys
::   params transferidos = 6,893,898 (51.6%)
::   [load] strict=True OK
```

**Si 5.2 NO da 8.11%** → ambiente roto, no entrenar.
**Si 5.3 reporta errores** → mapping cambió, no entrenar, avisar.

## 6. Entrenar Ruta A (1.5–2.5 h en RTX 2060)

```bat
python -u scripts\finetune_skeletongaitpp.py 2>&1 | tee reports\09_train_routeA_log.txt
```

**Hitos esperados en el log:**

```
[init] device=cuda
[init] GPU=NVIDIA GeForce RTX 2060  VRAM=6144 MB
[init] SkeletonGait++ params=13,356,951
[init] desde gaitbase_ft_multisession_best_iter1200.pt: 61 keys (51.6% de params)
[warmup] congelados N tensores heredados de GaitBase durante 200 iter
[eval-pre] ...
[it    50/1500] ...
[it   100/1500] ...
[eval@100] NR rank1=...% margen=+...
...
[warmup-end @ 201] descongelados N tensores; optimizer rebuildeado
...
[it   750/1500]  <-- LR cae a 0.001
...
[it  1250/1500] <-- LR cae a 0.0001
...
[ckpt] mejor modelo guardado en checkpoints/finetune/skeletongaitpp_best_iter<N>.pt
[done] tiempo total: ~120 min
```

**Vigila en otra terminal:**
```bat
nvidia-smi -l 5
:: GPU util > 80%, VRAM ~ 2-3 GB
```

**Síntomas → acción:**
| Síntoma | Acción |
|---|---|
| `tri = 0.0000` constante 100 iter | parar, triplet colapsado |
| OOM (VRAM > 5.5 GB) | editar `Config.batch_p = 3` y reanudar |
| val rank1 NR < 30 % @ iter 500 | parar, mapping sospechoso |
| log `[warmup-end @ 201]` no aparece | warm-up no se ejecutó, revisar |

## 7. Evaluar (10 min)

Anota `<N>` del checkpoint guardado (sale al final del log).

```bat
:: 7.1 SGPP solo
python scripts\openset_eval_skeletongaitpp.py ^
  --checkpoint checkpoints\finetune\skeletongaitpp_best_iter<N>.pt ^
  --tag skeletongaitpp_routeA

:: 7.2 Editar scripts\openset_eval_fusion.py linea ~40:
::     CKPT_SGPP = REPO / "checkpoints" / "finetune" / "skeletongaitpp_best_iter<N>.pt"
::     (cambia "iter800" por "iter<N>")

:: 7.3 Fusion tardia
python scripts\openset_eval_fusion.py
```

## 8. Decidir si Fase 4 cierra

Mira la tabla del barrido α en `reports/11_openset_fusion.json` (también stdout):

| Resultado | Significado | Acción |
|---|---|---|
| α* ∈ {0.4–0.7} y EER_ALL ≤ 6 % y TAR@FAR=0%_ALL ≥ 90 % | **Fase 4 cerrada con éxito** | actualizar producción a SGPP, recalibrar τ |
| α* ∈ {0.4–0.7} y EER_ALL ∈ [6, 8] % | **Mejora marginal** | decidir si vale la pena; SGPP requiere más cómputo en inferencia |
| α* = 1.0 (sólo silueta) | **No mejora vs baseline** | techo de datos (no de método). Documentar y aceptar Fase 3.5 como producción final |

## 9. Volver a la GTX 1650 con resultados

Copia de regreso (USB / cloud), 6 archivos:
```
checkpoints/finetune/skeletongaitpp_best_iter<N>.pt
reports/09_finetune_skeletongaitpp_results.json
reports/09_train_routeA_log.txt
reports/10_openset_skeletongaitpp_routeA.json
reports/10_openset_skeletongaitpp_routeA_pairs.csv
reports/11_openset_fusion.json
```

Aquí seguimos análisis y, según resultado §8, arrancamos Fase 6 (pipeline en
vivo) o documentamos cierre definitivo de Fase 4 con producción Fase 3.5.

## 10. Próximo arranque de sesión

> "Estoy en la PC con RTX 2060. Leí `reports/PHASE4_HANDOFF_EXPORT_2060_2026-04-27.md`.
> Ya hice los pasos §4 (descomprimir + venv). Voy a §5 (sanity)."

Si Ruta A falla y el techo es de datos (caso §8 fila 3), el plan ya tiene
considerado: **dataset NO se puede re-grabar** — Fase 3.5 queda como producción
final y arrancamos Fase 6 con ese checkpoint.
