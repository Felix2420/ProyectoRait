# Instrucciones para Claude — entrenamiento Ruta A en RTX 2060

> **Para Claude que arranca en la PC con RTX 2060.**
> Este archivo es tu fuente de verdad para esta sesión. Léelo entero antes de
> hacer nada. No asumas contexto previo: aquí está todo lo que necesitas.

---

## 0. Contexto del proyecto en una pantalla

- **Proyecto:** sistema de pase de lista por reconocimiento de marcha (gait
  recognition). 19 sujetos. Repositorio: `C:\Proyecto3\ProyectoChino\`.
- **Roadmap:** `CLAUDE.md` del repo (lee §6 si necesitas el contexto de fases).
- **Estado al llegar tú:** Fase 3.5 cerrada en producción
  (`gaitbase_ft_multisession_best_iter1200.pt`, EER 8.11 %, TAR@FAR=0% 86.49 %).
  Fase 4 (multimodal silueta+keypoints) abierta y por probar Ruta A.
- **Idioma con el usuario:** español. Estilo: técnico, directo, con justificación.
- **Hardware aquí:** RTX 2060 (6 GB Turing). Suficiente para SkeletonGait++
  (pico esperado ~2-3 GB VRAM).
- **Hardware del usuario por defecto:** GTX 1650 4 GB. Esta 2060 se usa SOLO
  para entrenar Ruta A; el resto del trabajo vuelve a la 1650.
- **Restricción dura:** dataset NO se puede re-grabar. Cualquier mejora debe
  venir del modelo/training.

## 1. Lo que ya se decidió (NO discutir, ejecutar)

1. Ruta B (ST-GCN ligero, fusión tardía) **se descartó con datos** — EER 62 %
   solo, fusión degrada para todo α<1.0. Detalles en
   `reports/13_openset_fusion_pose.json`.
2. **Ruta A:** re-entrenar SkeletonGait++ con 3 cambios YA aplicados a
   `scripts/finetune_skeletongaitpp.py`:
   - Init desde `gaitbase_ft_multisession_best_iter1200.pt` (61 keys, 51.6%
     params transferidos via `init_from_gaitbase()`).
   - Warm-up 200 iter con la rama silueta congelada.
   - `early_stop_patience = 6` y guardia `early_stop_min_iter = 1250`.
3. **No tocar el script** salvo que algo del Paso 2 falle. Si el sanity y la
   verificación de mapping pasan (Paso 1.b–1.c abajo), arrancar entrenamiento
   tal cual.

## 2. Tu plan de ejecución en esta sesión

### Paso 1 — sanity (10 min, obligatorio)

Ejecuta los 3 comandos en orden. Si alguno falla, **detén** y reporta al
usuario antes de seguir.

```bat
cd C:\Proyecto3\ProyectoChino
C:\Proyecto3\venv\Scripts\activate

:: 1.a Torch ve la GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
:: ESPERADO: 2.5.1+cu121 True NVIDIA GeForce RTX 2060
:: SI FALLA: torch no encuentra CUDA. Avisa al usuario. NO entrenes.

:: 1.b Reproducir baseline EER 8.11%
python scripts\openset_eval.py ^
  --checkpoint checkpoints\finetune\gaitbase_ft_multisession_best_iter1200.pt ^
  --tag sanity_2060 --class-num 13
:: ESPERADO en stdout:  ALL  EER= 8.11%  TAR@FAR<=1%=86.49%
:: SI DA OTRO NUMERO: ambiente roto. Avisa al usuario. NO entrenes.

:: 1.c Verificar mapping de pesos GaitBase -> SGPP
python scripts\_verify_init_transfer.py
:: ESPERADO:
::   transferidos = 61 keys
::   params transferidos = 6,893,898 (51.6%)
::   [load] strict=True OK
:: SI DA OTRO NUMERO: mapping roto. Avisa al usuario. NO entrenes.
```

### Paso 2 — entrenar (1.5–2.5 h)

```bat
python -u scripts\finetune_skeletongaitpp.py 2>&1 | tee reports\09_train_routeA_log.txt
```

Lánzalo en background con `run_in_background=true` y arma un Monitor con filtro
`grep -E "init|warmup|eval|it [0-9]|done|Error|Traceback|RuntimeError|FAILED|ckpt"`
para que el usuario vea el progreso sin leer todo el log.

**Hitos esperados que debes ver en orden:**

```
[init] device=cuda
[init] GPU=NVIDIA GeForce RTX 2060  VRAM=6144 MB
[init] SkeletonGait++ params=13,356,951
[init] desde gaitbase_ft_multisession_best_iter1200.pt: 61 keys (51.6% de params)
[warmup] congelados N tensores heredados de GaitBase durante 200 iter
[eval-pre] ...
[it    50/1500] tri=... ce=... active=... ce_acc=...
[it   100/1500] ...
[eval@100] NR rank1=...% margen=+...
...
[warmup-end @ 201] descongelados N tensores; optimizer rebuildeado
...
[it   750/1500]  <-- LR cae a 0.001 (primer milestone)
...
[it  1250/1500] <-- LR cae a 0.0001 (segundo milestone)
...
[ckpt] mejor modelo guardado en checkpoints/finetune/skeletongaitpp_best_iter<N>.pt
```

**Síntomas que requieren parar el entrenamiento (TaskStop) y avisar al usuario:**

| Síntoma | Causa probable | Acción |
|---|---|---|
| `tri = 0.0000` constante por > 100 iter consecutivas | triplet colapsado (todos los pares activos = 0) | parar, no insistir |
| OOM CUDA out of memory | batch demasiado grande | parar, sugerir editar `Config.batch_p = 3` y reanudar |
| val rank1 NR < 30 % @ iter 500 | mapping de pesos no aporta | parar, revisar `_verify_init_transfer.py` de nuevo |
| log NO muestra `[warmup-end @ 201]` | warm-up no se ejecutó (cambios al script perdidos) | parar, leer `scripts/finetune_skeletongaitpp.py` y revisar |
| Crash con traceback | bug | parar, copiar traceback al usuario |

**No interpretes ruido normal como problema:**
- val rank1 fluctuando ±10 % entre evals consecutivos es esperado (val tiene
  sólo 3 sujetos × 4 secs = 12 muestras).
- `tri` puede subir temporalmente tras un milestone — eso es normal.
- VRAM oscilando entre 1.5–3 GB es normal.

### Paso 3 — eval del checkpoint (10 min)

Cuando termine (o cuando reporte `[ckpt] mejor modelo guardado en …`), anota
el `<N>` del iter ganador y ejecuta:

```bat
:: 3.a Eval SGPP solo
python scripts\openset_eval_skeletongaitpp.py ^
  --checkpoint checkpoints\finetune\skeletongaitpp_best_iter<N>.pt ^
  --tag skeletongaitpp_routeA

:: 3.b Editar scripts\openset_eval_fusion.py linea ~40:
::     CKPT_SGPP = REPO / "checkpoints" / "finetune" / "skeletongaitpp_best_iter<N>.pt"
::     (cambia "iter800" por "iter<N>")
:: USA EL TOOL EDIT, no Bash.

:: 3.c Fusion tardia GaitBase + SGPP nuevo
python scripts\openset_eval_fusion.py
```

### Paso 4 — interpretar y reportar al usuario

Lee `reports/11_openset_fusion.json` (sale también en stdout). Identifica α
óptimo y métricas asociadas. Clasifica el resultado:

| Caso | Condición | Reporte al usuario |
|---|---|---|
| **A. Éxito** | α* ∈ {0.4–0.7} y EER_ALL ≤ 6 % y TAR@FAR=0%_ALL ≥ 90 % | "Fase 4 cerrada. EER bajó de 8.11% a X%. Producción puede actualizarse a SGPP+fusión con τ=Y. Caso `ricardomora` resuelto: Sí/No." |
| **B. Marginal** | α* ∈ {0.4–0.7} y EER_ALL ∈ (6, 8] % | "Mejora marginal: EER X% (vs 8.11%). La fusión funciona pero el techo de datos limita. Decide si vale la pena adoptar (más cómputo en inferencia)." |
| **C. Sin mejora** | α* = 1.0 (sólo silueta) o EER_ALL > 8 % | "Ruta A no superó Fase 3.5. El techo es de datos (ángulo lateral 90° + 19 sujetos), no de método. Recomendación: aceptar Fase 3.5 como producción final y arrancar Fase 6 (pipeline en vivo) con ese checkpoint." |

### Paso 5 — preparar archivos de regreso

Genera/asegura que existan estos 6 archivos para que el usuario los copie de
vuelta a la GTX 1650:

```
checkpoints/finetune/skeletongaitpp_best_iter<N>.pt
reports/09_finetune_skeletongaitpp_results.json   (lo crea el train script)
reports/09_train_routeA_log.txt                   (del tee)
reports/10_openset_skeletongaitpp_routeA.json     (lo crea openset_eval_skeletongaitpp.py)
reports/10_openset_skeletongaitpp_routeA_pairs.csv
reports/11_openset_fusion.json                    (lo crea openset_eval_fusion.py)
```

Adicionalmente, **escribe un handoff de cierre** en
`reports/PHASE4_HANDOFF_RESULTADO_2060_<fecha>.md` con:
- Tiempo total del entrenamiento.
- Iter ganador y val_margin_NR.
- Tabla del barrido α (de `11_openset_fusion.json`).
- Caso (A/B/C) y recomendación.
- Casos top-5 de errores (genuinos sim baja, impostores sim alta) — ya los
  imprime `openset_eval_fusion.py`, copialos al handoff.

Indica al usuario los 6 archivos a copiar + el handoff (7 archivos en total).

## 3. Reglas de conducta para esta sesión

1. **Reporta hitos al usuario** entre pasos 1, 2, 3, 4. No le mandes silencio
   de 2 horas: arma un Monitor del entrenamiento y deja que las notificaciones
   le indiquen el progreso.
2. **No modifiques `scripts/finetune_skeletongaitpp.py`** salvo que el usuario
   te lo pida explícitamente o el sanity falle por algo del script.
3. **No toques los checkpoints existentes** — `gaitbase_ft_multisession_best_iter1200.pt`
   debe seguir intacto. Producción Fase 3.5 es el fallback si Ruta A falla.
4. **Si algo en el Paso 1 falla, para.** No improvises soluciones. Reporta al
   usuario y espera instrucciones.
5. **No vuelvas a discutir si Ruta A vale la pena** — ya se decidió. Tu rol
   aquí es ejecutar y reportar resultados, no re-debatir el plan.
6. **Estilo de comunicación:** español, conciso, con justificación técnica
   cuando reportes resultados. Sin relleno.

## 4. Si necesitas más contexto del proyecto

Lee en este orden:
1. `CLAUDE.md` — roadmap general.
2. `reports/PHASE4_HANDOFF_2026-04-27.md` — qué se hizo el día 27 y por qué.
3. `reports/PHASE4_HANDOFF_RUTA_A_2026-04-27.md` — diagnóstico de Ruta A.
4. `reports/PHASE4_HANDOFF_EXPORT_2060_2026-04-27.md` — handoff específico de
   esta sesión (cubre lo mismo que este archivo pero más extenso).
5. `reports/13_openset_fusion_pose.json` — datos de Ruta B descartada.

## 5. Frase con la que arrancas tu primera respuesta al usuario

> "Listo. Estoy en la PC con RTX 2060. Leí `CLAUDE_INSTRUCTIONS_2060.md`.
> Voy a empezar por el Paso 1 (sanity). Te aviso cuando arranque el
> entrenamiento."

No hagas más preguntas previas. Si el sanity pasa, arranca el entrenamiento
sin esperar confirmación.
