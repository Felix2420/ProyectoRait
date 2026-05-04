# Reporte 07 — Plan de Fase 4: pipeline multimodal (siluetas + keypoints)

> Plan de trabajo, no resultados. Este documento se escribe **antes** de la
> fase para fijar qué modelo, qué datos, qué hparams, qué criterio de éxito.
> Se convertirá en reporte ejecutado una vez que termine la fase.
>
> Generado: 2026-04-22.
> Contexto: Fase 5 cerrada (TAR@FAR=0% = 86.49%, EER = 8.11% con el modelo
> solo-silueta multisession). El usuario confirmó que el proyecto requiere
> **siluetas + keypoints** como multimodal obligatorio. Fase 4, que había
> sido archivada en reporte 06, vuelve a estar activa.

---

## 1. Objetivo

Integrar una rama de keypoints al modelo actual para:

1. Cumplir el requisito funcional del proyecto (multimodal explícito).
2. Mejorar el caso difícil que persiste: `ricardomora → hectorsanchez`
   (ambos tienen build corporal similar en silueta; la pose puede
   desambiguar).
3. No degradar las métricas ya conseguidas.

## 2. Modelo propuesto

**Candidato principal: SkeletonGait++** (Fan et al., 2024, parte del model zoo
de OpenGait).

- Arquitectura dual-stream: una rama procesa siluetas (GaitBase-like) y otra
  procesa keypoints como mapas de calor. Fusión en embedding común.
- Compatible con el pipeline de datos actual (pkl de siluetas + pkl de
  keypoints).
- Checkpoint público disponible en OpenGait si la PC nueva permite reutilizarlo.

**Alternativas si SkeletonGait++ no rinde:**

- **MultimodalGait**: arquitectura similar, menos popular pero más liviana.
- **Late fusion manual**: entrenar modelo de keypoints por separado
  (p.ej. GaitGraph2) y fusionar embeddings por concatenación o weighted sum.
  Más control, más código a mantener.

Prioridad: **SkeletonGait++** por soporte oficial en OpenGait.

## 3. Datos necesarios

### 3.1 Keypoints por secuencia

Ya se definió en Fase 1 el extractor: **RTMPose-m** (17 keypoints COCO,
confidence).

**Estado actual (2026-04-22):**
- `data/poses/` está **vacío**.
- Hay que correr RTMPose sobre todos los videos (s1 + s2) y guardar:
  - Formato por decidir según lo que pida el dataloader de SkeletonGait++.
    Típicamente: `.pkl` con shape `(T, 17, 3)` = `(frames, keypoints, x/y/conf)`.
  - Ruta: `data/poses/<subject>/<cond>_<sesion>/090/seq00.pkl`.

### 3.2 Pkls multimodales

SkeletonGait++ espera **dos streams** en el mismo registro. Dos opciones:

1. **Un solo .pkl con dict `{"sils": ndarray, "pose": ndarray}`**.
2. **Dos .pkl separados** en rutas paralelas; dataloader fusiona por index.

Hay que leer el dataloader de SkeletonGait++ primero y adaptar `pack_to_pkl.py`
en consecuencia.

### 3.3 Check de sanity

Antes de entrenar:
- Verificar que para cada secuencia, **T de siluetas == T de keypoints**.
- Keypoints con confidence > 0.3 en ≥ 80% de los frames, o descartar/rellenar.

## 4. Splits

**Idénticos a Fase 3 y 3.5**: `configs/splits.yaml` (13 train, 3 val, 3 test,
subject-disjoint, seed=42). No se cambia nada para permitir comparación
directa de resultados.

## 5. Hparams de partida

Punto de partida = mismos que Fase 3.5, ajustados si la rama extra cambia
el régimen:

| Parámetro | Valor inicial | Razón |
|---|---|---|
| Batch | P=4 K=2 = 8 | Mismo que 3.5 |
| Optimizer | SGD momentum=0.9 wd=5e-4 | Mismo |
| LR | 0.01 | Mismo |
| Schedule | MultiStep [750, 1250] | Mismo |
| Max iter | 1500 | Mismo |
| Triplet margin | 0.2 | Mismo |
| CE scale / smoothing | 16 / 0.1 | Mismo |
| Frames por clip | 30 | Mismo |
| Loss dual-stream | como venga en SkeletonGait++ | No inventar |
| AMP float16 | sí | GTX 1650 lo necesitaba; en GPU nueva es opcional |
| Patience early stop | 3 sobre margen NR val | Mismo criterio que 3.5 |

**Si la GPU nueva es ≥ 12 GB VRAM**, subir batch a P=8 K=2 = 16 o P=4 K=4 = 16
para mejorar la mineria de triplets (el triplet loss colapsaba a 0 en Fase 3.5
con P=4×K=2 porque había pocos negatives duros).

## 6. Criterio de éxito

Fase 4 se considera cerrada como "éxito" si se cumple **al menos uno** de estos:

### 6.1 Mejora en el caso duro (prioritario)
- `ricardomora` NN y NR clasificados correctamente (top1 = `ricardomora`).
- Actualmente fallan ambos → si esto se resuelve, es una victoria clara de
  la rama de pose.

### 6.2 Mejora en métricas globales (secundario)
- **TAR @ FAR=0%** sube de 86.49% a ≥ **90%** (de 32/37 a 34+/37).
- **EER** baja de 8.11% a ≤ **5%**.
- Rank-1 cross-session NN y NR se mantienen ≥ 94% (no regresar).

### 6.3 No empeorar (bloqueador)
- Rank-1 NN o NR NO debe caer por debajo de 92%.
- EER NO debe subir por encima del 10%.

Si Fase 4 empeora las métricas globales (caso típico: fusión mal calibrada
o keypoints ruidosos), se documenta el intento y se deja el modelo
de Fase 3.5 como operativo.

## 7. Roadmap interno de Fase 4

### 7.1 Paso A — Extracción de keypoints (1–2 días)

- Revisar/extender `scripts/extract_all.py` para que también guarde salida
  de RTMPose en `data/poses/<subject>/<cond>_<sesion>/090/seq00.pkl`.
- Correr sobre los 75 videos (s1 + s2).
- Generar reporte `08_poses_extraction.md` con:
  - Total secuencias con pose exitosa.
  - Distribución de confidence por keypoint.
  - Casos problemáticos (oclusiones, confidence baja).

### 7.2 Paso B — Dataloader multimodal (0.5–1 día)

- Leer el dataloader de SkeletonGait++ en `third_party/OpenGait/`.
- Adaptar `pack_to_pkl.py` si es necesario.
- Correr `scripts/validate_opengait_dataloader.py` en modo multimodal
  (posiblemente extendido) para verificar que lee ambas ramas correctamente.

### 7.3 Paso C — Checkpoint inicial (0.5 día)

- Bajar checkpoint público de SkeletonGait++ (si existe).
- Si no, inicializar rama de silueta con `gaitbase_ft_multisession_best_iter1200.pt`
  y rama de pose desde random init + warm-up de 200 iter.
- Smoke test: `python scripts/smoke_test_skeletongaitpp.py` (a crear).

### 7.4 Paso D — Fine-tune (1 día ejecución + tiempo muerto)

- Script `scripts/finetune_skeletongaitpp.py` basado en
  `finetune_gaitbase_multisession.py`.
- Early stop sobre margen NR val (mismo criterio que 3.5).
- Reporte `09_finetune_skeletongaitpp.md`.

### 7.5 Paso E — Eval completo (0.5 día)

- Cross-session closed-set: `scripts/cross_session_eval.py` extendido para
  multimodal.
- Open-set: `scripts/openset_eval.py` extendido.
- Reporte `10_eval_multimodal.md` con comparativa vs Fase 3.5.

### 7.6 Paso F — Decisión (0.5 día)

- Si cumple criterio § 6 → checkpoint multimodal pasa a ser operativo.
- Si no → documentar, mantener Fase 3.5 como operativo, analizar por qué
  falló (keypoints ruidosos? fusión mal ponderada?).

**Total estimado**: 4–6 días en PC nueva con GPU holgada.

## 8. Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Keypoints de RTMPose malos (oclusión, ángulo 90° lateral difícil) | Media | Validar confidence en Paso A; si <80% de frames con conf>0.3, cambiar a RTMPose-l o usar YOLOv8-pose |
| SkeletonGait++ sin checkpoint público | Baja | Entrenar desde cero con silueta-init del modelo Fase 3.5 |
| Batch chico hace que triplet colapse (como en 3.5) | Alta | Subir P×K si GPU nueva lo permite; fijar early-stop más agresivo |
| Fusión late overfittea a la val (3 sujetos) | Media | Validar en test (también 3 sujetos) y reportar ambos |
| `ricardomora → hectorsanchez` sigue mal | Media | Aceptar que es caso inherente y escalar a Fase 6 sin él |

## 9. Qué pasa si Fase 4 no resuelve `ricardomora`

Plan B: **fusión asimétrica en inferencia**. En vez de un embedding único,
mantener dos embeddings (silueta + pose) y dos similitudes:

```
decision = match(sim_sil) if sim_sil >= tau_sil AND sim_pose >= tau_pose
         = unknown otherwise
```

Con dos τ separados, el doble filtro sube la confianza. Cuesta ~0.5 día
adicional sobre el pipeline multimodal ya entrenado.

## 10. Entregables esperados al cerrar Fase 4

- `scripts/finetune_skeletongaitpp.py`
- `scripts/pack_pose_to_pkl.py` (si se separan pkls)
- `data/poses/` poblado con 75 secuencias
- `checkpoints/finetune/skeletongaitpp_ft_multisession_best_iter<N>.pt`
- `reports/08_poses_extraction.md`
- `reports/09_finetune_skeletongaitpp.md`
- `reports/10_eval_multimodal.md` (tabla comparativa final)

## 11. Después de Fase 4

Independientemente del resultado, siguiente fase es **Fase 6 — Integración
del pipeline de pase de lista** (ver propuesta en reporte 06, sección 6.3).
Fase 4 no bloquea Fase 6: si el multimodal no supera al solo-silueta, se
integra el solo-silueta. Lo importante es tener el MVP funcionando.

---

**Plan cerrado. Esperar a estar en la PC nueva (con GPU ≥ 8 GB VRAM) para
ejecutar los pasos A–F. Mientras tanto, este documento sirve de contrato:
qué se va a hacer, con qué datos, y cómo se mide el éxito.**
