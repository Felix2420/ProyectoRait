"""
Genera el PDF: Modelos, Scripts y Arquitectura del Proyecto - Reconocimiento de Marcha
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import os

OUTPUT = r"C:\Proyecto3\ProyectoChino\reports\Modelos_Scripts_Arquitectura.pdf"

# ── Estilos ──────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def estilo(nombre, parent="Normal", **kw):
    return ParagraphStyle(nombre, parent=base[parent], **kw)

titulo     = estilo("titulo",    "Title",   fontSize=20, leading=26, spaceAfter=6,
                    textColor=colors.HexColor("#1a1a2e"))
subtitulo  = estilo("subtitulo", "Normal",  fontSize=11, leading=14, spaceAfter=16,
                    textColor=colors.HexColor("#4a4a6a"), alignment=TA_CENTER)
h1         = estilo("h1",        "Heading1", fontSize=14, leading=18, spaceBefore=18,
                    spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
h2         = estilo("h2",        "Heading2", fontSize=11, leading=14, spaceBefore=12,
                    spaceAfter=4, textColor=colors.HexColor("#2d4a8a"))
h3         = estilo("h3",        "Heading3", fontSize=10, leading=13, spaceBefore=8,
                    spaceAfter=3, textColor=colors.HexColor("#3a3a6a"), fontName="Helvetica-Bold")
body       = estilo("body",      "Normal",  fontSize=9,  leading=13, spaceAfter=4,
                    alignment=TA_JUSTIFY)
bullet     = estilo("bullet",    "Normal",  fontSize=9,  leading=13, spaceAfter=2,
                    leftIndent=14, firstLineIndent=-8)
code_style = estilo("code",      "Normal",  fontSize=8,  leading=11, spaceAfter=4,
                    fontName="Courier", backColor=colors.HexColor("#f4f4f4"),
                    leftIndent=12, rightIndent=12, borderPad=4)
nota       = estilo("nota",      "Normal",  fontSize=8,  leading=11, spaceAfter=4,
                    textColor=colors.HexColor("#555555"), leftIndent=10)

# ── Helpers ──────────────────────────────────────────────────────────────────
W = A4[0] - 4*cm   # ancho útil

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"),
                      spaceAfter=6, spaceBefore=2)

def sp(n=6):
    return Spacer(1, n)

def tabla(data, col_widths=None, header=True):
    if col_widths is None:
        n = len(data[0])
        col_widths = [W / n] * n
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("LEADING",     (0, 0), (-1, -1), 11),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f9f9f9")]),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP",    (0, 0), (-1, -1), True),
    ]
    if header:
        style_cmds += [
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style_cmds))
    return t

def p(text, style=body):
    return Paragraph(text, style)

def b(text):
    return Paragraph(f"• {text}", bullet)

def code(text):
    lines = text.strip().split("\n")
    return Paragraph("<br/>".join(lines), code_style)

# ── Contenido ────────────────────────────────────────────────────────────────
story = []

# Portada
story += [
    sp(20),
    p("Reconocimiento de Marcha para Asistencia Automatizada", titulo),
    p("Modelos Utilizados · Arquitecturas · Scripts · Modelos Descartados", subtitulo),
    hr(),
    p("Proyecto: ProyectoChino  |  Fecha: 2026-05-06  |  GPU: GTX 1650 4 GB", nota),
    sp(30),
]

# ═══════════════════════════════════════════════════════════════════════════════
story += [p("1. MODELOS EN PRODUCCIÓN", h1), hr()]

# ── YOLOv11n ──
story += [p("1.1 YOLOv11n — Detección de Personas", h2)]
story += [
    p("YOLOv11n es el detector de objetos que localiza a la persona en cada frame "
      "de la cámara antes de cualquier otro procesamiento.", body),
    sp(4),
]
story += [tabla(
    [["Parámetro", "Valor"],
     ["Peso del modelo", "6 MB"],
     ["Confianza mínima (conf)", "0.35"],
     ["IoU threshold", "0.50"],
     ["Latencia típica", "~10–20 ms / frame"],
     ["Selección de bbox", "Mayor área (persona más cercana)"]],
    col_widths=[W*0.45, W*0.55]
), sp(6)]
story += [
    p("<b>¿Cómo se usó?</b> Corre en cada frame a 15 fps. Cuando detecta múltiples "
      "personas, el pipeline selecciona la de mayor bounding box (más cercana a la "
      "cámara). Su salida alimenta directamente al FSM Tracker.", body),
    p("<b>¿Por qué YOLOv11n y no otro?</b> Cabe en 4 GB de VRAM junto con los "
      "demás modelos, tiene latencia menor a 20 ms, y la precisión de detección "
      "de personas en entornos controlados es más que suficiente. No se requería "
      "un modelo más pesado.", body),
]

story += [sp(8), p("1.2 RVM MobileNetV3 — Segmentación de Siluetas", h2)]
story += [
    p("Robust Video Matting (RVM) con backbone MobileNetV3 convierte cada frame "
      "en una silueta binaria de 64×44 px — la entrada que GaitBase necesita.", body),
    sp(4),
]
story += [tabla(
    [["Parámetro", "Valor"],
     ["Backbone", "MobileNetV3"],
     ["Downsample ratio", "0.25 (480×270 interno)"],
     ["Resolución de salida silueta", "64 × 44 px, binaria"],
     ["Latencia típica", "~30–40 ms / frame"],
     ["Estado interno", "Recurrente — se resetea con el FSM"]],
    col_widths=[W*0.45, W*0.55]
), sp(6)]
story += [
    p("<b>¿Cómo se usó?</b> Solo se activa cuando el tracker está en estado ACTIVE. "
      "El <i>downsample_ratio=0.25</i> permite procesar a resolución reducida "
      "internamente para ahorrar VRAM sin pérdida de calidad relevante en la "
      "silueta resultante.", body),
    p("<b>¿Por qué RVM y no BackgroundSubtractor?</b> Un sustractor de fondo clásico "
      "falla ante iluminación cambiante, sombras o fondos no perfectamente estáticos. "
      "RVM es robusto a estas variaciones y produce siluetas limpias incluso con "
      "condiciones imperfectas.", body),
    p("<b>¿Cómo ayudó al proyecto?</b> Es el paso de conversión video→silueta. "
      "Sin siluetas de buena calidad, GaitBase no puede extraer patrones de marcha "
      "confiables. La calidad del segmentador limita directamente las métricas "
      "finales del sistema.", body),
]

story += [sp(8), p("1.3 RTMPose — Extracción de Keypoints", h2)]
story += [
    p("RTMPose detecta 17 keypoints del esqueleto (hombros, codos, muñecas, caderas, "
      "rodillas, tobillos) sobre cada frame. Aunque la rama de pose fue descartada "
      "del score de identidad, RTMPose cumple dos roles en el proyecto.", body),
    sp(4),
]
story += [tabla(
    [["Rol", "Descripción"],
     ["Fase 1 (offline)", "Generar archivos .npy de keypoints para todo el dataset"],
     ["Pipeline en vivo", "Quality gating: descartar siluetas donde la pose "
      "es incoherente o el sujeto está parcialmente fuera de cuadro"]],
    col_widths=[W*0.30, W*0.70]
), sp(6)]

story += [sp(8), p("1.4 GaitBase (ResNet9 Fine-tuned) — Modelo Central de Identidad", h2)]
story += [
    p("GaitBase es el núcleo del sistema. Convierte una secuencia de siluetas en "
      "un embedding de identidad que permite comparar personas.", body),
    sp(4),
    p("<b>Origen:</b> Baseline del framework OpenGait (Fan et al., CVPR 2023 Highlight). "
      "Preentrenado en Gait3D — 4,000 sujetos en condiciones no controladas.", body),
    sp(6),
    p("<b>Arquitectura interna:</b>", h3),
]
story += [code(
    "Entrada: secuencia de siluetas  [T=60 frames × 64 × 44 px, binarias]\n"
    "  ↓ SetBlockWrapper\n"
    "     Aplica ResNet9 frame a frame (pesos compartidos)\n"
    "     ResNet9: canales [64→128→256→512], capas [1,1,1,1], strides [1,2,2,1]\n"
    "  ↓ Temporal Pooling\n"
    "     PackSequenceWrapper(torch.max) → máximo temporal → (512, H', W')\n"
    "  ↓ HorizontalPoolingPyramid (HPP)\n"
    "     bin_num=[16] → divide en 16 bandas horizontales (cabeza→pies)\n"
    "  ↓ SeparateFCs\n"
    "     512 → 256 dims por cada una de las 16 bandas\n"
    "  ↓ BNNeck (BatchNorm Bottleneck)\n"
    "     class_num=13 (sujetos de entrenamiento)\n"
    "Salida: embedding 256 × 16 = 4,096 dims, L2-normalizado\n"
    "        → similitud coseno contra galería"
), sp(6)]
story += [tabla(
    [["Componente", "Función"],
     ["ResNet9",    "Extrae características espaciales de cada silueta individual"],
     ["SetBlockWrapper", "Aplica ResNet9 sobre cada frame de la secuencia con pesos compartidos"],
     ["Temporal Pooling", "Condensa toda la secuencia en un solo descriptor (máximo temporal)"],
     ["HPP (16 bandas)", "Captura detalles locales: dinámica de cabeza, torso, piernas y pies por separado"],
     ["BNNeck", "Normaliza antes de la clasificación, estabiliza el espacio de embeddings"],
     ["L2-normalizado", "Permite usar similitud coseno como métrica de distancia (rango [−1, 1])"]],
    col_widths=[W*0.28, W*0.72]
), sp(6)]
story += [
    p("<b>¿Por qué ResNet9 y no ResNet50?</b> Las siluetas son imágenes binarias de "
      "64×44 px — mucho más simples que imágenes RGB naturales. ResNet9 con 4 capas "
      "es suficiente para extraer representaciones discriminativas y cabe en 1.25 GB "
      "de VRAM durante entrenamiento.", body),
    p("<b>¿Por qué HPP?</b> Dividir la silueta en 16 bandas horizontales permite que "
      "el modelo aprenda la dinámica de cada región del cuerpo de forma independiente. "
      "El movimiento de piernas no interfiere con el de brazos en el embedding.", body),
    sp(4),
    p("<b>Checkpoint en producción:</b>", h3),
]
story += [code(
    "checkpoints/finetune/gaitbase_ft_multisession_best_iter1200.pt\n"
    "Tamaño: ~27 MB  |  Iteración de parada anticipada: 1,200 / 1,500 máx.\n"
    "Entrenado con: 13 sujetos × s1 + s2 combinadas (~51 secuencias)"
), sp(4)]
story += [tabla(
    [["Métrica", "Valor"],
     ["Rank-1 NN (normal → normal)", "94.4%  (17/18 correctos)"],
     ["Rank-1 NR (normal → rápido)", "94.7%  (18/19 correctos)"],
     ["Rank-5",                       "100%"],
     ["EER open-set",                 "8.11%"],
     ["TAR @ FAR=0%",                 "86.49%  (32/37 genuinos aceptados)"],
     ["Umbral τ",                     "0.9847"],
     ["VRAM pico entrenamiento",      "1.25 GB"],
     ["Tiempo de entrenamiento",      "~177 min en GTX 1650"]],
    col_widths=[W*0.50, W*0.50]
), sp(8)]

# ═══════════════════════════════════════════════════════════════════════════════
story += [p("2. MODELOS DESCARTADOS", h1), hr()]

story += [p("2.1 GaitBase Pretrained sin Fine-tune", h2)]
story += [
    p("La primera evaluación fue correr el checkpoint preentrenado en Gait3D "
      "directamente sobre nuestro dataset, sin ninguna adaptación.", body),
    sp(4),
]
story += [tabla(
    [["Métrica", "Resultado"],
     ["EER",          "48.65%  (prácticamente aleatorio)"],
     ["TAR @ FAR=0%", "16%"]],
    col_widths=[W*0.45, W*0.55]
), sp(4)]
story += [
    p("<b>¿Por qué falló?</b> Gait3D tiene 4,000 sujetos en exteriores con "
      "múltiples cámaras y ángulos variables. El dominio de nuestro dataset — "
      "19 personas específicas, 90° lateral fijo, indoor — es completamente "
      "distinto. El modelo no puede generalizar sin adaptación.", body),
]

story += [sp(8), p("2.2 GaitBase Fine-tune solo con Sesión 1 (s1)", h2)]
story += [
    p("Se hizo fine-tuning usando únicamente los datos de la sesión 1 "
      "(paso normal) para los 13 sujetos de entrenamiento.", body),
    b("Resultado en closed-set: Rank-1 aceptable dentro de s1."),
    b("Problema crítico: degradación severa en cross-session (s1→s2)."),
    p("<b>¿Por qué falló?</b> El modelo aprendió la diferencia entre personas "
      "basándose parcialmente en la ropa de cada uno en s1. Cuando se evaluó "
      "con s2 (ropa diferente, velocidad diferente), el modelo no generalizó. "
      "Este resultado motivó el entrenamiento multi-sesión.", body),
]

story += [sp(8), p("2.3 SkeletonGait++ (SGPP) — Fusión Silueta + Pose", h2)]
story += [
    p("SkeletonGait++ es una arquitectura multimodal publicada en AAAI 2024. "
      "Combina una rama de silueta con una rama de skeleton map (heatmap de "
      "keypoints) con pesos de fusión aprendibles.", body),
    sp(4),
    p("<b>¿Cómo se intentó?</b> Se inicializó con los pesos del GaitBase "
      "fine-tuned (multisesión) y se añadió la rama de pose. Se entrenó "
      "optimizando el peso α de fusión entre 0.0 (solo pose) y 1.0 "
      "(solo silueta).", body),
]
story += [tabla(
    [["Experimento", "α óptimo", "EER", "TAR@FAR=0%"],
     ["SGPP fusión tardía", "1.0", "8.11%", "86.49%"],
     ["ST-GCN Lite fusión", "1.0", "8.11%", "86.49%"]],
    col_widths=[W*0.40, W*0.18, W*0.18, W*0.24]
), sp(6)]
story += [
    p("<b>Resultado: α*=1.0 en todos los experimentos.</b> El optimizador "
      "asignó peso cero a la rama de pose en todos los casos. Silueta sola gana.", body),
    sp(4),
    p("<b>Cuatro razones técnicas del fracaso:</b>", h3),
    b("<b>Ángulo lateral 90°:</b> Las extremidades contralaterales se solapan "
      "en la imagen. Los keypoints de RTMPose ven poca diferencia entre sujetos "
      "a este ángulo. La silueta ya captura prácticamente toda la información "
      "de marcha visible."),
    b("<b>Dataset insuficiente para pose:</b> Solo ~51 secuencias de entrenamiento "
      "(13 sujetos × ~4 tipos). Las redes de pose necesitan miles de ejemplos "
      "para no sobreajustarse al ruido de detección."),
    b("<b>Solo 2 sesiones:</b> La red de pose aprende la diferencia s1 vs s2 "
      "(velocidad, ropa) en lugar de la dinámica real de marcha. El modelo "
      "aprende ruido del detector de poses como 'firma de identidad'."),
    b("<b>Evidencia cuantitativa directa:</b> El optimizador nos dijo explícitamente "
      "que la pose no aporta. Corregir a ricardomora con SGPP rompió otros pares — "
      "el trade-off no fue ventajoso."),
    sp(4),
    p("<b>Estado final de la pose en el pipeline:</b> RTMPose sigue corriendo en "
      "runtime para quality gating (descartar siluetas mal extraídas), pero su "
      "salida NO entra en el score de similitud de identidad.", body),
]

story += [sp(8), p("2.4 ST-GCN Lite — Solo Pose", h2)]
story += [
    p("Versión ligera de Spatial-Temporal Graph Convolutional Network. Modela "
      "relaciones entre articulaciones del esqueleto a lo largo del tiempo "
      "usando grafos.", body),
    b("Entrenado únicamente con keypoints RTMPose (sin silueta)."),
    b("Resultado: α*=1.0 en fusión. Misma conclusión que SGPP."),
    p("<b>¿Por qué falló?</b> Además de las razones ya mencionadas, un GCN "
      "espaciotemporal tiene muchos parámetros relativos a 51 secuencias de "
      "entrenamiento. El sobreajuste fue inmediato.", body),
]

# ═══════════════════════════════════════════════════════════════════════════════
story += [p("3. HIPERPARÁMETROS DE ENTRENAMIENTO", h1), hr()]
story += [tabla(
    [["Parámetro", "Valor", "Justificación"],
     ["Optimizador",         "SGD",          "Estándar en OpenGait, más estable con TripletLoss"],
     ["Learning Rate",       "0.01",         "Configuración original de GaitBase"],
     ["Momentum",            "0.9",          "Estándar en SGD"],
     ["Weight Decay",        "5×10⁻⁴",       "Regularización — necesaria con dataset pequeño"],
     ["Batch (P×K)",         "P=4, K=2",     "4 identidades × 2 secuencias = 8 muestras / iter"],
     ["Frames por clip",     "30",           "Ventana aleatoria — actúa como data augmentation"],
     ["Iteraciones máx.",    "1,500",        "Early stop activado a iter 1,200"],
     ["LR Milestones",       "[750, 1250]",  "Decay ×0.1 en cada milestone"],
     ["Triplet Margin",      "0.2",          "Margen para TripletLoss"],
     ["CE Scale / Smooth",   "16 / 0.1",     "Label smoothing para dataset pequeño"],
     ["AMP float16",         "Activado",     "Necesario para caber en GTX 1650 4 GB"]],
    col_widths=[W*0.28, W*0.18, W*0.54]
), sp(4)]
story += [
    p("<b>Función de pérdida combinada:</b>", h3),
]
story += [code(
    "Loss = TripletLoss(margin=0.2)  +  CrossEntropyLoss(scale=16, smoothing=0.1)\n\n"
    "TripletLoss  → acerca embeddings de la misma persona, aleja los de distintas\n"
    "CrossEntropy → el modelo aprende a clasificar entre las 13 clases de train\n"
    "Label smooth → evita confianza extrema, mejora generalización en dataset pequeño"
)]

# ═══════════════════════════════════════════════════════════════════════════════
story += [p("4. SCRIPTS DEL PROYECTO", h1), hr()]

story += [p("4.1 Preprocesamiento — Fases 0 a 2", h2)]
story += [tabla(
    [["Script", "Qué hace"],
     ["audit_raw_dataset.py",
      "Cuenta secuencias, verifica calidad y reporta estadísticas del dataset crudo. "
      "Primera ejecución del proyecto."],
     ["extract_all.py",
      "Corre RVM (siluetas PNG) + RTMPose (keypoints NPY) sobre todos los videos crudos."],
     ["pack_to_pkl.py",
      "Empaqueta los PNGs/NPYs en archivos .pkl formato OpenGait (una secuencia por archivo)."],
     ["merge_sessions_pkl.py",
      "Fusiona PKLs de s1 y s2 en data/pkl_multisession/ para entrenamiento cross-session."],
     ["build_partition_json.py",
      "Genera partition_finetune.json con los índices train/val/test."],
     ["define_splits.py",
      "Genera configs/splits.yaml con los 13/3/3 sujetos por split (seed=42, inmutable)."],
     ["validate_opengait_dataloader.py",
      "Verifica que el dataloader lee los PKLs correctamente antes de entrenar."]],
    col_widths=[W*0.35, W*0.65]
), sp(8)]

story += [p("4.2 Entrenamiento — Fases 3 y 4", h2)]
story += [tabla(
    [["Script", "Estado", "Qué hace"],
     ["finetune_gaitbase.py",
      "Archivado",
      "Fine-tune de GaitBase solo con s1. Resultado descartado por degradación cross-session."],
     ["finetune_gaitbase_multisession.py",
      "✓ Producción",
      "Fine-tune con s1+s2 combinadas. Produce gaitbase_ft_multisession_best_iter1200.pt."],
     ["finetune_skeletongaitpp.py",
      "Descartado",
      "Fine-tune de SkeletonGait++. α*=1.0, silueta sola gana."],
     ["train_stgcn.py",
      "Descartado",
      "Entrenamiento de ST-GCN Lite solo con keypoints. Sobreajuste severo."],
     ["smoke_test_gaitbase.py",
      "Utilidad",
      "Test rápido: carga modelo, pasa un batch, verifica que no hay errores de dimensiones."]],
    col_widths=[W*0.38, W*0.16, W*0.46]
), sp(8)]

story += [p("4.3 Evaluación — Fases 3 a 5", h2)]
story += [tabla(
    [["Script", "Estado", "Qué hace"],
     ["cross_session_eval.py",
      "✓ Activo",
      "Evaluación closed-set: rank-1, rank-5, matriz de confusión en los 19 sujetos."],
     ["openset_eval.py",
      "✓ Activo",
      "Calibración de τ: genera curva DET, calcula EER y TAR@FAR=0%. Produce τ=0.9807."],
     ["openset_eval_skeletongaitpp.py",
      "Descartado",
      "Evaluación open-set de SGPP. Confirma α*=1.0."],
     ["openset_eval_fusion.py",
      "Descartado",
      "Evaluación de fusión con peso α variable. Resultado: α*=1.0 en todos los casos."],
     ["openset_eval_fusion_pose.py",
      "Descartado",
      "Variante de fusión tardía silueta+pose. Mismos resultados."]],
    col_widths=[W*0.38, W*0.16, W*0.46]
), sp(8)]

story += [p("4.4 Pipeline en Vivo — Fase 6", h2)]
story += [tabla(
    [["Script / Módulo", "Qué hace"],
     ["build_gallery.py",
      "Construye la galería: pasa s1 y s2 de los 19 sujetos por GaitBase. "
      "Guarda embeddings.npy (19 × K × 256 × 16) y valid_mask.npy."],
     ["infer_video.py",
      "Inferencia offline sobre un .mp4. Genera CSV de identidades con timestamps. "
      "Útil para testing sin cámara en vivo."],
     ["infer_live.py  ← PRINCIPAL",
      "Script de producción. Abre C920, corre el pipeline completo en tiempo real, "
      "muestra UI OpenCV y escribe log CSV automático. "
      "Hotkeys: q=salir, r=reset manual, f=pantalla completa."]],
    col_widths=[W*0.30, W*0.70]
), sp(8)]

story += [p("4.5 Módulos Internos — src/pipeline/", h2)]
story += [tabla(
    [["Módulo", "Responsabilidad"],
     ["capture.py",    "Lee frames de cámara o archivo de video con OpenCV."],
     ["detect.py",     "Wrapper de YOLOv11n. Devuelve bounding box de mayor área."],
     ["track.py",      "FSM de 4 estados: IDLE → WARMING → ACTIVE → RESET. "
                       "Warmup 15 frames, reset tras 15 frames sin detección."],
     ["segment.py",    "Wrapper de RVM MobileNetV3. Devuelve silueta binaria 64×44."],
     ["seq_buffer.py", "FIFO con ventana=60 frames y stride=30. "
                       "Emite una secuencia completa cada 30 frames (2 s a 15 fps)."],
     ["embed.py",      "Carga GaitBase fine-tuned. Convierte secuencia de siluetas "
                       "en embedding 256×16, L2-normalizado."],
     ["match.py",      "Compara embedding contra galería usando similitud coseno. "
                       "Devuelve mejor match y similitud. Aplica umbral τ."],
     ["confirm.py",    "Exige N=2 identificaciones consecutivas iguales por encima "
                       "de τ antes de confirmar la identidad."],
     ["log.py",        "Escribe fila en CSV con timestamp, subject, mean_sim, "
                       "n_sequences, frames_used y session_id."]],
    col_widths=[W*0.22, W*0.78]
)]

# ═══════════════════════════════════════════════════════════════════════════════
story += [sp(8), p("5. RESUMEN COMPARATIVO DE MODELOS", h1), hr()]
story += [tabla(
    [["Modelo", "Tipo", "EER", "TAR@FAR=0%", "Estado"],
     ["GaitBase pretrained\n(sin fine-tune)",    "Silueta",    "48.65%",  "16%",    "❌ Descartado"],
     ["GaitBase FT s1 solo",                     "Silueta",    "—",       "Degrada CS", "❌ Descartado"],
     ["GaitBase FT s1+s2",                       "Silueta",    "8.11%",   "86.49%", "✓ Producción"],
     ["SkeletonGait++",                           "Sil.+Pose",  "8.11%",   "86.49%", "❌ α*=1.0"],
     ["ST-GCN Lite",                              "Solo Pose",  "8.11%",   "86.49%", "❌ α*=1.0"]],
    col_widths=[W*0.30, W*0.16, W*0.12, W*0.18, W*0.24]
), sp(4)]
story += [
    p("Los modelos multimodales (SGPP, ST-GCN) producen exactamente las mismas "
      "métricas que GaitBase solo porque el optimizador descartó la rama de pose "
      "en todos los experimentos (α*=1.0). La silueta sola es suficiente y óptima "
      "para este dataset y ángulo de captura.", nota),
]

# ── Build ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
    title="Modelos, Scripts y Arquitectura — Reconocimiento de Marcha",
    author="ProyectoChino"
)
doc.build(story)
print(f"PDF generado: {OUTPUT}")
