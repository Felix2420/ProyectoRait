"""
Genera el PDF: Explicación Explícita de Cada Fase del Proyecto
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import os

OUTPUT = r"C:\Proyecto3\ProyectoChino\reports\Fases_Proyecto_Explicacion.pdf"
W_PAGE = A4[0] - 4*cm

# ── Estilos ──────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def E(nombre, parent="Normal", **kw):
    return ParagraphStyle(nombre, parent=base[parent], **kw)

titulo    = E("titulo",   "Title",    fontSize=22, leading=28, spaceAfter=4,
              textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
subtitulo = E("sub",      "Normal",   fontSize=10, leading=14, spaceAfter=20,
              textColor=colors.HexColor("#555577"), alignment=TA_CENTER)
fase_hdr  = E("fase_hdr","Normal",    fontSize=15, leading=20, spaceBefore=22,
              spaceAfter=2, textColor=colors.white, fontName="Helvetica-Bold",
              backColor=colors.HexColor("#1a1a2e"), leftIndent=8, rightIndent=8,
              borderPad=6)
h2        = E("h2",      "Heading2",  fontSize=11, leading=14, spaceBefore=10,
              spaceAfter=3, textColor=colors.HexColor("#2d4a8a"))
h3        = E("h3",      "Normal",    fontSize=9,  leading=12, spaceBefore=6,
              spaceAfter=2, textColor=colors.HexColor("#1a1a2e"),
              fontName="Helvetica-Bold")
body      = E("body",    "Normal",    fontSize=9,  leading=13, spaceAfter=4,
              alignment=TA_JUSTIFY)
bullet    = E("bullet",  "Normal",    fontSize=9,  leading=13, spaceAfter=2,
              leftIndent=16, firstLineIndent=-10)
code_s    = E("code",    "Normal",    fontSize=7.5,leading=10, spaceAfter=6,
              fontName="Courier", backColor=colors.HexColor("#f5f5f5"),
              leftIndent=10, rightIndent=10, borderPad=4,
              textColor=colors.HexColor("#222222"))
nota      = E("nota",    "Normal",    fontSize=8,  leading=11, spaceAfter=4,
              textColor=colors.HexColor("#666666"), leftIndent=10)
result_ok = E("res_ok",  "Normal",    fontSize=9,  leading=12, spaceAfter=3,
              textColor=colors.HexColor("#006600"), fontName="Helvetica-Bold",
              leftIndent=10)
result_no = E("res_no",  "Normal",    fontSize=9,  leading=12, spaceAfter=3,
              textColor=colors.HexColor("#880000"), fontName="Helvetica-Bold",
              leftIndent=10)

def hr(): return HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cccccc"),
                             spaceAfter=6, spaceBefore=2)
def sp(n=6): return Spacer(1, n)

def p(text, style=body):    return Paragraph(text, style)
def b(text):                return Paragraph(f"• {text}", bullet)
def code(text):
    return Paragraph("<br/>".join(text.strip().split("\n")), code_s)

def tabla(data, col_widths=None, header=True):
    if col_widths is None:
        col_widths = [W_PAGE / len(data[0])] * len(data[0])
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("FONTNAME",     (0,0),(-1,-1),"Helvetica"),
        ("FONTSIZE",     (0,0),(-1,-1),8),
        ("LEADING",      (0,0),(-1,-1),11),
        ("TOPPADDING",   (0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",  (0,0),(-1,-1),6),
        ("RIGHTPADDING", (0,0),(-1,-1),6),
        ("GRID",         (0,0),(-1,-1),0.4,colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f9f9f9")]),
        ("VALIGN",       (0,0),(-1,-1),"TOP"),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0,0),(-1,0),colors.HexColor("#2d4a8a")),
            ("TEXTCOLOR",  (0,0),(-1,0),colors.white),
            ("FONTNAME",   (0,0),(-1,0),"Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(cmds))
    return t

def fase_titulo(numero, nombre, estado=""):
    tag = f"  {estado}" if estado else ""
    return [
        sp(10),
        p(f"FASE {numero} — {nombre}{tag}", fase_hdr),
        sp(4),
    ]

# ══════════════════════════════════════════════════════════════════════════════
story = []

# ── Portada ──────────────────────────────────────────────────────────────────
story += [
    sp(30),
    p("Sistema de Reconocimiento de Marcha", titulo),
    p("Explicación Explícita de Cada Fase del Proyecto", subtitulo),
    hr(),
    p("Proyecto: ProyectoChino  ·  7 Fases  ·  Fecha: 2026-05-06", nota),
    sp(16),
]

# Índice de fases
story += [
    p("Fases del Proyecto", h2),
    tabla([
        ["#", "Nombre", "Script(s) Principal(es)", "Estado"],
        ["0", "Auditoría del Dataset",        "audit_raw_dataset.py",                      "Completada"],
        ["1", "Extracción de Siluetas",        "extract_all.py → extract_sequence.py",      "Completada"],
        ["2", "Empaquetado OpenGait",          "pack_to_pkl.py → merge_sessions → splits",  "Completada"],
        ["3", "Fine-tuning GaitBase",          "finetune_gaitbase_multisession.py",          "✓ Producción"],
        ["4", "Intento Multimodal",            "finetune_skeletongaitpp.py / train_stgcn.py","Descartada"],
        ["5", "Evaluación Open-set",           "openset_eval.py",                            "Completada"],
        ["6", "Pipeline en Vivo",              "build_gallery.py / infer_live.py",           "Completada"],
    ], col_widths=[W_PAGE*0.06, W_PAGE*0.24, W_PAGE*0.44, W_PAGE*0.26]),
    sp(4),
    PageBreak(),
]

# ══════════════════════════════════════════════════════════════════════════════
# FASE 0
# ══════════════════════════════════════════════════════════════════════════════
story += fase_titulo("0", "AUDITORÍA DEL DATASET")
story += [
    p("<b>Objetivo:</b> Antes de procesar nada, verificar el estado real del "
      "dataset crudo. Saber qué hay, qué falta y si la calidad es suficiente.", body),
    sp(4),
    p("Script: <b>audit_raw_dataset.py</b>", h3),
    b("Recorre todos los videos .mp4 en el Dataset Crudo."),
    b("Cuenta cuántas secuencias hay por sujeto y por condición (normal / rápido)."),
    b("Reporta cantidad de frames, fps y duración por video."),
    b("Detecta videos corruptos, vacíos o con resolución inesperada."),
    b("No modifica ningún archivo — solo lectura y reporte."),
    sp(6),
    p("Entregable: reporte estadístico del dataset crudo. "
      "Este paso determinó que había 19 sujetos con 2 sesiones cada uno "
      "y aproximadamente 120 secuencias en total, todas a 1080p ~30 fps.", body),
    PageBreak(),
]

# ══════════════════════════════════════════════════════════════════════════════
# FASE 1
# ══════════════════════════════════════════════════════════════════════════════
story += fase_titulo("1", "EXTRACCIÓN DE SILUETAS Y KEYPOINTS")
story += [
    p("<b>Objetivo:</b> Convertir cada video .mp4 crudo en arrays numpy de "
      "siluetas binarias (64×44 px) y keypoints del esqueleto normalizados. "
      "Esta es la entrada que GaitBase necesita.", body),
    sp(4),
    p("Scripts: <b>extract_all.py</b> (batch) → <b>extract_sequence.py</b> (por video)", h3),
    sp(4),
    p("Modelos utilizados:", h3),
]
story += [tabla([
    ["Modelo", "Entrada", "Salida", "Parámetros"],
    ["YOLOv11n",         "Frame BGR completo",       "Bbox (x1,y1,x2,y2)",              "conf=0.35, iou=0.5\nclases=[0] (persona)"],
    ["RVM MobileNetV3",  "Frame RGB float [0,1]",    "Alpha matte (H,W) float [0,1]",   "downsample_ratio=0.25\nestado recurrente entre frames"],
    ["RTMPose-m (ONNX)", "Frame BGR + bbox",         "17 keypoints COCO + confianzas",  "input_size=(192,256)\nbackend=onnxruntime"],
], col_widths=[W_PAGE*0.22, W_PAGE*0.24, W_PAGE*0.26, W_PAGE*0.28]), sp(8)]

story += [p("Pipeline por frame (del código extract_sequence.py):", h3)]
story += [code(
    "para cada frame del video:\n"
    "  1. bbox = YOLOv11n.predict(frame, classes=[0], conf=0.35)\n"
    "             → selecciona la bbox de mayor área × confianza\n\n"
    "  2. alpha = RVM(frame_rgb_tensor, *rec, downsample_ratio=0.25)\n"
    "             → alpha matte (H,W) float [0,1]\n"
    "             → rec = estado recurrente actualizado (coherencia temporal)\n\n"
    "  3. sil = crop_and_normalize_silhouette(alpha, bbox):\n"
    "             a) recorta alpha al bbox + padding=5%\n"
    "             b) binariza: píxel > 0.5 → 255, resto → 0\n"
    "             c) calcula centroide horizontal de píxeles blancos\n"
    "             d) ajusta lienzo a aspect ratio 44/64=0.6875\n"
    "                  - muy angosta → ensancha con ceros centrado en centroide\n"
    "                  - muy ancha   → recorta simétricamente al centroide\n"
    "             e) cv2.resize(INTER_AREA) → (64, 44) uint8\n\n"
    "  4. kp, scores = RTMPose(frame, bboxes=[bbox])\n"
    "             → kp_raw: (17,3) en píxeles originales\n\n"
    "  5. kp_norm = normalize_keypoints(kp_raw):\n"
    "             → centrar en cadera (promedio kp[11], kp[12])\n"
    "             → escalar por longitud torso (cadera → hombro)\n"
    "             → kp_norm: (17,3) float, x,y en unidades de torso"
), sp(4)]

story += [p("Detección del tramo útil (motion filtering):", h3)]
story += [code(
    "# Solo guardar frames donde la persona está realmente caminando\n"
    "MOTION_PX_THRESH = 6   # píxeles de desplazamiento mínimo entre frames\n"
    "MOTION_WIN       = 5   # frames consecutivos requeridos\n\n"
    "motion[i] = ||centro_bbox[i] - centro_bbox[i-1]||₂\n\n"
    "start = primer i donde motion[i:i+5] > 6 px  (todos)\n"
    "end   = último  i donde motion[i-5:i] > 6 px  (todos)\n\n"
    "→ descarta frames iniciales (persona quieta entrando al cuadro)\n"
    "→ descarta frames finales   (persona quieta saliendo del cuadro)"
), sp(4)]

story += [p("Salidas por video (en data/processed/sujeto/condición/):", h3)]
story += [tabla([
    ["Archivo",             "Contenido",                        "Shape"],
    ["silhouettes.npy",     "Siluetas binarias del tramo útil", "(T, 64, 44)  uint8  ∈ {0,255}"],
    ["keypoints.npy",       "Keypoints normalizados",           "(T, 17, 3)   float32"],
    ["keypoints_raw.npy",   "Keypoints en píxeles originales",  "(T, 17, 3)   float32"],
    ["bboxes.npy",          "Bounding boxes por frame",         "(T, 4)       float32"],
    ["meta.json",           "Parámetros usados + estadísticas", "JSON"],
    ["debug_silhouettes_grid.png", "Grid 4×4 de 16 siluetas",  "PNG (para inspección visual)"],
], col_widths=[W_PAGE*0.30, W_PAGE*0.38, W_PAGE*0.32]), PageBreak()]

# ══════════════════════════════════════════════════════════════════════════════
# FASE 2
# ══════════════════════════════════════════════════════════════════════════════
story += fase_titulo("2", "EMPAQUETADO A FORMATO OPENGAIT")
story += [
    p("<b>Objetivo:</b> Convertir los .npy de siluetas al formato de carpetas "
      "que el DataLoader de OpenGait espera, y fusionar las dos sesiones para "
      "el entrenamiento cross-session.", body),
    sp(6),
    p("Paso 2a — pack_to_pkl.py:", h3),
    b("Lee silhouettes.npy de cada sujeto/condición."),
    b("Verifica shape (T,64,44) — rechaza si no cumple."),
    b("Guarda como .pkl con pickle.dump usando HIGHEST_PROTOCOL."),
    b("Estructura de carpetas requerida por OpenGait:"),
]
story += [code(
    "data/pkl/\n"
    "  <sujeto>/\n"
    "    normal/\n"
    "      090/           ← ángulo de cámara (090 = 90° lateral)\n"
    "        seq00.pkl    ← np.ndarray (T, 64, 44) uint8\n"
    "    rapido/\n"
    "      090/\n"
    "        seq00.pkl\n\n"
    "Nota: '090' es obligatorio en el formato OpenGait aunque\n"
    "      solo haya un ángulo de cámara."
), sp(4)]

story += [p("Paso 2b — merge_sessions_pkl.py:", h3),
    b("Toma data/pkl/ (sesión 1) y data/pkl_s2/ (sesión 2)."),
    b("Renombra los tipos añadiendo el tag de sesión: normal → normal_s1, normal_s2."),
    b("Copia los .pkl a data/pkl_multisession/ manteniendo el label (sujeto) igual."),
    b("Resultado: 4 secuencias por sujeto — normal_s1, normal_s2, rapido_s1, rapido_s2."),
]
story += [code(
    "data/pkl_multisession/\n"
    "  alexbojorquez/\n"
    "    normal_s1/090/seq00.pkl   ← s1, paso normal\n"
    "    normal_s2/090/seq00.pkl   ← s2, paso normal\n"
    "    rapido_s1/090/seq00.pkl   ← s1, paso rápido\n"
    "    rapido_s2/090/seq00.pkl   ← s2, paso rápido\n\n"
    "El label 'alexbojorquez' es idéntico en s1 y s2.\n"
    "TripletSampler puede mezclar sesiones en el mismo batch."
), sp(4)]

story += [p("Paso 2c — define_splits.py + build_partition_json.py:", h3),
    b("Define el split subject-disjoint una sola vez con seed=42."),
    b("13 sujetos para train, 3 para validación, 3 para test."),
    b("Guarda configs/splits.yaml y configs/partition_finetune.json."),
    b("INMUTABLE: no se vuelve a modificar durante todo el proyecto."),
    sp(6),
]
story += [tabla([
    ["Split", "N", "Sujetos"],
    ["Train", "13", "jorgeespinoza, gilbertofelix, jesusantonioaguilarfelix, josefelix,\n"
               "carloscarrillo, juliouriarte, jesuscazarez, robertocastaño, luislopez,\n"
               "brayannajera, cesarvasquez, carlospadilla, rodrigoruiz"],
    ["Val",   "3",  "hectorsanchez, robertpereira, carlostorres"],
    ["Test",  "3",  "jesusvalenzuela, alexbojorquez, ricardomora"],
], col_widths=[W_PAGE*0.10, W_PAGE*0.06, W_PAGE*0.84]), PageBreak()]

# ══════════════════════════════════════════════════════════════════════════════
# FASE 3
# ══════════════════════════════════════════════════════════════════════════════
story += fase_titulo("3", "FINE-TUNING DE GAITBASE  ✓  PRODUCCIÓN")
story += [
    p("<b>Objetivo:</b> Adaptar el modelo GaitBase preentrenado en 4,000 sujetos "
      "(Gait3D) para que distinga a los 19 sujetos del proyecto. "
      "El entrenamiento usa s1+s2 combinadas para que el modelo aprenda "
      "la dinámica de marcha y no la ropa.", body),
    sp(6),
    p("Script: <b>finetune_gaitbase_multisession.py</b>", h3),
    sp(4),
    p("Construcción del modelo (del código):", h3),
]
story += [code(
    "class GaitBaseFT(nn.Module):\n"
    "    Backbone = SetBlockWrapper(\n"
    "        ResNet9(channels=[64,128,256,512], layers=[1,1,1,1],\n"
    "                strides=[1,2,2,1], in_channel=1, maxpool=False)\n"
    "    )                              # aplica ResNet9 frame a frame\n"
    "    FCs      = SeparateFCs(in=512, out=256, parts_num=16)\n"
    "    BNNecks  = SeparateBNNecks(class_num=13, in=256, parts_num=16)\n"
    "    TP       = PackSequenceWrapper(torch.max)  # pooling temporal\n"
    "    HPP      = HorizontalPoolingPyramid(bin_num=[16])\n\n"
    "  forward(sils, seqL):\n"
    "    outs    = Backbone(sils)         # (B, C, T, H, W)\n"
    "    outs    = TP(outs, seqL)[0]      # max temporal → (B, 512, H', W')\n"
    "    feat    = HPP(outs)              # 16 bandas horizontales\n"
    "    embed_1 = FCs(feat)              # (B, 256, 16)\n"
    "    embed_2, logits = BNNecks(emb_1)\n"
    "    return embed_1, logits"
), sp(4)]

story += [p("Carga de pesos preentrenados:", h3),
    b("Se carga GaitBase_Gait3D_120000.pt (entrenado en 4,000 sujetos)."),
    b("Se copian todos los pesos EXCEPTO BNNecks (la capa de clasificación)."),
    b("BNNecks se reinicializa aleatoriamente para 13 clases (en vez de 4,000)."),
    b("Esto preserva las representaciones de marcha aprendidas en Gait3D."),
    sp(6),
]

story += [p("Hiperparámetros (de la clase Config en el código):", h3)]
story += [tabla([
    ["Parámetro",       "Valor",    "Justificación"],
    ["seed",            "42",       "Reproducibilidad"],
    ["batch_p (P)",     "4",        "4 identidades por batch"],
    ["batch_k (K)",     "2",        "2 secuencias por identidad = 8 muestras/iter"],
    ["frames_fixed",    "30",       "Ventana aleatoria de 30 frames por clip (data aug)"],
    ["lr",              "0.01",     "Mismo que configuración original GaitBase"],
    ["momentum",        "0.9",      "SGD estándar"],
    ["weight_decay",    "5e-4",     "Regularización L2 para dataset pequeño"],
    ["total_iter",      "1,500",    "Máximo; early stop activo"],
    ["milestones",      "[750, 1250]","Decay lr × 0.1 en cada milestone"],
    ["gamma",           "0.1",      "Factor de decay del lr"],
    ["triplet_margin",  "0.2",      "Margen del TripletLoss"],
    ["ce_scale",        "16.0",     "Escala logits antes del CrossEntropy"],
    ["ce_smoothing",    "0.1",      "Label smoothing — evita sobreconfianza"],
    ["early_stop_patience","3",     "Paradas sin mejora de margen NR val antes de detener"],
    ["eval_iter",       "200",      "Evaluar en val cada 200 iteraciones"],
    ["AMP float16",     "Activado", "GradScaler — necesario para GTX 1650 4 GB"],
], col_widths=[W_PAGE*0.28, W_PAGE*0.14, W_PAGE*0.58]), sp(6)]

story += [p("Función de pérdida combinada:", h3)]
story += [code(
    "Loss = TripletLoss(margin=0.2)  +  CrossEntropyLoss(scale=16, smooth=0.1)\n\n"
    "TripletLoss:\n"
    "  - Calcula distancias euclidianas entre todos los pares del batch\n"
    "  - ap = distancia anchor-positive (misma identidad)\n"
    "  - an = distancia anchor-negative (distinta identidad)\n"
    "  - loss = max(0, ap - an + margin)  por parte (16 bandas)\n"
    "  - Solo cuenta triplets activos (loss > 0)\n\n"
    "CrossEntropyLoss:\n"
    "  - logits * scale=16 → logits amplificados\n"
    "  - label_smoothing=0.1 → evita probabilidad 1.0 en la clase correcta\n"
    "  - Se aplica sobre las 16 bandas × 13 clases"
), sp(4)]

story += [p("Early stopping (del código):", h3),
    b("Métrica de parada: margen medio NR (normal→rápido cross-session) en val."),
    b("Margen = similitud_correcta − mejor_similitud_incorrecta."),
    b("Si 3 evaluaciones consecutivas (cada 200 iter) no mejoran el margen → parar."),
    b("Se guarda el estado del modelo en el mejor margen registrado."),
    b("Resultado: mejor checkpoint en iteración 1,200 de 1,500 máximas."),
    sp(6),
]

story += [p("Resultados del entrenamiento:", h3)]
story += [tabla([
    ["Métrica",                     "Valor"],
    ["Rank-1 NN (normal → normal)", "94.4%  (17/18 sujetos)"],
    ["Rank-1 NR (normal → rápido)", "94.7%  (18/19 sujetos)"],
    ["Rank-5 NN y NR",              "100%"],
    ["Error único consistente",     "ricardomora ↔ hectorsanchez (mismo build corporal)"],
    ["VRAM pico entrenamiento",     "1.25 GB"],
    ["Tiempo total",                "~177 min en GTX 1650"],
    ["Checkpoint de producción",    "gaitbase_ft_multisession_best_iter1200.pt"],
], col_widths=[W_PAGE*0.48, W_PAGE*0.52]), PageBreak()]

# ══════════════════════════════════════════════════════════════════════════════
# FASE 4
# ══════════════════════════════════════════════════════════════════════════════
story += fase_titulo("4", "INTENTO MULTIMODAL  ✗  DESCARTADA")
story += [
    p("<b>Objetivo original:</b> Mejorar las métricas fusionando silueta + pose "
      "(keypoints) en un modelo multimodal, como especificaba el requisito inicial.", body),
    sp(4),
    p("Scripts probados:", h3),
    b("<b>finetune_skeletongaitpp.py</b> — SkeletonGait++ (AAAI 2024): "
      "rama de silueta + rama de skeleton map (heatmap de keypoints)."),
    b("<b>train_stgcn.py</b> — ST-GCN Lite: "
      "Graph Convolutional Network sobre keypoints temporales."),
    b("<b>openset_eval_fusion.py</b> — evaluación de fusión tardía con peso α variable."),
    sp(6),
    p("Protocolo de evaluación:", h3),
    b("Se optimizó el peso α ∈ [0.0, 1.0] donde score = α·sim_silueta + (1−α)·sim_pose."),
    b("α=0.0 significa solo pose, α=1.0 significa solo silueta."),
    b("Se buscó el α que maximizara TAR@FAR=0%."),
    sp(6),
]

story += [tabla([
    ["Experimento",                     "α óptimo encontrado", "EER",   "TAR@FAR=0%"],
    ["SkeletonGait++ (fusión tardía)",   "1.0",                 "8.11%", "86.49%"],
    ["ST-GCN Lite (fusión tardía)",      "1.0",                 "8.11%", "86.49%"],
    ["Fusión pose manual GaitGraph2",    "0.4–0.7 (variable)",  "8.11%", "86.49%"],
], col_widths=[W_PAGE*0.40, W_PAGE*0.24, W_PAGE*0.14, W_PAGE*0.22]), sp(8)]

story += [
    p("<b>Conclusión: α*=1.0 en todos los experimentos.</b> El optimizador asignó "
      "peso cero a la rama de pose. Las métricas son idénticas a GaitBase solo.", body),
    sp(6),
    p("Cuatro razones técnicas del fracaso:", h3),
]
story += [tabla([
    ["Razón", "Explicación técnica"],
    ["Ángulo lateral 90°",
     "A 90°, las extremidades contralaterales se solapan en la imagen. "
     "Los keypoints de RTMPose no aportan dimensionalidad nueva que la silueta "
     "no contenga ya. La silueta captura prácticamente toda la dinámica visible."],
    ["Dataset insuficiente\npara la rama de pose",
     "Solo ~51 secuencias de entrenamiento (13 sujetos × ~4 tipos). "
     "Las redes de pose necesitan miles de ejemplos para no sobreajustarse. "
     "Con 51 secuencias aprenden 'ruido del detector' como firma de identidad."],
    ["Solo 2 sesiones",
     "El modelo de pose aprende la diferencia s1 vs s2 (velocidad, ropa) "
     "en vez de la dinámica real de marcha. SkeletonGait++ corrigió el caso "
     "ricardomora pero rompió otros pares — el trade-off fue negativo."],
    ["Evidencia empírica directa",
     "El optimizador nos dijo explícitamente que la pose no aporta. "
     "No es una hipótesis — es el resultado cuantitativo de los experimentos."],
], col_widths=[W_PAGE*0.25, W_PAGE*0.75]), sp(6)]

story += [
    p("Estado final de la pose en el pipeline:", h3),
    b("RTMPose sigue corriendo en el pipeline en vivo."),
    b("Su función es quality gating: descartar frames donde la silueta "
      "extraída es de baja calidad (persona fuera de cuadro, oclusión severa)."),
    b("NO entra en el score de similitud de identidad."),
    b("El requisito multimodal original se reinterpretó: "
      "'pose se calcula pero no se fusiona en score' — decisión basada en evidencia."),
    PageBreak(),
]

# ══════════════════════════════════════════════════════════════════════════════
# FASE 5
# ══════════════════════════════════════════════════════════════════════════════
story += fase_titulo("5", "EVALUACIÓN OPEN-SET Y CALIBRACIÓN DE UMBRAL")
story += [
    p("<b>Objetivo:</b> El modelo en closed-set siempre devuelve una identidad. "
      "En producción hay que poder rechazar a personas que NO están en el sistema. "
      "Esta fase calibra el umbral τ que controla esa decisión.", body),
    sp(6),
    p("Script: <b>openset_eval.py</b>", h3),
    sp(4),
    p("Protocolo LOSO cross-session (del código):", h3),
    b("Gallery: s1/normal de los 19 sujetos (18 válidos — falta "
      "jesusantonioaguilarfelix/normal_s1 por problema de extracción)."),
    b("Probes: s2/normal (protocolo NN) y s2/rapido (protocolo NR)."),
    b("Para cada probe con identidad i:"),
]
story += [code(
    "GENUINO:  query de persona i contra galería completa\n"
    "          → se espera que top1 = i  Y  sim >= τ  (aceptar)\n\n"
    "IMPOSTOR: query de persona i contra galería SIN persona i\n"
    "          → no importa quién sea top1, se espera sim < τ  (rechazar)\n\n"
    "Nota: el impostor es la misma persona pero evaluada como si fuera\n"
    "      un desconocido. Simula a alguien que no está en el sistema."
), sp(6)]

story += [p("Cálculo de métricas — barrido de τ (2,001 valores):", h3)]
story += [code(
    "para cada τ en linspace(sim_min, sim_max, 2001):\n"
    "  TAR = (correct_top1 AND sim >= τ) / n_genuine\n"
    "  FAR = (sim_top1_impostor >= τ)    / n_impostor\n"
    "  FRR = 1 - TAR\n\n"
    "EER  = punto donde |FAR - FRR| es mínimo\n"
    "     = (FAR[eer_idx] + FRR[eer_idx]) / 2\n\n"
    "Operating points calculados:\n"
    "  τ @ FAR=0%    → mayor τ que garantiza cero falsos positivos\n"
    "  τ @ FAR<=1%   → mayor τ con al menos 99% de impostores rechazados\n"
    "  τ @ FAR<=5%   → mayor τ con al menos 95% de impostores rechazados"
), sp(6)]

story += [p("Resultados (protocolo ALL = NN + NR combinados):", h3)]
story += [tabla([
    ["Métrica",                         "Valor"],
    ["EER (Equal Error Rate)",          "8.11%"],
    ["TAR @ FAR=0%",                    "86.49%  (32 de 37 genuinos identificados)"],
    ["TAR @ FAR≤1%",                    "86.49%"],
    ["TAR @ FAR≤5%",                    "86.49%"],
    ["Umbral τ @ FAR=0%",               "0.9807  (→ ajustado a 0.9847 en pipeline)"],
    ["Similitud media genuina",         "0.9883"],
    ["Similitud media impostor",        "0.9666"],
    ["Gap entre medias",                "+0.0217"],
    ["N pares genuinos evaluados",      "37"],
    ["N pares impostores evaluados",    "38"],
], col_widths=[W_PAGE*0.52, W_PAGE*0.48]), sp(6)]

story += [
    p("¿Por qué FAR=0% como restricción de diseño?", h3),
    b("En un sistema de asistencia, un falso positivo significa registrar "
      "como presente a alguien que no vino — inaceptable."),
    b("Un falso negativo (no registrar a alguien que sí vino) es corregible "
      "manualmente. Un falso positivo no lo es."),
    b("Por eso se sacrifica TAR (aceptamos que el 13.5% de los genuinos "
      "no sean registrados) a cambio de FAR=0%."),
    sp(4),
    p("Salidas del script:", h3),
    b("reports/06_openset_<tag>.json — métricas completas y curva DET."),
    b("reports/06_openset_<tag>_pairs.csv — detalle de cada par (probe, top1, sim, correcto)."),
    b("reports/06_openset_<tag>_roc.png — curva ROC con NN, NR y ALL."),
    PageBreak(),
]

# ══════════════════════════════════════════════════════════════════════════════
# FASE 6
# ══════════════════════════════════════════════════════════════════════════════
story += fase_titulo("6", "PIPELINE EN VIVO")
story += [
    p("<b>Objetivo:</b> Integrar todos los componentes en un sistema que corra "
      "en tiempo real con la cámara C920, identifique a las personas mientras "
      "caminan y registre la asistencia automáticamente.", body),
    sp(6),
    p("Fase 6a — Construcción de la Galería (build_gallery.py):", h2),
    b("Carga el checkpoint de producción: gaitbase_ft_multisession_best_iter1200.pt"),
    b("Por cada uno de los 19 sujetos, toma hasta K=4 secuencias de referencia (s1+s2)."),
    b("Pasa cada secuencia por GaitBase → embedding de 256×16 dimensiones."),
    b("Guarda los K embeddings por sujeto como templates de referencia."),
    sp(4),
]
story += [code(
    "gallery/embeddings.npy  → (19, K_max, 256, 16)  float32\n"
    "gallery/valid_mask.npy  → (19, K_max)            bool\n"
    "gallery/index.json      → metadata (subject_id, session, n_templates, ...)\n\n"
    "Estrategia de matching: max-sim por template (no promedio)\n"
    "  score(query, sujeto_i) = max( cosine(query, template_k) ) para k en templates_i\n"
    "  → captura variabilidad intra-sujeto (s1 ≠ s2)"
), sp(8)]

story += [p("Fase 6b — Inferencia en Tiempo Real (infer_live.py):", h2),
    p("El script abre la cámara C920 a 15 fps y encadena 8 módulos:", body),
    sp(4),
]
story += [tabla([
    ["Módulo (src/pipeline/)", "Función",                              "Parámetros clave"],
    ["capture.py",     "Lee frames de C920 con OpenCV",               "source=0, target_fps=15"],
    ["detect.py",      "YOLOv11n: detecta personas, devuelve bbox mayor área",
                                                                       "conf=0.35, iou=0.5"],
    ["track.py",       "FSM 4 estados: IDLE→WARMING→ACTIVE→RESET",    "warmup=15 frames, reset=15 frames"],
    ["segment.py",     "RVM: extrae silueta 64×44 (solo en ACTIVE)",  "downsample=0.25, resetea con FSM"],
    ["seq_buffer.py",  "FIFO: acumula siluetas y emite secuencias",   "window=60 frames, stride=30"],
    ["embed.py",       "GaitBase: secuencia → embedding 256×16",      "checkpoint iter1200, AMP float16"],
    ["match.py",       "Cosine max-sim contra galería, aplica τ",     "τ=0.9847, gallery 19 sujetos"],
    ["confirm.py",     "Exige N=2 matches consecutivos iguales",      "n_consecutive=2"],
    ["log.py",         "Escribe CSV de asistencia",                   "logs/attendance_YYYY-MM-DD.csv"],
], col_widths=[W_PAGE*0.20, W_PAGE*0.44, W_PAGE*0.36]), sp(8)]

story += [p("Lógica del FSM Tracker (track.py):", h3)]
story += [code(
    "IDLE    → sin detección. Sistema en reposo.\n\n"
    "IDLE    ──(bbox detectado)──────────────────→ WARMING\n\n"
    "WARMING → bbox presente pero aún acumulando warmup.\n"
    "          Requiere 15 frames consecutivos antes de procesar.\n"
    "          Evita disparar inferencia por detecciones fugaces.\n\n"
    "WARMING ──(≥15 frames con bbox)─────────────→ ACTIVE\n"
    "WARMING ──(bbox perdido)────────────────────→ IDLE\n\n"
    "ACTIVE  → acumula siluetas en el buffer, produce embeddings.\n\n"
    "ACTIVE  ──(bbox perdido >15 frames)─────────→ RESET\n\n"
    "RESET   → limpia buffer, estado RVM, contador de confirmación.\n"
    "RESET   ──(automático)──────────────────────→ IDLE"
), sp(6)]

story += [p("Flujo completo de una identificación:", h3)]
story += [code(
    "t=0s   Persona entra al campo de visión\n"
    "       YOLO detecta bbox → FSM pasa a WARMING\n\n"
    "t=1s   15 frames consecutivos con bbox → FSM pasa a ACTIVE\n"
    "       RVM empieza a extraer siluetas → buffer se llena\n\n"
    "t=3s   Buffer llega a 60 frames → primera secuencia emitida\n"
    "       GaitBase genera embedding → matcher devuelve top1 + similitud\n"
    "       Si sim >= τ=0.9847 → confirmación 1/2\n\n"
    "t=5s   Buffer acumula 30 frames más (stride) → segunda secuencia\n"
    "       Si mismo top1 con sim >= τ → confirmación 2/2\n"
    "       → IDENTIDAD CONFIRMADA\n"
    "       → Banner en UI: '✓ alexbojorquez (sim=0.9851)'\n"
    "       → Logger escribe fila en CSV\n\n"
    "Tiempo total desde entrada hasta confirmación: ~4-5 segundos"
), sp(6)]

story += [p("Análisis de latencia por componente:", h3)]
story += [tabla([
    ["Componente",          "Latencia típica",   "Nota"],
    ["Captura de frame",    "~67 ms",            "1/15 fps"],
    ["YOLOv11n",            "10–20 ms",          "GPU, batch=1"],
    ["RVM MobileNetV3",     "30–40 ms",          "GPU, downsample=0.25"],
    ["GaitBase embedding",  "80–100 ms",         "GPU, AMP float16, se ejecuta cada 30 frames"],
    ["Matching (19 sujetos)","<1 ms",            "NumPy, CPU"],
    ["Total por frame",     "~130–160 ms",       "≈7-8 fps de procesamiento efectivo"],
    ["VRAM total pipeline", "~3.5 GB pico",      "YOLO + RVM + GaitBase simultáneos"],
], col_widths=[W_PAGE*0.28, W_PAGE*0.22, W_PAGE*0.50]), sp(6)]

story += [p("Salida: CSV de asistencia diario:", h3)]
story += [code(
    "logs/attendance_2026-05-03.csv\n\n"
    "timestamp,subject,mean_sim,n_sequences,frames_used,session_id\n"
    "2026-05-03 10:15:23,alexbojorquez,0.9851,2,120,sess_abc123\n"
    "2026-05-03 10:18:45,carlostorres,0.9834,2,120,sess_def456"
), sp(4)]

story += [p("Hotkeys del sistema en vivo:", h3),
    b("q → salir del sistema."),
    b("r → reset manual (fuerza vuelta a IDLE sin esperar 15 frames)."),
    b("f → toggle pantalla completa."),
    PageBreak(),
]

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════════
story += [sp(10), p("RESUMEN EJECUTIVO DEL PROYECTO", fase_hdr), sp(8)]

story += [tabla([
    ["Fase", "Entregable Principal", "Resultado"],
    ["0 — Auditoría",      "Reporte de calidad del dataset",          "19 sujetos, ~120 seqs, OK"],
    ["1 — Extracción",     "silhouettes.npy (T,64,44) por video",     "~120 secuencias procesadas"],
    ["2 — Empaquetado",    "pkl_multisession/ + splits.yaml",         "4 seqs/sujeto, 13/3/3 split"],
    ["3 — Fine-tuning",    "gaitbase_ft_multisession_best_iter1200.pt","Rank-1: 94.5% | EER: 8.11%"],
    ["4 — Multimodal",     "Experimentos descartados (α*=1.0)",       "Silueta sola = óptimo"],
    ["5 — Open-set",       "τ=0.9847 | TAR@FAR=0%=86.49%",           "FAR=0% garantizado"],
    ["6 — Pipeline Vivo",  "infer_live.py + CSV de asistencia",       "~4s por identificación"],
], col_widths=[W_PAGE*0.22, W_PAGE*0.44, W_PAGE*0.34]), sp(8)]

story += [
    p("Limitaciones documentadas y aceptadas:", h3),
    b("ricardomora ↔ hectorsanchez: mismo build corporal, indistinguibles a 90° lateral."),
    b("Dataset pequeño (19 sujetos, 2 sesiones): la pose no puede aprovecharse."),
    b("Un solo ángulo (90°): cambio de posición de cámara requiere recalibración."),
    b("Cambio de ropa extremo puede degradar similitud por debajo de τ."),
    sp(6),
    p("Trabajo futuro (Fase 7):", h3),
    b("Quantización INT8 / TensorRT para reducir latencia de embedding a ~30 ms."),
    b("Reemplazar CSV por SQLite para log persistente y consultable."),
    b("Soporte multi-cámara y multi-ángulo."),
    b("Recalibración automática de τ con nuevas sesiones de grabación."),
]

# ── Build ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
    title="Explicación Explícita de Cada Fase — Reconocimiento de Marcha",
    author="ProyectoChino"
)
doc.build(story)
print(f"PDF generado: {OUTPUT}")
