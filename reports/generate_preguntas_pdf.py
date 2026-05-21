"""Genera Preguntas_Asesor.pdf con la batería de Q&A para defensa de tesis."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

OUT = Path(__file__).resolve().parent / "Preguntas_Asesor.pdf"

# ----------------------------------------------------------- Estilos
styles = getSampleStyleSheet()
H_TITLE = ParagraphStyle(
    "Title", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=20, leading=24, spaceAfter=8, textColor=colors.HexColor("#1a1a1a"),
)
H_SUB = ParagraphStyle(
    "Sub", parent=styles["Normal"], fontName="Helvetica-Oblique",
    fontSize=11, leading=14, spaceAfter=18, textColor=colors.HexColor("#555"),
)
H_BLOCK = ParagraphStyle(
    "Block", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=14, leading=18, spaceBefore=14, spaceAfter=8,
    textColor=colors.HexColor("#0b3d91"),
    borderPadding=4, borderColor=colors.HexColor("#0b3d91"),
)
H_Q = ParagraphStyle(
    "Q", parent=styles["Normal"], fontName="Helvetica-Bold",
    fontSize=11, leading=14, spaceBefore=8, spaceAfter=4,
    textColor=colors.HexColor("#222"),
)
H_A = ParagraphStyle(
    "A", parent=styles["Normal"], fontName="Helvetica",
    fontSize=10.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=4,
    textColor=colors.HexColor("#111"),
)
H_BULLET = ParagraphStyle(
    "Bullet", parent=H_A, leftIndent=18, bulletIndent=6, spaceAfter=2,
)

# ----------------------------------------------------------- Contenido
BLOCKS = [
    ("Bloque 1 — Motivacion y problema", [
        ("P1. Por que reconocimiento por marcha y no por rostro o huella?",
         "La marcha es biometrica <b>a distancia y sin cooperacion del sujeto</b>: "
         "no requiere mirar a una camara ni tocar un sensor. Funciona con baja "
         "resolucion, oclusion parcial de cara y con ropa cubriendo extremidades. "
         "Para pase de lista es ideal porque no interrumpe el flujo de personas. "
         "La huella o el rostro requieren proximidad o cooperacion; la marcha no."),
        ("P2. Por que no usar simplemente un sistema comercial?",
         "Porque (a) los sistemas comerciales de gait son caros y propietarios, "
         "(b) el objetivo del proyecto es <b>aplicar y validar el estado del arte "
         "open-source en condiciones reales</b>, no comprar una caja negra, y "
         "(c) entrenar/calibrar localmente permite adaptarse a la camara y al "
         "ambiente especificos del aula."),
        ("P3. Cual es exactamente el problema que resuelves?",
         "Identificacion <b>open-set</b> de 19 sujetos a 90 grados lateral en video "
         "continuo, distinguiendo enrolados de no-enrolados, en tiempo casi-real "
         "con hardware modesto (GTX 1650)."),
    ]),
    ("Bloque 2 — Estado del arte", [
        ("P4. Que metodos de gait existen y cual elegiste?",
         "Tres familias: <b>basados en silueta</b> (GEI, GaitSet, GaitGL, GaitBase, "
         "DeepGaitV2) que usan mascaras binarias; <b>basados en esqueleto/pose</b> "
         "(PoseGait, GaitGraph, SkeletonGait); y <b>multimodales</b> (SkeletonGait++) "
         "que combinan ambos. Eleg&iacute; <b>GaitBase</b> porque (1) es el baseline oficial "
         "de OpenGait, (2) tiene preentreno en Gait3D con 4,000 sujetos, (3) "
         "arquitectura simple y reproducible, (4) mejor compromiso accuracy/velocidad "
         "para mi hardware."),
        ("P5. Probaste el multimodal? Por que lo descartaste?",
         "Si, entrene SkeletonGait++ (iter1000). <b>No mejoro</b> sobre GaitBase puro "
         "en este angulo y volumen de datos. La razon: a 90 grados lateral la silueta "
         "ya captura casi toda la informacion discriminativa; agregar pose introduce "
         "ruido del estimador de keypoints sin agregar senial nueva. Decision basada "
         "en evidencia empirica, documentada."),
        ("P6. Que es Gait3D y por que importa?",
         "Gait3D es un dataset publico con <b>4,000 sujetos en escenarios reales</b> "
         "(no laboratorio). El preentreno en Gait3D le dio al modelo un espacio de "
         "embedding general antes del fine-tuning con mis 19 sujetos. Sin transfer "
         "learning, 19 sujetos serian insuficientes para entrenar desde cero."),
    ]),
    ("Bloque 3 — Dataset", [
        ("P7. Cuantos sujetos, cuantas sesiones, que condiciones?",
         "<b>19 sujetos</b>, <b>2 sesiones</b> separadas en el tiempo, <b>90 grados "
         "lateral</b>, ropa similar entre sesiones. 13 de los 19 se usaron en "
         "fine-tune; los 6 restantes solo aparecen en galeria (validan transfer "
         "del espacio aprendido)."),
        ("P8. Por que solo 90 grados?",
         "Por restriccion de tiempo y porque el escenario objetivo (entrada a aula) "
         "es controlable: la camara se monta perpendicular al paso. Es una "
         "<b>limitacion documentada</b> &mdash; para multi-vista necesitarias re-grabar "
         "con varios angulos, lo que el dataset actual no permite."),
        ("P9. Por que solo 2 sesiones?",
         "2 sesiones es el <b>minimo</b> para evaluar generalizacion temporal "
         "(entrenas con s1, evaluas con s2 y viceversa). Con 1 sola sesion el "
         "modelo memoriza condiciones especificas del dia (iluminacion, ropa "
         "exacta). Con 2 ya puedes medir si el embedding es estable."),
        ("P10. Como evitas overfitting con dataset chico?",
         "(1) Transfer learning desde Gait3D, (2) <b>fine-tune corto</b> "
         "(iter 1,200 con early stopping por validacion), (3) augmentation estandar "
         "de OpenGait (flip horizontal, perspective), (4) weight decay en SGD, "
         "(5) leave-camera-out / multisession split para que train y test nunca "
         "compartan sesion."),
    ]),
    ("Bloque 4 — Arquitectura", [
        ("P11. Explicame tu arquitectura en una frase.",
         "ResNet9 2D que procesa cada frame como imagen independiente &rarr; "
         "max-pooling temporal sobre la secuencia &rarr; Horizontal Pooling "
         "Pyramid en 16 franjas anatomicas &rarr; 16 FCs separadas &rarr; "
         "embedding (256, 16) = 4,096-D L2-normalizado."),
        ("P12. Por que max temporal y no LSTM o 3D-conv?",
         "Empiricamente, en GaitBase el max temporal supera o iguala a alternativas "
         "mas complejas con menos computo. La marcha es periodica: el max captura "
         "la pose extrema de cada feature a lo largo del ciclo. LSTM/3D-conv son "
         "utiles cuando la dinamica temporal fina importa; en gait, la <b>envolvente</b> "
         "del movimiento es suficiente."),
        ("P13. Que es la Horizontal Pooling Pyramid (HPP)?",
         "Divide el feature map en <b>16 franjas horizontales</b> (cabeza, cuello, "
         "hombros, ..., pies) y hace pooling en cada una. Cada franja captura una "
         "zona anatomica. Las 16 FCs separadas aprenden representaciones distintas "
         "por zona &mdash; la firma de marcha vive en como se mueven esas zonas "
         "relativamente."),
        ("P14. Por que 256 dimensiones por parte y no 512 o 128?",
         "Es el valor estandar de OpenGait/GaitBase. 256 x 16 = 4,096-D total. "
         "Mas dimensiones aumentan capacidad y costo de matching; menos sacrifican "
         "discriminacion. No hay justificacion teorica fuerte, si empirica del paper."),
    ]),
    ("Bloque 5 — Entrenamiento", [
        ("P15. Que perdida usaste?",
         "<b>Triplet loss</b> sobre embeddings (separacion metrica) + "
         "<b>Cross-Entropy</b> sobre las cabezas BNNeck (clasificacion de las 13 "
         "clases vistas). Es el esquema multi-loss estandar en re-id: triplet da el "
         "espacio metrico, CE estabiliza el entrenamiento."),
        ("P16. Optimizador e hiperparametros?",
         "<b>SGD con momentum 0.9, weight decay 5e-4</b>, MultiStepLR. "
         "LR de fine-tune mucho menor que el de preentreno (orden de 1e-3 o menos) "
         "para no destruir los pesos de Gait3D."),
        ("P17. Por que iteracion 1,200?",
         "<b>Early stopping por validacion</b>: la metrica en validacion dejo de "
         "mejorar y empezo a degradar (overfitting). 1,200 fue el best &mdash; lo demas "
         "se descarto."),
        ("P18. Cuanto tardo el fine-tune?",
         "Aproximadamente 1-3 horas en GTX 1650 para iter 1,200 con dataset pequeno "
         "(verificable contra los logs de entrenamiento)."),
    ]),
    ("Bloque 6 — Evaluacion", [
        ("P19. Que metricas reportas y que significan?",
         "<b>EER (Equal Error Rate) = 8.11%</b>: punto donde FAR = FRR. Resumen "
         "general del modelo. <b>TAR@FAR=0% = 86.49%</b>: con CERO falsos positivos, "
         "identificas correctamente al 86.49% de los genuinos &mdash; esta es la metrica "
         "operativa, en pase de lista marcar a alguien como otro es inaceptable. "
         "<b>tau = 0.9847</b>: umbral calibrado para alcanzar FAR=0% absoluto."),
        ("P20. Como elegiste tau?",
         "Calibracion en validacion: barrido de tau desde 0 hasta 1, encuentro el "
         "menor tau que da FAR=0% en pares impostor del split de validacion. "
         "<b>No se ajusto en test</b> (hubiera sido data leakage)."),
        ("P21. Por que FAR=0% y no FAR=1%?",
         "Porque en pase de lista un falso positivo es <b>peor</b> que un falso "
         "negativo. Si marcas a Pedro como Juan, el sistema falla silenciosamente. "
         "Si rechazas a Juan, el insiste o se reintenta. Asimetria operacional."),
        ("P22. Que es un open-set y por que importa aqui?",
         "Open-set = el sistema debe <b>rechazar</b> identidades no enroladas, no "
         "solo elegir la mas parecida entre las conocidas. Closed-set siempre "
         "devuelve un nombre; open-set devuelve desconocido si la similitud al "
         "mejor match es menor que tau. <b>Confirmado en vivo el 2026-05-06</b>: "
         "persona no enrolada -> sim=0.9403 < 0.9847 -> UNK correcto."),
        ("P23. Que split usaste para validar?",
         "<b>Leave-Camera-Out / Multisession</b>: train con s1, test con s2 (y "
         "vice-versa). Garantiza que ningun frame del entrenamiento aparece en "
         "evaluacion. Es el split estandar en OpenGait."),
    ]),
    ("Bloque 7 — Pipeline en vivo", [
        ("P24. Describe el pipeline end-to-end.",
         "(1) Captura C920 a 15 fps. (2) Detector YOLOv11n -> bbox de persona "
         "(mayor area). (3) Tracker FSM con warmup 15 frames y reset 15 frames sin "
         "bbox. (4) Segmentacion RVM MobileNetV3 -> silueta binaria, crop y "
         "normalize a 64x44. (5) SequenceBuffer acumula ventana de 60 frames con "
         "stride 30. (6) GaitBase -> embedding 4,096-D. (7) Matcher max-sim contra "
         "galeria (19 sujetos, K&le;4 templates). (8) Confirmacion N=2 ventanas "
         "consecutivas con sim > tau -> escribe CSV."),
        ("P25. Por que N=2 confirmaciones?",
         "Para reducir falsos positivos transitorios. Una sola ventana puede "
         "acertar por casualidad si la silueta es ruidosa; <b>2 ventanas consecutivas</b> "
         "con la misma identidad y sim > tau son evidencia mucho mas fuerte. "
         "Costo: ~2-4 segundos extra de latencia."),
        ("P26. Por que RVM y no Mask R-CNN o U2Net?",
         "RVM es <b>stateful entre frames</b> (mantiene memoria recurrente), produce "
         "siluetas <b>temporalmente consistentes</b> sin parpadeo, y corre en tiempo "
         "real en MobileNetV3. Mask R-CNN procesa frame a frame independiente -> "
         "siluetas inestables."),
        ("P27. Que FPS reales tienes?",
         "<b>~6-8 fps</b> en GTX 1650, dominados por (a) RVM segmentacion, (b) "
         "embedding GaitBase. El detector y matcher son baratos. Suficiente para "
         "ventanas de 4 s con stride 2 s."),
        ("P28. Que pasa si dos personas entran al cuadro?",
         "Politica actual: <b>bbox de mayor area</b>. Una persona a la vez. Es una "
         "limitacion operacional documentada &mdash; para multi-persona necesitarias "
         "tracking multi-objeto (DeepSORT/ByteTrack) y un buffer de secuencia por "
         "track-ID."),
    ]),
    ("Bloque 8 — Limitaciones y criticas", [
        ("P29. Por que falla el par ricardomora <-> hectorsanchez?",
         "Build corporal muy similar (estatura, complexion, longitud de zancada). "
         "El embedding los acerca en el espacio metrico. Es una <b>limitacion "
         "inherente al modelo</b>, no del pipeline. Documentado. Mitigacion: mas "
         "sesiones o multi-vista."),
        ("P30. Si cambias la camara, funciona?",
         "<b>No directamente.</b> tau debe recalibrarse y, dependiendo de optica/"
         "altura/distancia, conviene re-enrolar. Es una restriccion documentada "
         "&mdash; la calibracion a hardware especifico es estandar en biometria."),
        ("P31. Y si la persona viste muy diferente al dia del enrolamiento?",
         "Robustez no validada formalmente &mdash; esta pendiente en Fase 6.5. La "
         "silueta capta principalmente forma corporal y dinamica, no textura, asi "
         "que un cambio de ropa <b>moderado</b> deberia tolerarse; ropa muy holgada "
         "(abrigo largo) si degrada."),
        ("P32. 86.49% es bueno?",
         "Para 19 sujetos con 2 sesiones a 90 grados con FAR=0%: <b>si, esta en "
         "el rango razonable</b> del estado del arte para datasets pequenos. CASIA-B "
         "con GaitBase reporta ~95%+ pero con ~10x mas datos por sujeto y splits "
         "mas balanceados. Es honesto reportar la limitacion de datos."),
        ("P33. Tu sistema es seguro contra suplantacion (spoofing)?",
         "<b>No formalmente evaluado.</b> Un atacante podria grabar a un sujeto "
         "enrolado y reproducir el video frente a la camara. Defensas posibles "
         "(no implementadas): liveness por profundidad, deteccion de pantalla, "
         "sensor IR. Limitacion de scope."),
        ("P34. Y la privacidad? Estas procesando biometricos.",
         "Pregunta importante. Los <b>embeddings no son reversibles a video</b> sin "
         "acceso al modelo y galeria. El CSV almacena solo identidad + timestamp + "
         "similitud, no imagen. Pero formalmente, gait es dato biometrico y aplica "
         "regulacion (GDPR/leyes locales). En despliegue real requeriria "
         "consentimiento informado de los enrolados."),
    ]),
    ("Bloque 9 — Trabajo futuro", [
        ("P35. Que falta?",
         "<b>Fase 6.5</b>: validacion end-to-end formal con TPR/FPR/latencia en "
         "vivo. <b>Fase 7</b>: quantizacion INT8/TensorRT (objetivo: 30 ms vs 100 ms "
         "actual), SQLite en lugar de CSV, manual de calibracion. <b>Largo plazo</b>: "
         "multi-vista, multi-persona simultanea, robustez a ropa/oclusion."),
        ("P36. Si tuvieras 6 meses mas, que harias?",
         "(1) Re-grabar dataset con 3-4 angulos. (2) Duplicar sesiones a 4-6 por "
         "sujeto en condiciones variadas. (3) Entrenar DeepGaitV2 (sucesor de "
         "GaitBase, mejor accuracy). (4) Integrar tracking multi-persona. "
         "(5) Dashboard web para administrar la galeria sin tocar codigo."),
    ]),
    ("Bloque 10 — Preguntas trampa", [
        ("P37. Que hiciste TU y que viene del paper/repo?",
         "Vino de OpenGait: arquitectura GaitBase, codigo de entrenamiento, "
         "preentreno Gait3D. <b>Trabajo propio</b>: dataset completo (19 sujetos, "
         "2 sesiones, captura/anotacion), pipeline en vivo end-to-end, calibracion "
         "open-set, integracion RVM+YOLO+GaitBase, FSM tracker, logica de "
         "confirmacion, evaluacion formal, comparacion multimodal vs unimodal con "
         "decision justificada."),
        ("P38. Si te lo pidieran, podrias reproducir todo desde cero?",
         "Si: dataset documentado, scripts versionados, configs en YAML, "
         "checkpoints guardados, reportes por fase en reports/. Cualquiera con el "
         "dataset y el repo puede reproducir el resultado."),
        ("P39. Cual es tu contribucion cientifica vs. de ingenieria?",
         "<b>Predominantemente ingenieria aplicada</b>: integracion de componentes "
         "del SOTA en un sistema funcional con evaluacion honesta. La contribucion "
         "cientifica menor esta en la <b>evaluacion comparativa de multimodal vs "
         "unimodal a 90 grados</b> con evidencia de que el multimodal no aporta en "
         "este regimen &mdash; un dato util para futuros proyectos."),
        ("P40. Por que deberia aprobarte este proyecto?",
         "Porque demuestra: (a) dominio del SOTA en gait, (b) capacidad de "
         "integracion de sistema completo, (c) evaluacion rigurosa con metricas "
         "operacionales no solo offline, (d) honestidad sobre limitaciones, "
         "(e) un sistema <b>funcionando en vivo</b> validado en condiciones reales."),
    ]),
]


# ----------------------------------------------------------- Build PDF
def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=LETTER,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title="Preguntas de asesor — Gait Recognition",
        author="Proyecto Gait Recognition",
    )
    story = []

    story.append(Paragraph(
        "Preguntas de asesor &mdash; Defensa de proyecto",
        H_TITLE,
    ))
    story.append(Paragraph(
        "Reconocimiento de marcha (Gait Recognition) &middot; "
        "GaitBase fine-tune &middot; 19 sujetos &middot; pipeline en vivo &middot; "
        "checkpoint 2026-05-07",
        H_SUB,
    ))

    for block_title, qa in BLOCKS:
        story.append(Paragraph(block_title, H_BLOCK))
        for q, a in qa:
            story.append(Paragraph(q, H_Q))
            story.append(Paragraph(a, H_A))
            story.append(Spacer(1, 4))

    doc.build(story)
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    build()
