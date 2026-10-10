import sys
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import numpy as np
import shutil
import subprocess
import os
import time
from contextlib import ExitStack
from pathlib import Path

# --- entrada / salida ---
if len(sys.argv) != 4:
    sys.exit("Uso: python3 spotlight.py <entrada> <salida> <modelo>")
INPUT, OUTPUT, MODEL = sys.argv[1:4]
if Path(INPUT).resolve() == Path(OUTPUT).resolve():
    sys.exit("La salida no puede ser el mismo archivo que la entrada (ffmpeg -y la sobrescribiría).")
TMP_VIDEO = OUTPUT + ".noaudio.mp4"

# --- parámetros del efecto ---
FINGER_TIP_IDS = [4, 8, 12, 16, 20]  # pulgar, índice, medio, anular, meñique
PINK_WEIGHTS_BGR = np.array([0.75, 0.35, 1.0], dtype=np.float32)  # tiñe el negativo de rosa
HOLD_FRAMES = 5  # si un frame puntual no detecta una mano, mantiene su última posición (sin difuminar) para que no parpadee

# --- verificaciones previas (fallar antes de procesar todo el video) ---
if shutil.which("ffmpeg") is None:
    sys.exit("No se encontró ffmpeg en el PATH (hace falta para conservar el audio).")
if not os.path.isfile(MODEL):
    sys.exit(f"No se encontró el modelo: {MODEL}")

cap = cv2.VideoCapture(INPUT)
if not cap.isOpened():
    sys.exit(f"No se pudo abrir el video de entrada: {INPUT}")
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    sys.exit(f"El video de entrada no reporta fps válidos: {INPUT}")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
diag = (width ** 2 + height ** 2) ** 0.5
line_thickness = max(4, int(diag * 0.006))  # grosor mínimo para que el contorno siempre sea visible

print(f"Video: {width}x{height} @ {fps:.3f}fps, {n_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(TMP_VIDEO, fourcc, fps, (width, height))
if not out.isOpened():
    cap.release()
    sys.exit(f"No se pudo crear el video temporal: {TMP_VIDEO}")

hands_detected_count = 0
held_hands = {}  # 'Left'/'Right' -> {"points": [...], "age": int}  (retención independiente por mano)
t0 = time.time()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=RunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def discard_tmp_on_error(exc_type, exc, tb):
    # no suprime la excepción: solo evita dejar el .noaudio.mp4 huérfano si el bucle falla
    if exc_type is not None and os.path.exists(TMP_VIDEO):
        os.remove(TMP_VIDEO)


with ExitStack() as stack, HandLandmarker.create_from_options(options) as landmarker:
    stack.push(discard_tmp_on_error)
    stack.callback(cap.release)
    stack.callback(out.release)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000.0 / fps))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        mask = np.zeros((height, width), dtype=np.uint8)
        hull = None

        if result.hand_landmarks:
            hands_detected_count += 1

        # actualiza, por mano (Left/Right), sus puntas de dedo detectadas este frame
        # la clave es etiqueta + ordinal ("Right0", "Right1"...) para que dos
        # manos con la misma etiqueta (num_hands=4) no se pisen entre sí
        seen_this_frame = set()
        label_counts = {}
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            base = result.handedness[i][0].category_name if result.handedness else "hand"
            n = label_counts.get(base, 0)
            label_counts[base] = n + 1
            label = f"{base}{n}"
            pts = [[int(hand_landmarks[t].x * width), int(hand_landmarks[t].y * height)] for t in FINGER_TIP_IDS]
            held_hands[label] = {"points": pts, "age": 0}
            seen_this_frame.add(label)

        # las manos no vistas este frame puntual retienen su última posición tal
        # cual (sin interpolar ni difuminar) hasta HOLD_FRAMES; así una mano no
        # borra la figura de la otra si a esta última le falla la detección
        for label in list(held_hands.keys()):
            if label in seen_this_frame:
                continue
            held_hands[label]["age"] += 1
            if held_hands[label]["age"] > HOLD_FRAMES:
                del held_hands[label]

        points = [pt for entry in held_hands.values() for pt in entry["points"]]
        if len(points) >= 2:
            hull = cv2.convexHull(np.array(points, dtype=np.int32))

        if hull is not None:
            if len(hull) >= 3:
                cv2.fillConvexPoly(mask, hull, 255)
            # contorno con grosor mínimo: cuando la figura se achica a una
            # línea (dedos casi tocándose) el relleno solo puede rasterizar
            # 0 píxeles; el trazo garantiza que la línea siga visible
            cv2.polylines(mask, [hull], isClosed=True, color=255, thickness=line_thickness)

        # el negativo rosa solo se calcula dentro de la figura (no en todo el frame)
        mask_bool = mask.astype(bool)
        composite = frame.copy()
        if mask_bool.any():
            region = frame[mask_bool].astype(np.float32)
            composite[mask_bool] = np.clip((255 - region) * PINK_WEIGHTS_BGR, 0, 255).astype(np.uint8)
        out.write(composite)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{n_frames}  ({time.time()-t0:.1f}s)")

cap.release()
out.release()

print(f"Frames procesados: {frame_idx}, con mano(s) detectada(s): {hands_detected_count}")
print(f"Tiempo de procesamiento (visión): {time.time()-t0:.1f}s")

print("Muxeando audio original...")
subprocess.run(
    [
        "ffmpeg", "-y",
        "-i", TMP_VIDEO,
        "-i", INPUT,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        OUTPUT,
    ],
    check=True,
)
os.remove(TMP_VIDEO)
print(f"Listo: {OUTPUT}")
