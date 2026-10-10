# hand-spotlight-effect

Detecta manos en un video con MediaPipe HandLandmarker y dibuja, cuadro a
cuadro, la figura (polígono convexo) formada por las puntas de los 5 dedos
de cada mano detectada. Dentro de esa figura se aplica un efecto negativo
con tinte rosa; el resto del frame queda intacto. El audio original se
conserva.

## Requisitos

```
pip install -r requirements.txt
```

También hace falta el modelo `hand_landmarker.task` de MediaPipe (no se
versiona en este repo por tamaño). Se puede descargar, por ejemplo, de:

```
curl -sSL -o hand_landmarker.task \
  https://raw.githubusercontent.com/IoT-gamer/hand_landmarker/main/android/src/main/assets/hand_landmarker.task
```

Y `ffmpeg` disponible en el PATH (para remuxar el audio original).

## Uso

```
python3 spotlight.py entrada.mp4 salida.mp4 hand_landmarker.task
```

Los tres argumentos son posicionales y opcionales: si se omiten, usa
`input.mp4`, `output_fingertip.mp4` y `hand_landmarker.task` en el
directorio actual.

## Parámetros del efecto (constantes al inicio de `spotlight.py`)

- `FINGER_TIP_IDS`: landmarks de MediaPipe usados como vértices (por
  defecto los 5 dedos de cada mano).
- `PINK_WEIGHTS_BGR`: pesos por canal que tiñen el negativo de rosa.
- `HOLD_FRAMES`: cuántos frames retiene la última posición de una mano si
  la detección falla puntualmente, para evitar parpadeo.
