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

También hace falta un modelo Hand Landmarker de MediaPipe (formato `.task`;
no se versiona en este repo). Se obtiene de la documentación oficial de
MediaPipe.

Y `ffmpeg` disponible en el PATH (para remuxar el audio original).

## Uso

```
python3 spotlight.py <entrada> <salida> <modelo>
```

Los tres argumentos son posicionales y obligatorios.

## Parámetros del efecto (constantes al inicio de `spotlight.py`)

- `FINGER_TIP_IDS`: landmarks de MediaPipe usados como vértices (por
  defecto los 5 dedos de cada mano).
- `PINK_WEIGHTS_BGR`: pesos por canal que tiñen el negativo de rosa.
- `HOLD_FRAMES`: cuántos frames retiene la última posición de una mano si
  la detección falla puntualmente, para evitar parpadeo.
