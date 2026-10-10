# hand-spotlight-effect

Detecta manos en un video con MediaPipe HandLandmarker y dibuja, cuadro a
cuadro, la figura (polígono convexo) formada por las puntas de los 5 dedos
de cada mano detectada. Dentro de esa figura se aplica un efecto negativo
con tinte rosa; el resto del frame queda intacto. El audio original se
conserva (si el video de entrada no tiene audio, la salida queda solo con
video).

## Requisitos

```
pip install -r requirements.txt
```

Versiones fijadas a propósito:

- `mediapipe==0.10.35`: la `1.0.1` aborta en macOS.
- `opencv-contrib-python<5`: es el paquete de OpenCV que `mediapipe` ya
  requiere; no instales `opencv-python` al lado, ambos pisan el módulo `cv2`.

También hace falta un modelo Hand Landmarker de MediaPipe (formato `.task`;
no se versiona en este repo, `*.task` está en su `.gitignore`). Se obtiene
de la documentación oficial de MediaPipe.

Y `ffmpeg` disponible en el PATH (para remuxar el audio original).

## Uso

```
python3 spotlight.py <entrada> <salida> <modelo>
```

Los tres argumentos son posicionales y obligatorios.

Antes de procesar, el script verifica que `ffmpeg` esté en el PATH, que el
modelo exista y que el video de entrada se abra con fps válidos; si algo
falla, termina enseguida con un mensaje en vez de fallar al final.

## Cómo funciona

1. OpenCV lee el video cuadro a cuadro y MediaPipe (`RunningMode.VIDEO`,
   hasta 4 manos) devuelve los landmarks de cada mano.
2. Se toman las puntas de los dedos (`FINGER_TIP_IDS`) de todas las manos y
   se calcula su envolvente convexa. El relleno y un contorno de grosor
   mínimo forman una máscara.
3. Dentro de la máscara se reemplaza el píxel por su negativo teñido de rosa;
   solo se calcula en esa región.
4. Los cuadros se escriben a un video temporal sin audio
   (`<salida>.noaudio.mp4`).
5. `ffmpeg` recodifica (H.264, CRF 18) y mezcla el audio original; luego se
   borra el temporal.

Cada mano se identifica por su etiqueta (`Left`/`Right`) más un ordinal por
cuadro (`Right0`, `Right1`, ...), así dos manos con la misma etiqueta no se
pisan. Si una mano no se detecta en un cuadro puntual, conserva su última
posición hasta `HOLD_FRAMES` cuadros para que la figura no parpadee.

## Parámetros del efecto (constantes al inicio de `spotlight.py`)

- `FINGER_TIP_IDS`: landmarks de MediaPipe usados como vértices (por
  defecto los 5 dedos de cada mano).
- `PINK_WEIGHTS_BGR`: pesos por canal que tiñen el negativo de rosa.
- `HOLD_FRAMES`: cuántos frames retiene la última posición de una mano si
  la detección falla puntualmente, para evitar parpadeo.

## Limitaciones

- Procesa videos ya grabados; no funciona con webcam en vivo.
- El video se codifica dos veces (`mp4v` temporal y luego H.264), lo que
  añade tiempo y una pérdida de calidad pequeña.
- Si el script se interrumpe a mitad de proceso puede quedar el archivo
  temporal `<salida>.noaudio.mp4`; se puede borrar a mano.
- La etiqueta de mano que entrega MediaPipe puede alternar entre cuadros
  (`Left`/`Right`), lo que puede dejar una figura "fantasma" hasta
  `HOLD_FRAMES` cuadros.
