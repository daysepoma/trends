# look-away-player

Plays a video fullscreen, with audio and in a loop, whenever you are not
looking at the screen. As soon as you look back, the player is closed.

"Not looking" means any of the following, measured from the webcam:

- no face is detected,
- your eyes are closed,
- your head is pitched up or down beyond a threshold,
- your head is turned sideways beyond a threshold.

This is head-pose estimation, not eye-gaze tracking: it does not know where
your pupils point, only where your head is facing.

## Requirements

- macOS (developed and tested on Apple Silicon, Python 3.14).
- VLC, with the `vlc` command available on
  `PATH`. With Homebrew: `brew install --cask vlc`.
- Camera permission for the terminal app that runs the script:
  *System Settings → Privacy & Security → Camera*.
- `face_landmarker.task` next to `main.py` (already versioned in this
  repository; it is the float16 MediaPipe Face Landmarker model).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py <video> [options]
```

Press `Ctrl+C` to quit. Any VLC window that is open is closed on exit.

| Option | Default | Meaning |
| --- | --- | --- |
| `video` | (required) | Video file to play when you look away. |
| `--camera` | `0` | Camera index. |
| `--threshold` | `2.0` | Seconds without looking before playback starts. |
| `--ear-threshold` | `0.21` | Eye aspect ratio below this means eyes closed. |
| `--pitch-threshold` | `18.0` | Head pitch (degrees) beyond this means looking away. |
| `--yaw-threshold` | `25.0` | Head yaw (degrees) beyond this means looking away. |
| `--debug` | off | Print `looking`, `ear`, `pitch` and `yaw` for every frame. |

### Calibrating the thresholds

Defaults are a starting point; faces, cameras and lighting differ. Run with
`--debug`, look straight at the screen and note the values, then look away
and compare:

```bash
python main.py <video> --debug
```

- `ear` is typically around 0.25-0.40 with open eyes and drops towards 0.1
  when they close. Set `--ear-threshold` between the two.
- `pitch` and `yaw` should be close to 0 when you face the screen. If your
  camera sits above or below the display, you will see a constant offset;
  raise the matching threshold accordingly.

## How it works

1. **Capture**: OpenCV reads frames from the webcam.
2. **Landmarks**: MediaPipe `FaceLandmarker` (running mode `VIDEO`, one face)
   returns facial landmarks for each frame.
3. **Eyes**: the eye aspect ratio (EAR) is computed from six landmarks per
   eye and averaged. Low EAR means closed eyes.
4. **Head pose**: six landmarks (nose tip, chin, eye corners, mouth corners)
   are matched to a generic 3D face model with `cv2.solvePnP`; pitch and yaw
   are extracted from the rotation. Angles are normalised to ±90°.
5. **Decision**: you are *looking* only if eyes are open **and** pitch and yaw
   are within their thresholds.
6. **Playback**: once you have not been looking for `--threshold` seconds, the
   script launches `vlc --fullscreen --loop` as a separate process. When you
   look again it terminates that process (and kills it after 2 s if it does
   not exit).

If VLC exits by itself (window closed, decode error) the script reports it and
waits another full threshold before launching it again, so a failing player is
never respawned on every frame.

## Design notes and known limits

- **VLC is launched as a subprocess instead of using `python-vlc`.** Embedding
  `libvlc` on macOS requires handing it a Cocoa `NSView`; without one it plays
  audio but fails with `No drawable-nsobject found` and shows no video.
- **Startup latency.** Each time playback is triggered a new VLC process is
  started, which takes a moment on top of `--threshold`. Keeping a single
  VLC instance running and controlling it over its remote-control interface
  would remove that delay but is not implemented.
- **Generic head model.** `solvePnP` uses a standard 3D face and an
  approximate camera matrix (focal length = image width), so absolute angles
  can be off by several degrees. Use `--debug` to calibrate rather than
  trusting the absolute numbers.
- **Not gaze tracking.** Looking at a second monitor with your head straight
  is not detected, and looking at the screen with your head turned slightly is
  still "looking" as long as it is within the thresholds.
- **Single face.** Only the first detected face is considered.
- **Camera glitches.** Up to 30 consecutive failed frame reads are tolerated;
  after that the script exits with an error message.

## Pinned dependencies

| Package | Constraint | Why |
| --- | --- | --- |
| `mediapipe` | `==0.10.35` | `1.0.1` aborts on macOS (`CHECK failed: Service is unavailable`) and drops the legacy `solutions` API. |
| `opencv-contrib-python` | `<5` | `mediapipe` depends on it; version 5 removed `CascadeClassifier`. Do not install `opencv-python` alongside it, both provide `cv2`. |
