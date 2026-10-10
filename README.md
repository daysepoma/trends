# trends

Small computer-vision experiments driven by the webcam or by video files.
Each project lives in its own folder with its own `requirements.txt` and is
meant to run in its own virtual environment.

| Project | What it does | Input |
| --- | --- | --- |
| [`look-away-player`](look-away-player/README.md) | Plays a video fullscreen, in a loop, while you are not looking at the screen. | Live webcam |
| [`hand-spotlight-effect`](hand-spotlight-effect/README.md) | Draws the polygon formed by your fingertips on a video and applies a pink negative effect inside it. | Video file |

## Common setup

Tested on macOS (Apple Silicon) with Python 3.14. Both projects use
MediaPipe Tasks and OpenCV.

```bash
cd <project-folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependency versions are pinned on purpose (see each project's README):

- `mediapipe==0.10.35`: `1.0.1` crashes on macOS and removed the legacy
  `solutions` API.
- `opencv-contrib-python<5`: OpenCV 5 removed `CascadeClassifier` and
  installing `opencv-python` next to `mediapipe`'s own OpenCV package makes
  both write the same `cv2` module.

## Privacy

Videos are never meant to be committed. The root `.gitignore` excludes common
video and animated-image formats (`*.mp4`, `*.mov`, `*.avi`, `*.mkv`,
`*.webm`, `*.m4v`, `*.wmv`, `*.flv`, `*.mpg`, `*.mpeg`, `*.3gp`, `*.gif`).
Scripts receive media paths as command-line arguments, so no file outside this
repository is referenced from the code.

## Repository layout

```
trends/
├── look-away-player/
│   ├── main.py
│   ├── face_landmarker.task     # MediaPipe model, versioned (3.8 MB)
│   ├── requirements.txt
│   └── README.md
├── hand-spotlight-effect/
│   ├── spotlight.py
│   ├── requirements.txt
│   └── README.md
└── README.md
```
