import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (263, 387, 385, 362, 380, 373)

# Classic 6-point head pose model: nose tip, chin, eye corners, mouth corners.
POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

MAX_FAILED_READS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a video (with audio) fullscreen, in a loop, whenever you aren't actually looking at the screen."
    )
    parser.add_argument("video", help="Path to the video file to play when you look away")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Seconds without looking before playback starts (default: 2.0)",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.21,
        help="Eye-aspect-ratio below this counts as eyes closed (default: 0.21)",
    )
    parser.add_argument(
        "--pitch-threshold",
        type=float,
        default=18.0,
        help="Head pitch in degrees beyond this counts as looking away (default: 18.0)",
    )
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=25.0,
        help="Head yaw in degrees beyond this counts as looking away (default: 25.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print live ear/pitch/yaw/looking values to help calibrate thresholds",
    )
    return parser.parse_args()


def eye_aspect_ratio(landmarks, indices, w: int, h: int) -> float:
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = math.dist(p2, p6) + math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def estimate_head_pose(landmarks, w: int, h: int) -> tuple[float, float] | None:
    """Return (pitch, yaw) in degrees, or None if the pose can't be solved."""
    image_points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in POSE_LANDMARKS],
        dtype=np.float64,
    )
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    ok, rotation_vector, _ = cv2.solvePnP(
        POSE_MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rmat)[0]
    pitch, yaw = angles[0], angles[1]
    if pitch < -90:
        pitch = 180 + pitch
    elif pitch > 90:
        pitch = pitch - 180
    # A near-frontal face can decompose as yaw ~ +/-180 (Euler ambiguity).
    if yaw > 90:
        yaw = yaw - 180
    elif yaw < -90:
        yaw = yaw + 180
    return pitch, yaw


def is_looking(
    ear: float | None,
    pose: tuple[float, float] | None,
    ear_threshold: float,
    pitch_threshold: float,
    yaw_threshold: float,
) -> bool:
    """True when eyes are open and the head faces the screen within the thresholds."""
    if ear is None or pose is None:
        return False
    pitch, yaw = pose
    return ear >= ear_threshold and abs(pitch) <= pitch_threshold and abs(yaw) <= yaw_threshold


def find_vlc_binary() -> str:
    path = shutil.which("vlc")
    if path is None:
        raise SystemExit(
            "Could not find the 'vlc' command on PATH. Install it with: brew install --cask vlc"
        )
    return path


def stop_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")
    vlc_bin = find_vlc_binary()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model file: {MODEL_PATH}")

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    process: subprocess.Popen | None = None
    away_since: float | None = None
    failed_reads = 0
    started_at = time.monotonic()
    last_timestamp_ms = -1

    print("Running. Press Ctrl+C to quit.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= MAX_FAILED_READS:
                    raise SystemExit(
                        f"Camera stopped delivering frames ({failed_reads} failed reads in a row)."
                    )
                time.sleep(0.05)
                continue
            failed_reads = 0

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = max(last_timestamp_ms + 1, int((time.monotonic() - started_at) * 1000))
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            looking = False
            ear = None
            pitch = None
            yaw = None
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                ear = (
                    eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                    + eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                ) / 2.0
                pose = estimate_head_pose(landmarks, w, h)
                if pose is not None:
                    pitch, yaw = pose
                looking = is_looking(
                    ear, pose, args.ear_threshold, args.pitch_threshold, args.yaw_threshold
                )

            if args.debug:
                print(f"looking={looking} ear={ear} pitch={pitch} yaw={yaw}")

            now = time.monotonic()

            # VLC may exit on its own (window closed, decode error). Drop the dead
            # handle and require a fresh threshold before relaunching so a failing
            # VLC can't be respawned on every frame.
            if process is not None and process.poll() is not None:
                print(f"VLC exited unexpectedly (code {process.returncode}).")
                process = None
                away_since = now

            if looking:
                away_since = None
                if process is not None:
                    stop_process(process)
                    process = None
            elif away_since is None:
                away_since = now
            elif process is None and now - away_since >= args.threshold:
                process = subprocess.Popen(
                    [
                        vlc_bin,
                        "--fullscreen",
                        "--loop",
                        "--no-video-title-show",
                        "--quiet",
                        str(video_path),
                    ],
                    stdout=subprocess.DEVNULL,
                )
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        landmarker.close()
        if process is not None:
            stop_process(process)


if __name__ == "__main__":
    main()
