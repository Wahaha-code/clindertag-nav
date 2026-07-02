import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ASSET_ROOT = Path(__file__).resolve().parent
BUILD_ROOT = ASSET_ROOT / "build"
if BUILD_ROOT.exists():
    sys.path.insert(0, str(BUILD_ROOT))

from cylindertag_cpp import CylinderTagRunner


ESC_KEY = 27
DEFAULT_FILENAME = "usb_cylindertag_recording.mp4"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800


def annotate_markers(
    image: np.ndarray,
    markers,
    pose_map,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length: float,
) -> None:
    for marker in markers:
        marker_id = marker["marker_id"]
        pose = pose_map.get(marker_id)
        if pose is None:
            continue

        rvec = np.array(pose["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(pose["tvec"], dtype=np.float64).reshape(3, 1)
        base = pose.get("model_base")
        axis_vec = pose.get("model_axis")
        if base is None or axis_vec is None:
            continue

        origin = np.array(base, dtype=np.float64).reshape(1, 3)
        axis_z = np.array(axis_vec, dtype=np.float64)
        norm_z = np.linalg.norm(axis_z)
        if norm_z < 1e-8:
            continue
        axis_z /= norm_z

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(ref, axis_z)) > 0.95:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_x = np.cross(axis_z, ref)
        norm_x = np.linalg.norm(axis_x)
        if norm_x < 1e-8:
            continue
        axis_x /= norm_x
        axis_y = np.cross(axis_z, axis_x)

        endpoints = np.vstack(
            [
                origin + axis_x.reshape(1, 3) * axis_length,
                origin + axis_y.reshape(1, 3) * axis_length,
                origin + axis_z.reshape(1, 3) * axis_length,
            ]
        )

        model_points = np.vstack([origin, endpoints])
        image_points, _ = cv2.projectPoints(
            model_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        image_points = image_points.reshape(-1, 2).astype(int)
        origin_pt = tuple(image_points[0])
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
        for idx in range(3):
            cv2.arrowedLine(
                image,
                origin_pt,
                tuple(image_points[idx + 1]),
                colors[idx],
                4,
                cv2.LINE_AA,
                0,
                0.2,
            )


def open_usb_camera(args) -> cv2.VideoCapture:
    backend = cv2.CAP_ANY
    if args.backend == "v4l2":
        backend = cv2.CAP_V4L2

    source = str(args.camera_device) if args.camera_device else args.camera_index
    capture = cv2.VideoCapture(source, backend)
    if args.width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        capture.set(cv2.CAP_PROP_FPS, args.fps)
    if args.fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    return capture


def run(args) -> None:
    runner = CylinderTagRunner(
        str(args.marker_path),
        str(args.model_path),
        str(args.camera_path),
        adaptive_thresh=args.adaptive_thresh,
        enable_subpix=not args.disable_subpix,
        subpix_dist=args.subpix_window,
    )
    cam_info = runner.camera_params()
    camera_matrix = np.asarray(cam_info["intrinsic"], dtype=np.float64)
    dist_coeffs = np.asarray(cam_info["dist_coeffs"], dtype=np.float64)

    capture = open_usb_camera(args)
    if not capture.isOpened():
        source = args.camera_device if args.camera_device else args.camera_index
        print(f"Failed to open USB camera source {source}")
        return

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(
        f"USB camera {args.camera_device if args.camera_device else args.camera_index}: "
        f"{actual_width}x{actual_height}@{actual_fps:.1f}"
    )

    video_writer = None
    fps_counter = 0
    fps_timer = time.time()
    fps_value: Optional[float] = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Failed to read frame from USB camera")
                break

            if args.flip >= 0:
                frame = cv2.flip(frame, args.flip)

            display = frame.copy()

            if args.record and video_writer is None:
                height, width = display.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_name = f"{timestamp}_{args.output_prefix}"
                fps = actual_fps if actual_fps and actual_fps > 1.0 else 30.0
                video_writer = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
                if video_writer.isOpened():
                    print(f"Recording USB video to {output_name}")
                else:
                    print(f"Failed to open video writer for file: {output_name}")
                    video_writer = None

            result = runner.process(frame)
            markers = result["markers"]
            poses = result["poses"]
            pose_map = {pose["marker_id"]: pose for pose in poses if pose["marker_id"] >= 0}

            if markers:
                annotate_markers(
                    display,
                    markers,
                    pose_map,
                    camera_matrix,
                    dist_coeffs,
                    axis_length=args.axis_length,
                )
            else:
                cv2.putText(
                    display,
                    "CylinderTag not detected",
                    (20, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_timer = now
                fps_counter = 0

            if fps_value is not None:
                cv2.putText(
                    display,
                    f"FPS: {fps_value:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            if video_writer is not None:
                video_writer.write(display)

            view = (
                display
                if (display.shape[1], display.shape[0]) == (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                else cv2.resize(
                    display,
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
            )
            cv2.imshow("CylinderTag USB Viewer", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC_KEY or key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        capture.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time CylinderTag pose estimation using a USB camera."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV USB camera index.")
    parser.add_argument(
        "--camera-device",
        type=Path,
        default=None,
        help="Video device path such as /dev/video0. Overrides --camera-index.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested capture FPS.")
    parser.add_argument(
        "--fourcc",
        type=str,
        default="MJPG",
        help="Requested capture FOURCC, for example MJPG or YUYV. Empty string disables it.",
    )
    parser.add_argument(
        "--backend",
        choices=["any", "v4l2"],
        default="v4l2",
        help="OpenCV capture backend.",
    )
    parser.add_argument(
        "--flip",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Frame flip mode: -1 disabled, 0 vertical, 1 horizontal.",
    )
    parser.add_argument(
        "--marker-path",
        type=Path,
        default=ASSET_ROOT / "CTag_2f12c.marker",
        help="Path to marker dictionary file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ASSET_ROOT / "CTag_2f12c_d32.model",
        help="Path to reconstructed 3D model.",
    )
    parser.add_argument(
        "--camera-path",
        type=Path,
        default=ASSET_ROOT / "cameraParams.yml",
        help="Camera calibration file for this USB camera.",
    )
    parser.add_argument("--adaptive-thresh", type=int, default=5, help="Adaptive threshold window size.")
    parser.add_argument("--disable-subpix", action="store_true", help="Disable sub-pixel corner refinement.")
    parser.add_argument("--subpix-window", type=int, default=5, help="Sub-pixel refinement window radius.")
    parser.add_argument("--axis-length", type=float, default=50.0, help="Axis length in mm.")
    parser.add_argument("--record", action="store_true", help="Record the annotated stream to disk.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_FILENAME,
        help="Base filename for recordings.",
    )
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
