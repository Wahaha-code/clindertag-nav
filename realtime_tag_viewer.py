# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
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
from pyorbbecsdk import (
    Config,
    FormatConvertFilter,
    OBAlignMode,
    OBConvertFormat,
    OBFormat,
    OBSensorType,
    Pipeline,
    OBError,
    VideoStreamProfile,
)


ESC_KEY = 27
DEFAULT_FILENAME = "color_recording_1280x800.mp4"
TARGET_WIDTH = 3840
TARGET_HEIGHT = 2160
TARGET_FPS = 30
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800


def determine_convert_format(color_format: OBFormat) -> Optional[OBConvertFormat]:
    if color_format == OBFormat.I420:
        return OBConvertFormat.I420_TO_RGB888
    if color_format == OBFormat.MJPG:
        return OBConvertFormat.MJPG_TO_RGB888
    if color_format == OBFormat.YUYV:
        return OBConvertFormat.YUYV_TO_RGB888
    if color_format == OBFormat.NV21:
        return OBConvertFormat.NV21_TO_RGB888
    if color_format == OBFormat.NV12:
        return OBConvertFormat.NV12_TO_RGB888
    if color_format == OBFormat.UYVY:
        return OBConvertFormat.UYVY_TO_RGB888
    return None


def frame_to_bgr_image(frame, convert_filter_holder=[]):
    color_format = frame.get_format()
    width, height = frame.get_width(), frame.get_height()

    def reshape3(buf):
        img = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        return np.ascontiguousarray(img)

    # 直接三通道
    if color_format == OBFormat.RGB:
        return cv2.cvtColor(reshape3(frame.get_data()), cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return reshape3(frame.get_data())

    # 初始化或复用转换器
    if not convert_filter_holder:
        convert_filter_holder.append(FormatConvertFilter())
    convert_filter = convert_filter_holder[0]

    convert_format = determine_convert_format(color_format)
    if convert_format is None:
        print(f"Unsupported color format: {color_format}")
        return None

    convert_filter.set_format_convert_format(convert_format)
    rgb_frame = convert_filter.process(frame)
    if rgb_frame is None:
        print(f"Failed to convert frame from format {color_format}")
        return None

    rgb = reshape3(rgb_frame.get_data())
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)



def init_stream_profile(pipeline: Pipeline, sensor: OBSensorType):
    profile_list = pipeline.get_stream_profile_list(sensor)
    if sensor == OBSensorType.COLOR_SENSOR:
        try:
            profile: VideoStreamProfile = profile_list.get_video_stream_profile(
                TARGET_WIDTH, TARGET_HEIGHT, OBFormat.RGB, TARGET_FPS
            )
            return profile
        except OBError as exc:
            print(f"Failed to acquire {TARGET_WIDTH}x{TARGET_HEIGHT}@{TARGET_FPS} color profile: {exc}")
            profile = profile_list.get_default_video_stream_profile()
            print(f"Falling back to default color profile: {profile}")
            return profile
    return profile_list.get_default_video_stream_profile()


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

        # Build orthonormal basis
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(ref, axis_z)) > 0.95:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_x = np.cross(axis_z, ref)
        norm_x = np.linalg.norm(axis_x)
        if norm_x < 1e-8:
            continue
        axis_x /= norm_x
        axis_y = np.cross(axis_z, axis_x)

        endpoints = np.vstack([
            origin + axis_x.reshape(1, 3) * axis_length,
            origin + axis_y.reshape(1, 3) * axis_length,
            origin + axis_z.reshape(1, 3) * axis_length,
        ])

        model_points = np.vstack([origin, endpoints])
        image_points, _ = cv2.projectPoints(model_points, rvec, tvec, camera_matrix, dist_coeffs)
        image_points = image_points.reshape(-1, 2).astype(int)
        origin_pt = tuple(image_points[0])
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))  # X, Y, Z
        for idx in range(3):
            end_pt = tuple(image_points[idx + 1])
            cv2.arrowedLine(image, origin_pt, end_pt, colors[idx], 4, cv2.LINE_AA, 0, 0.2)


def run(args):
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

    pipeline = Pipeline()
    config = Config()

    try:
        color_profile = init_stream_profile(pipeline, OBSensorType.COLOR_SENSOR)
        config.enable_stream(color_profile)
    except Exception as exc:
        print(f"Failed to configure Orbbec streams: {exc}")
        return

    try:
        pipeline.enable_frame_sync()
    except Exception:
        pass

    try:
        pipeline.start(config)
    except Exception as exc:
        print(f"Failed to start Orbbec pipeline: {exc}")
        return

    color_fps = color_profile.get_fps()
    video_writer = None
    fps_counter = 0
    fps_timer = time.time()
    fps_value: Optional[float] = None

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                continue

            display = color_image.copy()

            if args.record and video_writer is None:
                height, width = color_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_prefix = args.output_prefix or DEFAULT_FILENAME
                output_name = f"{timestamp}_{filename_prefix}"
                video_writer = cv2.VideoWriter(output_name, fourcc, color_fps, (width, height))
                if video_writer.isOpened():
                    print(f"Recording color video to {output_name}")
                else:
                    print(f"Failed to open video writer for file: {output_name}")
                    video_writer = None

            result = runner.process(color_image)
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
                else cv2.resize(display, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
            )
            cv2.imshow("CylinderTag Real-Time Viewer", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC_KEY or key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        pipeline.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time CylinderTag pose estimation using Orbbec RGB feed.")
    parser.add_argument("--marker-path", type=Path, default=ASSET_ROOT / "CTag_2f12c.marker", help="Path to marker dictionary file.")
    parser.add_argument("--model-path", type=Path, default=ASSET_ROOT / "CTag_2f12c_d32.model", help="Path to reconstructed 3D model.")
    parser.add_argument("--camera-path", type=Path, default=ASSET_ROOT / "cameraParams.yml", help="Camera calibration file.")
    parser.add_argument("--adaptive-thresh", type=int, default=5, help="Adaptive threshold window size.")
    parser.add_argument("--disable-subpix", action="store_true", help="Disable sub-pixel corner refinement.")
    parser.add_argument("--subpix-window", type=int, default=5, help="Sub-pixel refinement window radius.")
    parser.add_argument("--axis-length", type=float, default=50.0, help="Axis length (in mm) for drawFrameAxes visualization.")
    parser.add_argument("--record", action="store_true", help="Record the annotated stream to disk.")
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_FILENAME, help="Base filename for recordings.")
    return parser.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
