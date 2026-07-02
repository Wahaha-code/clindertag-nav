import argparse
import csv
import math
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ASSET_ROOT = Path(__file__).resolve().parent
BUILD_ROOT = ASSET_ROOT / "build"
if BUILD_ROOT.exists():
    sys.path.insert(0, str(BUILD_ROOT))

GALAXY_PYTHON_API = (
    Path.home()
    / "Downloads"
    / "Galaxy_Linux_Python_2.4.2503.9202"
    / "Galaxy_Linux_Python_2.4.2503.9202"
    / "api"
)
if GALAXY_PYTHON_API.exists():
    sys.path.insert(0, str(GALAXY_PYTHON_API))

from cylindertag_cpp import CylinderTagRunner

try:
    import gxipy as gx
except ImportError as exc:
    raise SystemExit(
        "gxipy is not available. Install the Daheng Galaxy Python SDK or keep "
        f"{GALAXY_PYTHON_API} available."
    ) from exc


ESC_KEY = 27
DEFAULT_FILENAME = "daheng_cylindertag_recording.mp4"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800


def as_vector(values, size: int = 3) -> Optional[np.ndarray]:
    if values is None:
        return None
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.all(np.isfinite(vector)):
        return None
    return vector


def rotation_angle_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    rot_a, _ = cv2.Rodrigues(rvec_a.reshape(3, 1))
    rot_b, _ = cv2.Rodrigues(rvec_b.reshape(3, 1))
    relative = rot_b @ rot_a.T
    trace_value = float(np.trace(relative))
    cos_angle = max(-1.0, min(1.0, (trace_value - 1.0) * 0.5))
    return math.degrees(math.acos(cos_angle))


class DetectionMetrics:
    def __init__(self, window_size: int, csv_path: Optional[Path], print_every_s: float):
        self.window_size = max(2, int(window_size))
        self.print_every_s = max(0.0, float(print_every_s))
        self.start_time = time.time()
        self.last_print_time = self.start_time

        self.camera_frames = 0
        self.empty_frames = 0
        self.processed_frames = 0
        self.detected_frames = 0
        self.pose_frames = 0
        self.total_markers = 0
        self.total_processing_ms = 0.0
        self.reprojection_errors = []
        self.observation_counts = []

        self.translation_window = deque(maxlen=self.window_size)
        self.rotation_step_window = deque(maxlen=self.window_size)
        self.translation_step_window = deque(maxlen=self.window_size)
        self.previous_tvec = None
        self.previous_rvec = None
        self.rotation_jump_count = 0

        self.csv_file = None
        self.csv_writer = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = csv_path.open("w", newline="")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=[
                    "time_s",
                    "camera_frame",
                    "processed_frame",
                    "process_ms",
                    "detected",
                    "pose_valid",
                    "marker_count",
                    "primary_marker_id",
                    "feature_count",
                    "observation_count",
                    "reprojection_error",
                    "t_x_mm",
                    "t_y_mm",
                    "t_z_mm",
                    "translation_step_mm",
                    "rotation_step_deg",
                ],
            )
            self.csv_writer.writeheader()

    def note_camera_frame(self) -> None:
        self.camera_frames += 1

    def note_empty_frame(self) -> None:
        self.empty_frames += 1

    def update_detection(
        self,
        camera_frame: int,
        process_ms: float,
        markers,
        poses,
    ) -> None:
        self.processed_frames += 1
        marker_count = len(markers)
        detected = marker_count > 0
        if detected:
            self.detected_frames += 1
        self.total_markers += marker_count
        self.total_processing_ms += process_ms

        valid_poses = [
            pose
            for pose in poses
            if pose.get("marker_id", -1) >= 0
            and as_vector(pose.get("rvec")) is not None
            and as_vector(pose.get("tvec")) is not None
        ]
        pose = valid_poses[0] if valid_poses else None
        pose_valid = pose is not None
        if pose_valid:
            self.pose_frames += 1

        marker_id = -1
        feature_count = 0
        observation_count = 0
        reprojection_error = math.nan
        tvec = None
        rvec = None
        translation_step = math.nan
        rotation_step = math.nan

        if detected:
            primary_marker = markers[0]
            marker_id = int(primary_marker.get("marker_id", -1))
            feature_count = len(primary_marker.get("feature_pos", []))

        if pose_valid:
            marker_id = int(pose.get("marker_id", marker_id))
            observation_count = int(pose.get("observation_count", 0))
            reprojection_error = float(pose.get("reprojection_error", math.nan))
            if math.isfinite(reprojection_error):
                self.reprojection_errors.append(reprojection_error)
            if observation_count > 0:
                self.observation_counts.append(observation_count)

            tvec = as_vector(pose.get("tvec"))
            rvec = as_vector(pose.get("rvec"))
            if tvec is not None:
                self.translation_window.append(tvec)
                if self.previous_tvec is not None:
                    translation_step = float(np.linalg.norm(tvec - self.previous_tvec))
                    self.translation_step_window.append(translation_step)
                self.previous_tvec = tvec
            if rvec is not None:
                if self.previous_rvec is not None:
                    rotation_step = rotation_angle_deg(self.previous_rvec, rvec)
                    self.rotation_step_window.append(rotation_step)
                    if rotation_step > 90.0:
                        self.rotation_jump_count += 1
                self.previous_rvec = rvec

        if self.csv_writer is not None:
            elapsed = time.time() - self.start_time
            self.csv_writer.writerow(
                {
                    "time_s": f"{elapsed:.3f}",
                    "camera_frame": camera_frame,
                    "processed_frame": self.processed_frames,
                    "process_ms": f"{process_ms:.3f}",
                    "detected": int(detected),
                    "pose_valid": int(pose_valid),
                    "marker_count": marker_count,
                    "primary_marker_id": marker_id,
                    "feature_count": feature_count,
                    "observation_count": observation_count,
                    "reprojection_error": (
                        f"{reprojection_error:.6f}"
                        if math.isfinite(reprojection_error)
                        else ""
                    ),
                    "t_x_mm": f"{tvec[0]:.6f}" if tvec is not None else "",
                    "t_y_mm": f"{tvec[1]:.6f}" if tvec is not None else "",
                    "t_z_mm": f"{tvec[2]:.6f}" if tvec is not None else "",
                    "translation_step_mm": (
                        f"{translation_step:.6f}"
                        if math.isfinite(translation_step)
                        else ""
                    ),
                    "rotation_step_deg": (
                        f"{rotation_step:.6f}" if math.isfinite(rotation_step) else ""
                    ),
                }
            )
            self.csv_file.flush()

    def summary(self) -> dict:
        elapsed = max(1e-9, time.time() - self.start_time)
        detection_rate = (
            self.detected_frames / self.processed_frames
            if self.processed_frames
            else 0.0
        )
        pose_rate = (
            self.pose_frames / self.processed_frames if self.processed_frames else 0.0
        )
        avg_process_ms = (
            self.total_processing_ms / self.processed_frames
            if self.processed_frames
            else 0.0
        )
        avg_markers = (
            self.total_markers / self.processed_frames if self.processed_frames else 0.0
        )
        avg_reproj = (
            float(np.mean(self.reprojection_errors)) if self.reprojection_errors else math.nan
        )
        avg_obs = (
            float(np.mean(self.observation_counts)) if self.observation_counts else math.nan
        )

        if len(self.translation_window) >= 2:
            translations = np.vstack(self.translation_window)
            translation_std = np.std(translations, axis=0)
            translation_std_norm = float(np.linalg.norm(translation_std))
        else:
            translation_std = np.array([math.nan, math.nan, math.nan])
            translation_std_norm = math.nan

        translation_step_rms = (
            float(np.sqrt(np.mean(np.square(self.translation_step_window))))
            if self.translation_step_window
            else math.nan
        )
        rotation_step_rms = (
            float(np.sqrt(np.mean(np.square(self.rotation_step_window))))
            if self.rotation_step_window
            else math.nan
        )

        return {
            "elapsed_s": elapsed,
            "camera_frames": self.camera_frames,
            "empty_frames": self.empty_frames,
            "processed_frames": self.processed_frames,
            "detected_frames": self.detected_frames,
            "pose_frames": self.pose_frames,
            "camera_fps": self.camera_frames / elapsed,
            "process_fps": self.processed_frames / elapsed,
            "detection_rate": detection_rate,
            "pose_rate": pose_rate,
            "avg_process_ms": avg_process_ms,
            "avg_markers": avg_markers,
            "avg_reprojection_error": avg_reproj,
            "avg_observation_count": avg_obs,
            "translation_std_x_mm": float(translation_std[0]),
            "translation_std_y_mm": float(translation_std[1]),
            "translation_std_z_mm": float(translation_std[2]),
            "translation_std_norm_mm": translation_std_norm,
            "translation_step_rms_mm": translation_step_rms,
            "rotation_step_rms_deg": rotation_step_rms,
            "rotation_jump_count": self.rotation_jump_count,
        }

    def should_print(self) -> bool:
        if self.print_every_s <= 0:
            return False
        now = time.time()
        if now - self.last_print_time < self.print_every_s:
            return False
        self.last_print_time = now
        return True

    def format_overlay(self):
        summary = self.summary()
        return [
            f"Detect: {summary['detection_rate'] * 100:.1f}% "
            f"Pose: {summary['pose_rate'] * 100:.1f}%",
            f"Proc: {summary['avg_process_ms']:.1f} ms "
            f"Rep: {summary['avg_reprojection_error']:.2f}",
            f"Jitter: {summary['translation_step_rms_mm']:.2f} mm "
            f"{summary['rotation_step_rms_deg']:.1f} deg",
        ]

    def print_summary(self, prefix: str = "Metrics") -> None:
        summary = self.summary()
        print(
            f"{prefix}: "
            f"camera_frames={summary['camera_frames']}, "
            f"processed={summary['processed_frames']}, "
            f"detected={summary['detected_frames']} "
            f"({summary['detection_rate'] * 100:.1f}%), "
            f"pose={summary['pose_frames']} ({summary['pose_rate'] * 100:.1f}%), "
            f"avg_process={summary['avg_process_ms']:.2f}ms, "
            f"avg_reproj={summary['avg_reprojection_error']:.3f}, "
            f"t_step_rms={summary['translation_step_rms_mm']:.3f}mm, "
            f"r_step_rms={summary['rotation_step_rms_deg']:.3f}deg, "
            f"rot_jumps={summary['rotation_jump_count']}"
        )

    def close(self) -> None:
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()


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


def draw_metrics_overlay(image: np.ndarray, lines) -> None:
    x = 20
    y = 78
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 220, 255),
            1,
            cv2.LINE_AA,
        )
        y += 28


def print_devices(device_manager, dev_info_list) -> None:
    for idx, info in enumerate(dev_info_list, start=1):
        vendor = info.get("vendor_name", "")
        model = info.get("model_name", "")
        serial = info.get("sn", "")
        display_name = info.get("display_name", "")
        print(f"[{idx}] {vendor} {model} sn={serial} {display_name}")


def set_if_available(feature, value) -> None:
    try:
        if feature.is_implemented() and feature.is_writable():
            feature.set(value)
    except Exception as exc:
        print(f"Failed to set camera feature: {exc}")


def set_named_feature(owner, name: str, value) -> None:
    try:
        feature = getattr(owner, name)
    except Exception as exc:
        print(f"Camera feature {name} is not available: {exc}")
        return
    set_if_available(feature, value)


def get_int_feature(owner, name: str):
    try:
        feature = getattr(owner, name)
        if feature.is_implemented() and feature.is_readable():
            return int(feature.get())
    except Exception:
        pass
    return None


def align_int_feature_value(feature, requested: int) -> int:
    info = feature.get_range()
    min_value = int(info["min"])
    max_value = int(info["max"])
    inc = max(1, int(info["inc"]))
    value = max(min_value, min(max_value, int(requested)))
    return min_value + ((value - min_value) // inc) * inc


def set_int_feature_aligned(owner, name: str, requested: int) -> Optional[int]:
    try:
        feature = getattr(owner, name)
        if not (feature.is_implemented() and feature.is_writable()):
            return None
        value = align_int_feature_value(feature, requested)
        feature.set(value)
        return value
    except Exception as exc:
        print(f"Failed to set {name}={requested}: {exc}")
        return None


def configure_resolution(cam, args) -> None:
    if args.binning > 1:
        set_int_feature_aligned(cam, "BinningHorizontal", args.binning)
        set_int_feature_aligned(cam, "BinningVertical", args.binning)
    if args.decimation > 1:
        set_int_feature_aligned(cam, "DecimationHorizontal", args.decimation)
        set_int_feature_aligned(cam, "DecimationVertical", args.decimation)

    if args.width <= 0 and args.height <= 0:
        return

    # ROI is safest when offsets are zeroed before shrinking Width/Height.
    set_int_feature_aligned(cam, "OffsetX", 0)
    set_int_feature_aligned(cam, "OffsetY", 0)

    actual_width = None
    actual_height = None
    if args.width > 0:
        actual_width = set_int_feature_aligned(cam, "Width", args.width)
    if args.height > 0:
        actual_height = set_int_feature_aligned(cam, "Height", args.height)

    if args.roi_center:
        sensor_width = get_int_feature(cam, "WidthMax") or get_int_feature(cam, "SensorWidth")
        sensor_height = get_int_feature(cam, "HeightMax") or get_int_feature(cam, "SensorHeight")
        actual_width = actual_width or get_int_feature(cam, "Width")
        actual_height = actual_height or get_int_feature(cam, "Height")
        if sensor_width and actual_width:
            set_int_feature_aligned(cam, "OffsetX", max(0, (sensor_width - actual_width) // 2))
        if sensor_height and actual_height:
            set_int_feature_aligned(cam, "OffsetY", max(0, (sensor_height - actual_height) // 2))

    width = get_int_feature(cam, "Width")
    height = get_int_feature(cam, "Height")
    offset_x = get_int_feature(cam, "OffsetX")
    offset_y = get_int_feature(cam, "OffsetY")
    print(f"Camera ROI: {width}x{height} offset=({offset_x}, {offset_y})")


def set_stream_buffer_mode(stream, mode_name: str) -> None:
    modes = {
        "oldest_first": gx.GxDSStreamBufferHandlingModeEntry.OLDEST_FIRST,
        "oldest_first_overwrite": gx.GxDSStreamBufferHandlingModeEntry.OLDEST_FIRST_OVERWRITE,
        "newest_only": gx.GxDSStreamBufferHandlingModeEntry.NEWEST_ONLY,
    }
    mode = modes.get(mode_name)
    if mode is not None:
        set_if_available(stream.StreamBufferHandlingMode, mode)


def open_daheng_camera(args):
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        raise RuntimeError("No Daheng/Galaxy camera was enumerated.")

    print_devices(device_manager, dev_info_list)
    cam = None
    last_exc = None
    for attempt in range(1, args.open_retries + 1):
        try:
            print(f"Opening Daheng camera index {args.camera_index}, attempt {attempt}...")
            cam = device_manager.open_device_by_index(args.camera_index)
            break
        except Exception as exc:
            last_exc = exc
            print(f"Open attempt {attempt} failed: {exc}")
            time.sleep(0.5)
            device_manager.update_device_list()
    if cam is None:
        raise last_exc if last_exc is not None else RuntimeError("Failed to open camera")

    set_named_feature(cam, "TriggerMode", gx.GxSwitchEntry.OFF)
    if args.exposure_us > 0:
        set_named_feature(cam, "ExposureTime", float(args.exposure_us))
    if args.gain >= 0:
        set_named_feature(cam, "Gain", float(args.gain))
    if args.acquisition_fps > 0:
        set_named_feature(cam, "AcquisitionFrameRateMode", gx.GxSwitchEntry.ON)
        set_named_feature(cam, "AcquisitionFrameRate", float(args.acquisition_fps))
    configure_resolution(cam, args)
    stream = cam.data_stream[0]
    set_stream_buffer_mode(stream, args.buffer_mode)
    if args.stream_transfer_size > 0:
        set_if_available(stream.StreamTransferSize, int(args.stream_transfer_size))
    if args.stream_transfer_urbs > 0:
        set_if_available(stream.StreamTransferNumberUrb, int(args.stream_transfer_urbs))

    return device_manager, cam


def raw_image_to_bgr(cam, raw_image):
    if raw_image is None:
        return None
    if raw_image.get_status() != gx.GxFrameStatusList.SUCCESS:
        return None

    if cam.PixelColorFilter.is_implemented():
        rgb_image = raw_image.convert("RGB")
        if rgb_image is None:
            return None
        numpy_image = rgb_image.get_numpy_array()
        if numpy_image is None:
            return None
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    mono = raw_image.get_numpy_array()
    if mono is None:
        return None
    return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)


def read_counter(feature):
    try:
        if feature.is_implemented() and feature.is_readable():
            return feature.get()
    except Exception:
        pass
    return None


def print_stream_counters(cam) -> None:
    stream = cam.data_stream[0]
    counters = {
        "delivered": read_counter(stream.StreamDeliveredFrameCount),
        "lost": read_counter(stream.StreamLostFrameCount),
        "incomplete": read_counter(stream.StreamIncompleteFrameCount),
    }
    print(
        "Stream counters: "
        + ", ".join(
            f"{name}={value}" for name, value in counters.items() if value is not None
        )
    )


def run(args) -> None:
    metrics_path = None
    if not args.disable_metrics:
        if args.metrics_csv:
            metrics_path = args.metrics_csv
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_path = ASSET_ROOT / f"{timestamp}_daheng_metrics.csv"
        print(f"Writing metrics to {metrics_path}")

    metrics = DetectionMetrics(
        window_size=args.metrics_window,
        csv_path=metrics_path,
        print_every_s=args.metrics_print_every,
    )

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

    try:
        device_manager, cam = open_daheng_camera(args)
    except Exception as exc:
        print(f"Failed to open Daheng camera: {exc}")
        metrics.close()
        return

    video_writer = None
    fps_counter = 0
    frame_counter = 0
    fps_timer = time.time()
    fps_value: Optional[float] = None
    last_markers = []
    last_pose_map = {}

    interrupted = False
    try:
        cam.stream_on()
        while True:
            raw_image = cam.data_stream[0].get_image()
            frame = raw_image_to_bgr(cam, raw_image)
            if frame is None:
                metrics.note_empty_frame()
                continue
            metrics.note_camera_frame()

            if args.flip >= 0:
                frame = cv2.flip(frame, args.flip)

            display = frame.copy()

            if args.record and video_writer is None:
                height, width = display.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_name = f"{timestamp}_{args.output_prefix}"
                fps = args.record_fps if args.record_fps > 0 else 30.0
                video_writer = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
                if video_writer.isOpened():
                    print(f"Recording Daheng video to {output_name}")
                else:
                    print(f"Failed to open video writer for file: {output_name}")
                    video_writer = None

            frame_counter += 1
            should_detect = (
                not args.preview_only
                and args.process_every_n > 0
                and frame_counter % args.process_every_n == 0
            )

            if should_detect:
                process_start = time.perf_counter()
                result = runner.process(frame)
                process_ms = (time.perf_counter() - process_start) * 1000.0
                markers = result["markers"]
                poses = result["poses"]
                last_markers = markers
                last_pose_map = {
                    pose["marker_id"]: pose for pose in poses if pose["marker_id"] >= 0
                }
                metrics.update_detection(frame_counter, process_ms, markers, poses)
                if metrics.should_print():
                    metrics.print_summary("Metrics")

            if not args.preview_only:
                if last_markers:
                    annotate_markers(
                        display,
                        last_markers,
                        last_pose_map,
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

            if not args.disable_metrics:
                draw_metrics_overlay(display, metrics.format_overlay())

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
            cv2.imshow("CylinderTag Daheng Viewer", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC_KEY or key == ord("q"):
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        try:
            cam.stream_off()
        except Exception:
            pass
        try:
            print_stream_counters(cam)
        except Exception:
            pass
        if not args.disable_metrics:
            metrics.print_summary("Final metrics")
            if metrics_path is not None:
                print(f"Metrics CSV: {metrics_path}")
        metrics.close()
        try:
            cam.close_device()
        except Exception:
            pass
        _ = device_manager
        if interrupted:
            print("Stopped by user.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time CylinderTag pose estimation using a Daheng Galaxy camera."
    )
    parser.add_argument("--camera-index", type=int, default=1, help="Galaxy camera index, 1-based.")
    parser.add_argument("--open-retries", type=int, default=3, help="Camera open retry count.")
    parser.add_argument("--width", type=int, default=0, help="Requested camera ROI width.")
    parser.add_argument("--height", type=int, default=0, help="Requested camera ROI height.")
    parser.add_argument("--roi-center", action="store_true", help="Center the requested ROI.")
    parser.add_argument("--binning", type=int, default=1, help="Set horizontal/vertical binning.")
    parser.add_argument("--decimation", type=int, default=1, help="Set horizontal/vertical decimation.")
    parser.add_argument("--exposure-us", type=float, default=10000.0, help="Exposure time in microseconds.")
    parser.add_argument("--gain", type=float, default=10.0, help="Camera gain. Use -1 to leave unchanged.")
    parser.add_argument(
        "--acquisition-fps",
        type=float,
        default=20.0,
        help="Camera acquisition frame rate. Use 0 to leave unchanged.",
    )
    parser.add_argument(
        "--stream-transfer-size",
        type=int,
        default=0,
        help="U3V stream transfer block size. Use 0 to leave unchanged.",
    )
    parser.add_argument(
        "--stream-transfer-urbs",
        type=int,
        default=0,
        help="U3V stream transfer URB count. Use 0 to leave unchanged.",
    )
    parser.add_argument(
        "--buffer-mode",
        choices=["oldest_first", "oldest_first_overwrite", "newest_only"],
        default="newest_only",
        help="Galaxy stream buffer handling mode. newest_only avoids displaying stale frames.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only show the camera stream without running CylinderTag detection.",
    )
    parser.add_argument(
        "--process-every-n",
        type=int,
        default=3,
        help="Run CylinderTag detection every N frames and reuse the latest pose between detections.",
    )
    parser.add_argument("--debug", action="store_true", help="Print traceback for camera open failures.")
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
        help="Camera calibration file for this Daheng camera.",
    )
    parser.add_argument("--adaptive-thresh", type=int, default=5, help="Adaptive threshold window size.")
    parser.add_argument("--disable-subpix", action="store_true", help="Disable sub-pixel corner refinement.")
    parser.add_argument("--subpix-window", type=int, default=5, help="Sub-pixel refinement window radius.")
    parser.add_argument("--axis-length", type=float, default=50.0, help="Axis length in mm.")
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Disable detection/stability metric collection and CSV output.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="CSV path for per-detection metrics. Defaults to timestamp_daheng_metrics.csv.",
    )
    parser.add_argument(
        "--metrics-window",
        type=int,
        default=60,
        help="Rolling pose stability window in processed detection frames.",
    )
    parser.add_argument(
        "--metrics-print-every",
        type=float,
        default=2.0,
        help="Print metric summary every N seconds. Use 0 to disable periodic prints.",
    )
    parser.add_argument("--record", action="store_true", help="Record the annotated stream to disk.")
    parser.add_argument("--record-fps", type=float, default=30.0, help="FPS used for saved video.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_FILENAME,
        help="Base filename for recordings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run(args)
    except Exception:
        if args.debug:
            traceback.print_exc()
        else:
            raise


if __name__ == "__main__":
    main()
