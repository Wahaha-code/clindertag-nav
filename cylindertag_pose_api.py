from __future__ import annotations

import importlib.machinery
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ASSET_ROOT = Path(__file__).resolve().parent
BUILD_ROOT = ASSET_ROOT / "build"
if BUILD_ROOT.exists():
    sys.path.insert(0, str(BUILD_ROOT))

try:
    from cylindertag_cpp import CylinderTagRunner
except ModuleNotFoundError as exc:
    built_modules = sorted(BUILD_ROOT.glob("cylindertag_cpp*.so")) if BUILD_ROOT.exists() else []
    suffixes = ", ".join(importlib.machinery.EXTENSION_SUFFIXES)
    found = ", ".join(path.name for path in built_modules) or "none"
    raise ModuleNotFoundError(
        "Cannot import cylindertag_cpp. Build the Python extension for the "
        f"current Python ({sys.version.split()[0]}), or run with the Python "
        f"version matching the existing module. Expected suffixes: {suffixes}. "
        f"Found in build/: {found}."
    ) from exc


@dataclass(frozen=True)
class TagPose:
    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray
    transform_camera_tag: np.ndarray
    reprojection_error: float
    observation_count: int

    def as_dict(self) -> dict:
        return {
            "marker_id": self.marker_id,
            "rvec": self.rvec.copy(),
            "tvec": self.tvec.copy(),
            "rotation_matrix": self.rotation_matrix.copy(),
            "transform_camera_tag": self.transform_camera_tag.copy(),
            "reprojection_error": self.reprojection_error,
            "observation_count": self.observation_count,
        }


class CylinderTagPoseEstimator:
    """Image-in, pose-out CylinderTag interface.

    The returned transform follows OpenCV solvePnP convention:
    X_camera = R_camera_tag * X_tag + t_camera_tag.
    Units are the same as the model file, normally millimeters.
    """

    def __init__(
        self,
        marker_path: Path | str = ASSET_ROOT / "CTag_2f12c.marker",
        model_path: Path | str = ASSET_ROOT / "CTag_2f12c_d32.model",
        camera_path: Path | str = ASSET_ROOT / "cameraParams.yml",
        adaptive_thresh: int = 5,
        enable_subpix: bool = True,
        subpix_dist: int = 5,
    ) -> None:
        self.runner = CylinderTagRunner(
            str(marker_path),
            str(model_path),
            str(camera_path),
            adaptive_thresh=adaptive_thresh,
            enable_subpix=enable_subpix,
            subpix_dist=subpix_dist,
        )

    def detect(self, image: np.ndarray, marker_id: Optional[int] = None) -> Optional[TagPose]:
        """Return the selected tag pose in the camera frame, or None if not found."""
        if image is None:
            raise ValueError("image must not be None")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.dtype != np.uint8:
            raise TypeError("image must be uint8")
        if image.ndim not in (2, 3):
            raise ValueError("image must be HxW grayscale or HxWx3 BGR/RGB")

        result = self.runner.process(np.ascontiguousarray(image))
        poses = [
            pose
            for pose in result["poses"]
            if int(pose.get("marker_id", -1)) >= 0
        ]
        if marker_id is not None:
            poses = [pose for pose in poses if int(pose["marker_id"]) == marker_id]
        if not poses:
            return None

        pose = min(
            poses,
            key=lambda item: float(item.get("reprojection_error", float("inf"))),
        )
        rvec = np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(pose["tvec"], dtype=np.float64).reshape(3, 1)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3:4] = tvec

        return TagPose(
            marker_id=int(pose["marker_id"]),
            rvec=rvec,
            tvec=tvec,
            rotation_matrix=rotation_matrix,
            transform_camera_tag=transform,
            reprojection_error=float(pose.get("reprojection_error", -1.0)),
            observation_count=int(pose.get("observation_count", 0)),
        )


_DEFAULT_ESTIMATOR: Optional[CylinderTagPoseEstimator] = None


def detect_tag_pose(image: np.ndarray, marker_id: Optional[int] = None) -> Optional[TagPose]:
    """Convenience function: pass one image, get tag pose in camera frame."""
    global _DEFAULT_ESTIMATOR
    if _DEFAULT_ESTIMATOR is None:
        _DEFAULT_ESTIMATOR = CylinderTagPoseEstimator()
    return _DEFAULT_ESTIMATOR.detect(image, marker_id=marker_id)
