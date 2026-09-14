"""RGB color-tag detector coupled to the axis-locked pick/place sequencer.

This is an experiment/prototype for the optional RGB track.  It uses only the
128x128 scene image and the 26D proprioceptive state; it deliberately does not
read parcel actors, tags, or bin poses from the simulator.  The camera is fixed
by :mod:`warehouse_sort.env`, so a plane-ray calibration is sufficient to turn
the colored tag centroids into table XY coordinates.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np
import torch

from .axis_locked_sequencer import AxisLockedSequencer, SequencerTarget


# Values are the fixed 128x128 scene camera in warehouse_sort.env.  A caller can
# pass the matrices returned by ``scene_camera.get_params()`` to the detector if
# the environment camera is changed.
DEFAULT_K = np.array(
    [[117.151215, 0.0, 64.0], [0.0, 117.151215, 64.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DEFAULT_EXTRINSIC_CV = np.array(
    [[0.0, 1.0, 0.0, 0.0],
     [0.7926, 0.0, -0.6097, 0.0305],
     [-0.6097, 0.0, -0.7926, 0.8597]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RGBTarget:
    color: int
    xy: tuple[float, float]
    pixel: tuple[float, float]
    area: int


class RGBParcelDetector:
    """Detect colored parcel tags and infer their world XY locations."""

    def __init__(
        self,
        intrinsic: Optional[np.ndarray] = None,
        extrinsic_cv: Optional[np.ndarray] = None,
        tag_z: float = 0.0625,
        # The colored top tag is mounted at local (-.012,+.012).  Pixel
        # centroids are biased by the visible rectangle, so this calibrated
        # correction is intentionally a little different from that nominal
        # offset.  On the fixed camera it leaves <~6 mm error over the hard
        # spawn zone.
        tag_to_parcel_xy: tuple[float, float] = (0.016, -0.008),
        bin_half_y: float = 0.36,
    ):
        self.K = np.asarray(intrinsic if intrinsic is not None else DEFAULT_K, dtype=np.float64)
        self.E = np.asarray(extrinsic_cv if extrinsic_cv is not None else DEFAULT_EXTRINSIC_CV, dtype=np.float64)
        if self.K.shape == (1, 3, 3):
            self.K = self.K[0]
        if self.E.shape == (1, 3, 4):
            self.E = self.E[0]
        self._R = self.E[:, :3]
        self._t = self.E[:, 3]
        self._R_inv = self._R.T
        self._camera_center = -self._R_inv @ self._t
        self._K_inv = np.linalg.inv(self.K)
        self.tag_z = float(tag_z)
        self.tag_to_parcel_xy = np.asarray(tag_to_parcel_xy, dtype=np.float64)
        self.bin_half_y = float(bin_half_y)

    def pixel_to_world(self, uv: Iterable[float], z: float) -> np.ndarray:
        """Intersect a camera ray with the horizontal world plane ``z``."""
        ray_cam = self._K_inv @ np.array([float(uv[0]), float(uv[1]), 1.0])
        ray_world = self._R_inv @ ray_cam
        lam = (float(z) - self._camera_center[2]) / ray_world[2]
        return self._camera_center + lam * ray_world

    @staticmethod
    def _components(mask: np.ndarray, area_min: int, area_max: int):
        _, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for st, c in zip(stats[1:], centroids[1:]):
            area = int(st[cv2.CC_STAT_AREA])
            if area_min <= area <= area_max:
                out.append((float(c[0]), float(c[1]), area))
        return out

    def _mask_components(self, frame_rgb: np.ndarray, color: int):
        hsv = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2HSV)
        if color == 0:  # red; hue can wrap, although this scene stays near 0
            mask = cv2.inRange(hsv, np.array([0, 60, 45], np.uint8), np.array([12, 255, 255], np.uint8))
        elif color == 1:  # blue
            mask = cv2.inRange(hsv, np.array([90, 30, 30], np.uint8), np.array([140, 255, 255], np.uint8))
        else:  # pragma: no cover
            raise ValueError(color)
        return self._components(mask, area_min=3, area_max=100)

    def _mask(self, frame_rgb: np.ndarray, color: int) -> np.ndarray:
        hsv = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2HSV)
        if color == 0:
            lo, hi = np.array([0, 60, 45], np.uint8), np.array([12, 255, 255], np.uint8)
        elif color == 1:
            lo, hi = np.array([90, 30, 30], np.uint8), np.array([140, 255, 255], np.uint8)
        else:  # pragma: no cover
            raise ValueError(color)
        return cv2.inRange(hsv, lo, hi)

    def detect_bins(self, frame_rgb: np.ndarray) -> dict[int, float]:
        """Return ``color -> pixel-u`` for the two large colored bin regions."""
        result: dict[int, float] = {}
        for color in (0, 1):
            _, _, stats, centroids = cv2.connectedComponentsWithStats(self._mask(frame_rgb, color), 8)
            candidates = [
                (int(st[cv2.CC_STAT_AREA]), float(c[0]))
                for st, c in zip(stats[1:], centroids[1:])
                if int(st[cv2.CC_STAT_AREA]) >= 300
            ]
            if candidates:
                result[color] = max(candidates)[1]
        return result

    def detect(self, frame_rgb: np.ndarray) -> list[RGBTarget]:
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"expected HxWx3 RGB frame, got {frame.shape}")
        targets: list[RGBTarget] = []
        for color in (0, 1):
            for u, v, area in self._mask_components(frame, color):
                p = self.pixel_to_world((u, v), self.tag_z)
                p[:2] += self.tag_to_parcel_xy
                targets.append(RGBTarget(color, (float(p[0]), float(p[1])), (u, v), area))
        return targets

    def bin_target(self, color: int, bin_pixel_u: float) -> tuple[float, float]:
        """Return a bin XY from its visible centroid's horizontal side.

        The task randomizes only which side each color occupies; bin XY itself
        is fixed at (0,+/-0.36).  Calling code may use a detected large color
        component's centroid for ``bin_pixel_u``.
        """
        return (0.0, self.bin_half_y if bin_pixel_u >= self.K[0, 2] else -self.bin_half_y)


class RGBAxisLockedPolicy:
    """Target-lock wrapper: detect once on reset, then run one target at a time."""

    def __init__(self, detector: Optional[RGBParcelDetector] = None, **sequencer_kwargs):
        self.detector = detector or RGBParcelDetector()
        self.sequencer_kwargs = sequencer_kwargs
        self.sequencer = AxisLockedSequencer(**sequencer_kwargs)
        self.targets: list[SequencerTarget] = []
        self._locked_index = 0

    def reset(self, frame_rgb: np.ndarray, bin_pixel_u: Optional[dict[int, float]] = None):
        detected = self.detector.detect(frame_rgb)
        # Default side inference is color-specific only through the large bin
        # component.  If no bin centroids are supplied, the known hard/easy
        # default is used; callers should pass the measured centroids for swaps.
        bin_pixel_u = {**self.detector.detect_bins(frame_rgb), **(bin_pixel_u or {})}
        # This fallback is only for synthetic/unit-test frames with no visible
        # bin; normal WarehouseSort frames always supply both large components.
        bin_pixel_u = {0: 110.0, 1: 16.0, **bin_pixel_u}
        self.targets = [
            SequencerTarget(t.xy, self.detector.bin_target(t.color, bin_pixel_u[t.color]))
            for t in detected
        ]
        # Greedy route ordering accounts for the fact that the next source is
        # reached from the *previous bin*, not from the origin.  This tends to
        # group same-side colors and avoids spending a whole route crossing
        # from +Y to -Y after every parcel.
        remaining = list(self.targets)
        ordered: list[SequencerTarget] = []
        current = (0.0, 0.0)
        while remaining:
            pick = min(
                remaining,
                key=lambda t: (
                    abs(current[0] - t.parcel_xy[0])
                    + abs(current[1] - t.parcel_xy[1])
                    + abs(t.parcel_xy[0] - t.bin_xy[0])
                    + abs(t.parcel_xy[1] - t.bin_xy[1])
                ),
            )
            ordered.append(pick)
            remaining.remove(pick)
            current = pick.bin_xy
        self.targets = ordered
        self._locked_index = 0
        self.sequencer.reset()

    @property
    def done(self) -> bool:
        return self._locked_index >= len(self.targets)

    def act(self, frame_rgb: np.ndarray, state: torch.Tensor) -> torch.Tensor:
        if self.done:
            return torch.zeros(4, dtype=torch.float32, device=state.device)
        if self.sequencer.target is None:
            self.sequencer.set_target(self.targets[self._locked_index])
        action = self.sequencer.act(state)
        if self.sequencer.completed:
            self._locked_index += 1
            self.sequencer.reset()
        return action
