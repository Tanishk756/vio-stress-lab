"""Sensor Degradation & Fault-Injection Engine for VIO Stress Testing.

Simulates real-world robotics sensor failure modes:
1. Camera Blackout: complete visual loss (tunnel entrance, camera failure, lighting cut).
2. Motion Blur & Outliers: aggressive motion blur + mismatched keypoints.
3. IMU Bias Step Shock: sudden mechanical shock or temperature spike causing bias jump.
4. Camera-IMU Time Desynchronization: uncompensated hardware trigger latency.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np

from vio_lab.dataset import ImuSample, StereoFrame


@dataclass
class DegradationConfig:
    """Configuration parameters for sensor degradation stress testing."""
    # 1. Camera Blackout
    blackout_enabled: bool = False
    blackout_t_start: float = 10.0
    blackout_t_end: float = 13.0

    # 2. Motion Blur & Feature Mismatch Outliers
    motion_blur_enabled: bool = False
    blur_t_start: float = 7.0
    blur_t_end: float = 9.5
    blur_kernel_size: int = 15
    outlier_ratio: float = 0.35  # Fraction of corrupted/mismatched 2D points

    # 3. IMU Bias Step Shock
    imu_bias_shock_enabled: bool = False
    bias_shock_t: float = 18.0
    acc_bias_step: np.ndarray = field(default_factory=lambda: np.array([0.35, -0.20, 0.15], dtype=np.float64))
    gyro_bias_step: np.ndarray = field(default_factory=lambda: np.array([0.03, -0.02, 0.015], dtype=np.float64))

    # 4. Camera-IMU Timestamp Desynchronization
    time_desync_enabled: bool = False
    cam_time_offset_s: float = 0.025  # +25 ms delay


class FaultInjectionEngine:
    """Middleware applying configurable faults to IMU and Stereo streams."""

    def __init__(self, config: Optional[DegradationConfig] = None):
        self.config = config if config is not None else DegradationConfig()
        self._blur_kernel = self._create_motion_blur_kernel(self.config.blur_kernel_size)

    def _create_motion_blur_kernel(self, size: int) -> np.ndarray:
        """Create a directional motion blur convolution kernel."""
        kernel = np.zeros((size, size), dtype=np.float32)
        # Diagonal motion streak
        for i in range(size):
            kernel[i, i] = 1.0
        return kernel / float(size)

    def process_event(self, event: ImuSample | StereoFrame) -> ImuSample | StereoFrame:
        """Apply active sensor degradation mechanisms to the event."""
        if isinstance(event, ImuSample):
            return self._corrupt_imu(event)
        elif isinstance(event, StereoFrame):
            return self._corrupt_stereo(event)
        return event

    def _corrupt_imu(self, imu: ImuSample) -> ImuSample:
        # Check IMU Bias Step Shock
        if self.config.imu_bias_shock_enabled and imu.t >= self.config.bias_shock_t:
            acc_corrupted = imu.acc + self.config.acc_bias_step
            gyro_corrupted = imu.gyro + self.config.gyro_bias_step
            return ImuSample(t=imu.t, acc=acc_corrupted, gyro=gyro_corrupted)
        return imu

    def _corrupt_stereo(self, frame: StereoFrame) -> StereoFrame:
        t = frame.t
        img_l = frame.img_l.copy()
        img_r = frame.img_r.copy()
        visible_lm = frame.visible_landmarks.copy() if frame.visible_landmarks is not None else None
        visible_kpts = frame.visible_keypoints.copy() if frame.visible_keypoints is not None else None

        # 1. Camera-IMU Time Desync
        if self.config.time_desync_enabled:
            t += self.config.cam_time_offset_s

        # 2. Camera Blackout
        if self.config.blackout_enabled and (self.config.blackout_t_start <= frame.t <= self.config.blackout_t_end):
            # Zero image intensity and invalidate all features
            img_l.fill(0)
            img_r.fill(0)
            visible_lm = {}
            visible_kpts = {}
            return StereoFrame(
                t=t,
                img_l=img_l,
                img_r=img_r,
                T_wb_gt=frame.T_wb_gt,
                visible_landmarks=visible_lm,
                visible_keypoints=visible_kpts
            )

        # 3. Motion Blur & Outliers
        if self.config.motion_blur_enabled and (self.config.blur_t_start <= frame.t <= self.config.blur_t_end):
            # Apply directional blur convolution
            img_l = cv2.filter2D(img_l, -1, self._blur_kernel)
            img_r = cv2.filter2D(img_r, -1, self._blur_kernel)

            # Corrupt feature observations with gross mismatched outliers
            if visible_kpts is not None and len(visible_kpts) > 0:
                kpt_ids = list(visible_kpts.keys())
                num_outliers = int(len(kpt_ids) * self.config.outlier_ratio)
                outlier_ids = np.random.choice(kpt_ids, size=num_outliers, replace=False)

                for oid in outlier_ids:
                    # Inject gross 40-80 px false match displacement
                    err = np.random.uniform(40.0, 80.0, size=2) * np.random.choice([-1, 1], size=2)
                    visible_kpts[oid] = visible_kpts[oid] + err

        return StereoFrame(
            t=t,
            img_l=img_l,
            img_r=img_r,
            T_wb_gt=frame.T_wb_gt,
            visible_landmarks=visible_lm,
            visible_keypoints=visible_kpts
        )
