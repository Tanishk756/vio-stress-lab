"""High-Level Visual-Inertial Odometry Pipeline fusing IMU and Stereo Vision."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from vio_lab.dataset import ImuSample, StereoFrame, GroundTruthSample
from vio_lab.eskf import Eskf, EskfConfig
from vio_lab.frontend import StereoKltFrontend, VisualPoseEstimate
from vio_lab.se3_math import nominal_imu_step, quat_to_rot


@dataclass
class TrajectoryRecord:
    """Historical telemetry for evaluation and visualization."""
    timestamps: list[float] = field(default_factory=list)
    poses_eskf: list[np.ndarray] = field(default_factory=list)      # 4x4
    poses_gt: list[np.ndarray] = field(default_factory=list)        # 4x4
    poses_imu_only: list[np.ndarray] = field(default_factory=list)  # 4x4
    poses_vis_only: list[np.ndarray] = field(default_factory=list)  # 4x4
    covariances: list[np.ndarray] = field(default_factory=list)     # 15x15
    chi2_values: list[float] = field(default_factory=list)
    chi2_accepted: list[bool] = field(default_factory=list)
    ba_estimates: list[np.ndarray] = field(default_factory=list)    # [3]
    bg_estimates: list[np.ndarray] = field(default_factory=list)    # [3]
    ba_gt: list[np.ndarray] = field(default_factory=list)           # [3]
    bg_gt: list[np.ndarray] = field(default_factory=list)           # [3]
    vis_estimates: list[VisualPoseEstimate] = field(default_factory=list)


class VioPipeline:
    """Orchestrates high-rate IMU propagation, visual frontend PnP, and ESKF measurement updates."""

    def __init__(
        self,
        eskf: Eskf,
        frontend: StereoKltFrontend,
        degradation_hook: Optional[Callable[[ImuSample | StereoFrame], ImuSample | StereoFrame]] = None
    ):
        self.eskf = eskf
        self.frontend = frontend
        self.degradation_hook = degradation_hook

        # Baseline pure IMU dead-reckoning state (no visual updates)
        self.imu_only_p = self.eskf.p.copy()
        self.imu_only_v = self.eskf.v.copy()
        self.imu_only_q = self.eskf.q.copy()
        self.imu_only_ba = self.eskf.ba.copy()
        self.imu_only_bg = self.eskf.bg.copy()

        # Visual-only last pose
        self.last_vis_T_wb = self.eskf.T_wb.copy()

        self.last_imu_t: Optional[float] = None
        self.records = TrajectoryRecord()

    def process_event(
        self,
        event: ImuSample | StereoFrame,
        gt_sample: Optional[GroundTruthSample] = None
    ) -> None:
        """Process either an IMU sample or a stereo frame."""
        # Apply degradation filter if active
        if self.degradation_hook is not None:
            event = self.degradation_hook(event)

        if isinstance(event, ImuSample):
            self._handle_imu(event)
        elif isinstance(event, StereoFrame):
            self._handle_stereo(event, gt_sample)

    def _handle_imu(self, imu: ImuSample) -> None:
        if self.last_imu_t is None:
            self.last_imu_t = imu.t
            return

        dt = imu.t - self.last_imu_t
        self.last_imu_t = imu.t

        if dt <= 0.0 or dt > 0.5:
            return

        # 1. ESKF IMU propagation
        self.eskf.predict_imu(dt, imu.acc, imu.gyro)

        # 2. Pure IMU dead-reckoning propagation (no corrections)
        self.imu_only_p, self.imu_only_v, self.imu_only_q, _, _ = nominal_imu_step(
            self.imu_only_p, self.imu_only_v, self.imu_only_q,
            self.imu_only_ba, self.imu_only_bg,
            imu.acc, imu.gyro, dt, self.eskf.config.gravity
        )

    def _handle_stereo(
        self,
        frame: StereoFrame,
        gt_sample: Optional[GroundTruthSample]
    ) -> None:
        # Visual frontend tracking
        prior_T_wb = self.eskf.T_wb
        vis = self.frontend.process_frame(
            frame.t, frame.img_l, frame.img_r,
            prior_T_wb=prior_T_wb,
            synthetic_landmarks=frame.visible_landmarks,
            synthetic_keypoints=frame.visible_keypoints
        )


        chi2_val = 0.0
        accepted = False

        if vis.valid:
            self.last_vis_T_wb = vis.T_wb.copy()
            # Perform ESKF measurement update
            accepted, chi2_val = self.eskf.update_pose(vis.T_wb, vis.R_cov)

        # Construct pure IMU pose matrix
        T_imu_only = np.eye(4, dtype=np.float64)
        T_imu_only[:3, :3] = quat_to_rot(self.imu_only_q)
        T_imu_only[:3, 3] = self.imu_only_p

        # Record telemetry
        self.records.timestamps.append(frame.t)
        self.records.poses_eskf.append(self.eskf.T_wb.copy())
        self.records.poses_imu_only.append(T_imu_only)
        self.records.poses_vis_only.append(self.last_vis_T_wb.copy())
        self.records.covariances.append(self.eskf.P.copy())
        self.records.chi2_values.append(chi2_val)
        self.records.chi2_accepted.append(accepted)
        self.records.ba_estimates.append(self.eskf.ba.copy())
        self.records.bg_estimates.append(self.eskf.bg.copy())
        self.records.vis_estimates.append(vis)

        if frame.T_wb_gt is not None:
            self.records.poses_gt.append(frame.T_wb_gt.copy())
        elif gt_sample is not None:
            self.records.poses_gt.append(gt_sample.T_wb.copy())
            self.records.ba_gt.append(gt_sample.ba.copy())
            self.records.bg_gt.append(gt_sample.bg.copy())
        else:
            self.records.poses_gt.append(np.eye(4))
