"""Rerun 3D Telemetry Dashboard and Multimodal Recording Logger."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import rerun as rr

from vio_lab.dataset import StereoFrame, GroundTruthSample
from vio_lab.eskf import Eskf
from vio_lab.frontend import VisualPoseEstimate
from vio_lab.pipeline import TrajectoryRecord


class RerunVisualizer:
    """Logs multimodal 3D state estimation telemetry, camera views, and error scalars to Rerun."""

    def __init__(
        self,
        app_id: str = "vio-stress-lab",
        recording_id: str = "vio_stress_demo",
        save_path: Optional[str | Path] = None,
        spawn_viewer: bool = False
    ):
        self.app_id = app_id
        self.save_path = Path(save_path) if save_path is not None else None

        # Initialize rerun recording
        rr.init(self.app_id, spawn=spawn_viewer)

        # Set up coordinates: Z up, X right, Y forward or standard robotics
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Buffers for continuous trajectory line strips
        self.traj_gt: list[list[float]] = []
        self.traj_eskf: list[list[float]] = []
        self.traj_vis: list[list[float]] = []
        self.traj_imu: list[list[float]] = []

    def log_frame(
        self,
        t: float,
        frame: StereoFrame,
        eskf: Eskf,
        vis: VisualPoseEstimate,
        imu_only_p: Optional[np.ndarray] = None,
        gt_p: Optional[np.ndarray] = None,
        chi2_val: float = 0.0,
        accepted: bool = True
    ) -> None:
        """Log a synchronized sensor and filter time-step to Rerun."""
        rr.set_time("stable_time", duration=t)


        # 1. Trajectory positions
        p_eskf = eskf.p.tolist()
        self.traj_eskf.append(p_eskf)
        rr.log("world/eskf_vio", rr.LineStrips3D([self.traj_eskf], colors=[[0, 229, 255, 255]]))

        if gt_p is not None:
            self.traj_gt.append(gt_p.tolist())
            rr.log("world/ground_truth", rr.LineStrips3D([self.traj_gt], colors=[[0, 255, 100, 255]]))

        if vis.valid:
            self.traj_vis.append(vis.T_wb[:3, 3].tolist())
            rr.log("world/vision_only", rr.LineStrips3D([self.traj_vis], colors=[[255, 145, 0, 200]]))

        if imu_only_p is not None:
            self.traj_imu.append(imu_only_p.tolist())
            rr.log("world/imu_dead_reckoning", rr.LineStrips3D([self.traj_imu], colors=[[255, 23, 68, 220]]))

        # 2. 3-Sigma position uncertainty ellipsoid
        radii, _ = eskf.get_3sigma_position_ellipsoid()
        rr.log(
            "world/eskf_vio/uncertainty_3sigma",
            rr.Ellipsoids3D(
                half_sizes=[radii.tolist()],
                centers=[p_eskf],
                colors=[[0, 229, 255, 90]]
            )
        )

        # 3. 3D Map / Landmarked points
        if len(vis.tracked_pts_3d) > 0:
            rr.log(
                "world/landmarks",
                rr.Points3D(
                    vis.tracked_pts_3d,
                    colors=[[255, 235, 59, 180]],
                    radii=0.03
                )
            )

        # 4. Camera 2D Image & Tracked Keypoints
        rr.log("cam0/image", rr.Image(frame.img_l))
        if len(vis.tracked_pts_2d) > 0:
            color = [0, 255, 0, 255] if accepted else [255, 0, 0, 255]
            rr.log(
                "cam0/image/tracked_features",
                rr.Points2D(vis.tracked_pts_2d, colors=[color], radii=2.5)
            )

        # 5. Telemetry & Error Metrics
        rr.log("telemetry/chi2", rr.Scalars([chi2_val]))
        rr.log("telemetry/inliers", rr.Scalars([vis.num_inliers]))
        rr.log("telemetry/inlier_ratio", rr.Scalars([vis.inlier_ratio]))

        # Estimated IMU biases
        rr.log("telemetry/bias_acc_x", rr.Scalars([eskf.ba[0]]))
        rr.log("telemetry/bias_acc_y", rr.Scalars([eskf.ba[1]]))
        rr.log("telemetry/bias_acc_z", rr.Scalars([eskf.ba[2]]))
        rr.log("telemetry/bias_gyro_x", rr.Scalars([eskf.bg[0]]))
        rr.log("telemetry/bias_gyro_y", rr.Scalars([eskf.bg[1]]))
        rr.log("telemetry/bias_gyro_z", rr.Scalars([eskf.bg[2]]))

        if gt_p is not None:
            pos_err = float(np.linalg.norm(eskf.p - gt_p))
            rr.log("telemetry/position_error_m", rr.Scalars([pos_err]))

    def log_pipeline_record(self, records: TrajectoryRecord, save_path: Optional[str | Path] = None) -> None:
        """Stream an entire TrajectoryRecord into Rerun and save if path specified."""
        dest = save_path if save_path is not None else self.save_path

        for i, t in enumerate(records.timestamps):
            rr.set_time("stable_time", duration=t)


            p_eskf = records.poses_eskf[i][:3, 3]
            self.traj_eskf.append(p_eskf.tolist())
            rr.log("world/eskf_vio", rr.LineStrips3D([self.traj_eskf], colors=[[0, 229, 255, 255]]))

            if i < len(records.poses_gt):
                p_gt = records.poses_gt[i][:3, 3]
                self.traj_gt.append(p_gt.tolist())
                rr.log("world/ground_truth", rr.LineStrips3D([self.traj_gt], colors=[[0, 255, 100, 255]]))
                rr.log("telemetry/position_error_m", rr.Scalars([float(np.linalg.norm(p_eskf - p_gt))]))

            if i < len(records.poses_imu_only):
                p_imu = records.poses_imu_only[i][:3, 3]
                self.traj_imu.append(p_imu.tolist())
                rr.log("world/imu_dead_reckoning", rr.LineStrips3D([self.traj_imu], colors=[[255, 23, 68, 220]]))

            if i < len(records.poses_vis_only):
                p_vis = records.poses_vis_only[i][:3, 3]
                self.traj_vis.append(p_vis.tolist())
                rr.log("world/vision_only", rr.LineStrips3D([self.traj_vis], colors=[[255, 145, 0, 200]]))

            # 3-sigma position ellipsoid from recorded covariance
            if i < len(records.covariances):
                P_pos = records.covariances[i][0:3, 0:3]
                eigvals = np.maximum(np.linalg.eigvalsh(P_pos), 1e-12)
                radii = 3.0 * np.sqrt(eigvals)
                rr.log(
                    "world/eskf_vio/uncertainty_3sigma",
                    rr.Ellipsoids3D(
                        half_sizes=[radii.tolist()],
                        centers=[p_eskf.tolist()],
                        colors=[[0, 229, 255, 90]]
                    )
                )

            # Vis estimates & scalars
            if i < len(records.vis_estimates):
                vis = records.vis_estimates[i]
                rr.log("telemetry/inliers", rr.Scalars([vis.num_inliers]))
                rr.log("telemetry/inlier_ratio", rr.Scalars([vis.inlier_ratio]))

            if i < len(records.chi2_values):
                rr.log("telemetry/chi2", rr.Scalars([records.chi2_values[i]]))

            if i < len(records.ba_estimates):
                ba = records.ba_estimates[i]
                rr.log("telemetry/bias_acc_x", rr.Scalars([ba[0]]))
                rr.log("telemetry/bias_acc_y", rr.Scalars([ba[1]]))
                rr.log("telemetry/bias_acc_z", rr.Scalars([ba[2]]))

            if i < len(records.bg_estimates):
                bg = records.bg_estimates[i]
                rr.log("telemetry/bias_gyro_x", rr.Scalars([bg[0]]))
                rr.log("telemetry/bias_gyro_y", rr.Scalars([bg[1]]))
                rr.log("telemetry/bias_gyro_z", rr.Scalars([bg[2]]))

        if dest is not None:
            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(dest_path))
            print(f"Rerun recording saved to {dest_path}")
