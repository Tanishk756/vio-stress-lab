"""Dataset loaders and multiplexers for EuRoC MAV ASL format and Synthetic 6-DOF trajectories."""

from __future__ import annotations
import csv
from dataclasses import dataclass, field
import heapq
import os
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple, Union
import cv2
import numpy as np

from vio_lab.se3_math import (
    quat_to_rot,
    rot_to_quat,
    so3_exp,
    so3_log,
    transform_point,
)


@dataclass
class ImuSample:
    """IMU measurement at timestamp t."""
    t: float  # Timestamp in seconds
    acc: np.ndarray  # [ax, ay, az] in m/s^2
    gyro: np.ndarray  # [wx, wy, wz] in rad/s


@dataclass
class StereoFrame:
    """Stereo image pair at timestamp t."""
    t: float
    img_l: np.ndarray  # Left grayscale image
    img_r: np.ndarray  # Right grayscale image
    T_wb_gt: Optional[np.ndarray] = None  # Ground truth 4x4 pose (if available)
    # 3D landmarks visible in world frame
    visible_landmarks: Optional[dict[int, np.ndarray]] = None
    # 2D projected subpixel coordinates on left image [u, v]
    visible_keypoints: Optional[dict[int, np.ndarray]] = None



@dataclass
class GroundTruthSample:
    """6-DOF ground truth state at timestamp t."""
    t: float
    p: np.ndarray  # [px, py, pz]
    q: np.ndarray  # [qx, qy, qz, qw]
    v: np.ndarray  # [vx, vy, vz]
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def T_wb(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = quat_to_rot(self.q)
        T[:3, 3] = self.p
        return T


SensorEvent = Union[ImuSample, StereoFrame]


class SyntheticVioDataset:
    """Generates synthetic 6-DOF figure-eight trajectory with 200 Hz IMU and 20 Hz stereo imagery.

    Enables instant offline unit testing, algorithm benchmarking, and fault-injection
    validation without downloading large external datasets.
    """

    def __init__(
        self,
        duration_s: float = 30.0,
        imu_rate_hz: float = 200.0,
        cam_rate_hz: float = 20.0,
        num_landmarks: int = 400,
        seed: int = 42,
        acc_noise_std: float = 0.002,
        gyro_noise_std: float = 0.0002,
        acc_bias_walk_std: float = 0.0001,
        gyro_bias_walk_std: float = 0.00001,
    ):
        self.duration_s = duration_s
        self.imu_rate_hz = imu_rate_hz
        self.cam_rate_hz = cam_rate_hz
        self.dt_imu = 1.0 / imu_rate_hz
        self.dt_cam = 1.0 / cam_rate_hz
        self.seed = seed
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Camera intrinsics (EuRoC pinhole model)
        self.width = 752
        self.height = 480
        self.fx = 458.654
        self.fy = 437.283
        self.cx = 367.215
        self.cy = 248.375
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.baseline = 0.11007  # Stereo baseline in meters

        # Extrinsics: camera frame to body frame (cam0)
        # Cam coords: X right, Y down, Z forward
        # Body coords: X forward, Y left, Z up
        self.R_bc0 = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ], dtype=np.float64)
        self.p_bc0 = np.array([0.05, 0.0, 0.0], dtype=np.float64)
        self.T_bc0 = np.eye(4, dtype=np.float64)
        self.T_bc0[:3, :3] = self.R_bc0
        self.T_bc0[:3, 3] = self.p_bc0

        # Noise settings
        self.acc_noise_std = acc_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.acc_bias_walk_std = acc_bias_walk_std
        self.gyro_bias_walk_std = gyro_bias_walk_std

        # Static background texture
        self.bg_texture = np.zeros((self.height, self.width), dtype=np.uint8)
        # Create subtle smooth gradients and texture
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        self.bg_texture = (120 + 20 * np.sin(10 * xx) * np.cos(10 * yy)).astype(np.uint8)

        # Precompute trajectory and landmarks
        np.random.seed(self.seed)
        self.landmarks = self._generate_landmarks(num_landmarks)
        self.gt_samples: list[GroundTruthSample] = []
        self._precompute_trajectory()


    def _generate_landmarks(self, n: int) -> np.ndarray:
        """Create a 3D point cloud surrounding the trajectory corridor."""
        pts = []
        for _ in range(n):
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(3.5, 7.5)
            z = np.random.uniform(-2.5, 3.5)
            # Surround Lissajous bounds: x in [-3, 3], y in [-2, 2]
            cx = 2.5 * np.cos(theta)
            cy = 2.0 * np.sin(2.0 * theta)
            px = cx + r * np.cos(theta)
            py = cy + r * np.sin(theta)
            pts.append([px, py, z])
        return np.array(pts, dtype=np.float64)

    def _trajectory_pos(self, t: float) -> np.ndarray:
        """Lissajous figure-eight 3D position."""
        ax, ay, az = 2.5, 1.8, 0.4
        wx = 2.0 * np.pi / 12.0
        wy = 2.0 * wx
        wz = wx
        x = ax * np.sin(wx * t)
        y = ay * np.sin(wy * t)
        z = az * np.cos(wz * t)
        return np.array([x, y, z], dtype=np.float64)

    def _precompute_trajectory(self):
        """Precompute smooth analytic states, IMU biases, and ground truth."""
        t_values = np.arange(0.0, self.duration_s + self.dt_imu, self.dt_imu)

        # Initialize biases
        ba = np.array([0.05, -0.04, 0.02], dtype=np.float64)
        bg = np.array([0.002, -0.003, 0.001], dtype=np.float64)

        for t in t_values:
            eps = 1e-4
            p = self._trajectory_pos(t)
            p_next = self._trajectory_pos(t + eps)
            p_prev = self._trajectory_pos(t - eps)

            v = (p_next - p_prev) / (2.0 * eps)
            a_world = (p_next - 2.0 * p + p_prev) / (eps**2)

            # Body orientation: tangent to velocity with banking roll
            vel_norm = np.linalg.norm(v)
            if vel_norm > 1e-3:
                fwd = v / vel_norm
            else:
                fwd = np.array([1.0, 0.0, 0.0])

            up_approx = np.array([0.0, 0.0, 1.0])
            right = np.cross(fwd, up_approx)
            if np.linalg.norm(right) > 1e-3:
                right = right / np.linalg.norm(right)
                up = np.cross(right, fwd)
            else:
                right = np.array([0.0, 1.0, 0.0])
                up = np.array([0.0, 0.0, 1.0])

            R_wb = np.column_stack([fwd, right, up])
            q = rot_to_quat(R_wb)

            # Random-walk bias update
            ba += np.random.normal(0.0, self.acc_bias_walk_std, 3) * np.sqrt(self.dt_imu)
            bg += np.random.normal(0.0, self.gyro_bias_walk_std, 3) * np.sqrt(self.dt_imu)

            self.gt_samples.append(GroundTruthSample(
                t=t, p=p, q=q, v=v, ba=ba.copy(), bg=bg.copy()
            ))

    def _render_stereo_frame(self, t: float, T_wb: np.ndarray) -> StereoFrame:
        """Render realistic stereo images and extract projected landmark positions."""
        R_wb = T_wb[:3, :3]
        p_wb = T_wb[:3, 3]

        # Cam0 pose: T_wc0 = T_wb * T_bc0
        T_wc0 = T_wb @ self.T_bc0
        R_c0w = T_wc0[:3, :3].T
        p_c0w = -R_c0w @ T_wc0[:3, 3]

        # Render left and right images with clean background
        img_l = np.full((self.height, self.width), 25, dtype=np.uint8)
        img_r = np.full((self.height, self.width), 25, dtype=np.uint8)

        visible_dict = {}
        visible_kpts = {}

        for lm_id, p_world in enumerate(self.landmarks):
            # Transform to cam0 frame
            p_c0 = R_c0w @ p_world + p_c0w
            if p_c0[2] < 0.5 or p_c0[2] > 25.0:
                continue

            # Project into left image
            u_l = (self.fx * p_c0[0]) / p_c0[2] + self.cx
            v_l = (self.fy * p_c0[1]) / p_c0[2] + self.cy

            # Project into right image (horizontal disparity = fx * baseline / Z)
            disparity = (self.fx * self.baseline) / p_c0[2]
            u_r = u_l - disparity
            v_r = v_l

            margin = 20
            if (margin <= u_l < self.width - margin and margin <= v_l < self.height - margin and
                    margin <= u_r < self.width - margin and margin <= v_r < self.height - margin):
                visible_dict[lm_id] = p_world
                # Store subpixel keypoint coordinates with slight measurement jitter (0.25 px)
                jitter = np.random.normal(0.0, 0.25, 2)
                visible_kpts[lm_id] = np.array([u_l + jitter[0], v_l + jitter[1]], dtype=np.float32)

                # Draw high-contrast feature patch on both images
                pt_l = (int(round(u_l)), int(round(v_l)))
                pt_r = (int(round(u_r)), int(round(v_r)))

                cv2.circle(img_l, pt_l, 4, 240, -1)
                cv2.circle(img_r, pt_r, 4, 240, -1)

        return StereoFrame(
            t=t,
            img_l=img_l,
            img_r=img_r,
            T_wb_gt=T_wb,
            visible_landmarks=visible_dict,
            visible_keypoints=visible_kpts
        )


    def events(self) -> Generator[SensorEvent, None, None]:
        """Yield timestamp-multiplexed IMU and Stereo events in chronological order."""
        imu_idx = 0
        total_imu = len(self.gt_samples)
        next_cam_t = 0.0

        for i in range(total_imu - 1):
            gt_curr = self.gt_samples[i]
            gt_next = self.gt_samples[i + 1]
            t = gt_curr.t

            # Generate stereo frame if due
            if t >= next_cam_t - 1e-6:
                yield self._render_stereo_frame(t, gt_curr.T_wb)
                next_cam_t += self.dt_cam

            # IMU acceleration: a_m = R_wb^T (a_world - g) + ba + noise
            R_wb = quat_to_rot(gt_curr.q)
            a_world = (gt_next.v - gt_curr.v) / self.dt_imu
            a_body = R_wb.T @ (a_world - self.gravity)
            acc_meas = a_body + gt_curr.ba + np.random.normal(0, self.acc_noise_std, 3)

            # IMU angular velocity: omega_m = omega_body + bg + noise
            R_next = quat_to_rot(gt_next.q)
            delta_R = R_wb.T @ R_next
            omega_body = so3_log(delta_R) / self.dt_imu
            gyro_meas = omega_body + gt_curr.bg + np.random.normal(0, self.gyro_noise_std, 3)

            yield ImuSample(t=t, acc=acc_meas, gyro=gyro_meas)


class EurocAslDataset:
    """Parses standard EuRoC MAV ASL dataset directory structure (e.g., MH_01_easy)."""

    def __init__(self, dataset_path: Union[str, Path]):
        self.root = Path(dataset_path)
        # Handle cases where path is dataset root or mav0 directory
        if (self.root / "mav0").exists():
            self.mav0 = self.root / "mav0"
        else:
            self.mav0 = self.root

        self.imu_csv = self.mav0 / "imu0" / "data.csv"
        self.cam0_csv = self.mav0 / "cam0" / "data.csv"
        self.cam1_csv = self.mav0 / "cam1" / "data.csv"
        self.gt_csv = self.mav0 / "state_groundtruth_estimate0" / "data.csv"

        if not self.imu_csv.exists():
            raise FileNotFoundError(f"EuRoC IMU CSV not found at {self.imu_csv}")

        # Standard EuRoC cam0/cam1 intrinsics and extrinsics
        self.K = np.array([
            [458.654, 0.0, 367.215],
            [0.0, 437.283, 248.375],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.baseline = 0.11007
        self.T_bc0 = np.array([
            [0.01486554, -0.99988093, 0.0041403, -0.02164015],
            [0.99955725, 0.01496721, 0.02571553, -0.06467699],
            [-0.02577444, 0.00375619, 0.99966073, 0.00981073],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self._gt_samples: Optional[list[GroundTruthSample]] = None

    @property
    def gt_samples(self) -> list[GroundTruthSample]:
        if self._gt_samples is None:
            self._gt_samples = self.load_ground_truth()
        return self._gt_samples

    def load_ground_truth(self) -> list[GroundTruthSample]:
        """Load ground truth samples into a list if available."""
        samples: list[GroundTruthSample] = []
        if not self.gt_csv.exists():
            return samples

        with open(self.gt_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                t_sec = float(row[0]) * 1e-9
                p = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=np.float64)
                # EuRoC format: qw, qx, qy, qz -> our convention: [qx, qy, qz, qw]
                q = np.array([float(row[5]), float(row[6]), float(row[7]), float(row[4])], dtype=np.float64)
                v = np.array([float(row[8]), float(row[9]), float(row[10])], dtype=np.float64)
                bg = np.array([float(row[11]), float(row[12]), float(row[13])], dtype=np.float64)
                ba = np.array([float(row[14]), float(row[15]), float(row[16])], dtype=np.float64)
                samples.append(GroundTruthSample(t=t_sec, p=p, q=q, v=v, ba=ba, bg=bg))

        return samples

    def events(self, max_seconds: Optional[float] = None) -> Generator[SensorEvent, None, None]:
        """Yield timestamp-sorted IMU and Stereo events from EuRoC CSV files."""
        # Read IMU records
        imu_records: list[tuple[float, np.ndarray, np.ndarray]] = []
        with open(self.imu_csv, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                t = float(row[0]) * 1e-9
                gyro = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=np.float64)
                acc = np.array([float(row[4]), float(row[5]), float(row[6])], dtype=np.float64)
                imu_records.append((t, acc, gyro))

        # Read Cam0 & Cam1 image mappings
        cam0_records: list[tuple[float, Path]] = []
        cam0_dir = self.mav0 / "cam0" / "data"
        with open(self.cam0_csv, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                t = float(row[0]) * 1e-9
                filename = row[1].strip()
                cam0_records.append((t, cam0_dir / filename))

        cam1_dict: dict[float, Path] = {}
        cam1_dir = self.mav0 / "cam1" / "data"
        if self.cam1_csv.exists():
            with open(self.cam1_csv, "r", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if not row or row[0].startswith("#"):
                        continue
                    t = float(row[0]) * 1e-9
                    cam1_dict[t] = cam1_dir / row[1].strip()

        # Priority queue multiplexing
        # Elements: (timestamp, priority_order, event_type, data)
        # priority_order: 0 for IMU (high rate), 1 for camera
        pq = []
        for t, acc, gyro in imu_records:
            heapq.heappush(pq, (t, 0, "imu", (acc, gyro)))

        for t, path_l in cam0_records:
            path_r = cam1_dict.get(t, None)
            heapq.heappush(pq, (t, 1, "cam", (path_l, path_r)))

        t_start = pq[0][0] if pq else 0.0
        gt_list = self.gt_samples
        gt_times = np.array([g.t for g in gt_list]) if gt_list else None

        while pq:
            t, _, ev_type, data = heapq.heappop(pq)
            if max_seconds is not None and (t - t_start) > max_seconds:
                break

            if ev_type == "imu":
                acc, gyro = data
                yield ImuSample(t=t, acc=acc, gyro=gyro)
            elif ev_type == "cam":
                path_l, path_r = data
                img_l = cv2.imread(str(path_l), cv2.IMREAD_GRAYSCALE)
                img_r = cv2.imread(str(path_r), cv2.IMREAD_GRAYSCALE) if path_r else img_l
                if img_l is not None and img_r is not None:
                    T_gt = None
                    if gt_times is not None and len(gt_times) > 0:
                        idx = int(np.clip(np.searchsorted(gt_times, t), 0, len(gt_times) - 1))
                        if abs(gt_times[idx] - t) < 0.05:
                            T_gt = gt_list[idx].T_wb
                    yield StereoFrame(t=t, img_l=img_l, img_r=img_r, T_wb_gt=T_gt)
