"""Multi-State Constraint Kalman Filter (MSCKF) Implementation.

Based on Mourikis & Roumeliotis (2007): "A Multi-State Constraint Kalman Filter
for Vision-aided Inertial Navigation".

Eliminates 3D landmark positions from the filter state via QR nullspace projection:
    r_j = H_x * delta_x + H_f * delta_p_f + n
    Q_2^T * H_f = 0  ==>  Q_2^T * r_j = Q_2^T * H_x * delta_x + Q_2^T * n
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from vio_lab.eskf import Eskf, EskfConfig
from vio_lab.se3_math import (
    compute_error_state_jacobian_fx,
    nominal_imu_step,
    quat_mult,
    quat_normalize,
    quat_to_rot,
    rot_to_quat,
    skew,
    so3_exp_quat,
    so3_log,
)


@dataclass
class CameraClone:
    """Cloned historical camera pose in world frame."""
    clone_id: int
    t: float
    p: np.ndarray  # [px, py, pz]
    q: np.ndarray  # [qx, qy, qz, qw]

    @property
    def R(self) -> np.ndarray:
        return quat_to_rot(self.q)


@dataclass
class FeatureObservation:
    """Multi-view visual observations of a persistent 3D landmark."""
    feature_id: int
    # Map from clone_id to normalized camera coordinates [x/z, y/z]
    observations: dict[int, np.ndarray] = field(default_factory=dict)


class MsckfFilter:
    """MSCKF fusing high-rate IMU with sliding-window multi-state constraint updates."""

    def __init__(
        self,
        config: Optional[EskfConfig] = None,
        max_clones: int = 10,
        T_bc: Optional[np.ndarray] = None,
        init_p: Optional[np.ndarray] = None,
        init_v: Optional[np.ndarray] = None,
        init_q: Optional[np.ndarray] = None,
        init_ba: Optional[np.ndarray] = None,
        init_bg: Optional[np.ndarray] = None,
    ):
        self.config = config if config is not None else EskfConfig()
        self.max_clones = max_clones

        # Extrinsics: camera in body frame
        self.T_bc = np.eye(4) if T_bc is None else np.asarray(T_bc, dtype=np.float64)
        self.R_bc = self.T_bc[:3, :3]
        self.p_bc = self.T_bc[:3, 3]

        # Nominal IMU state
        self.p = np.zeros(3, dtype=np.float64) if init_p is None else np.asarray(init_p, dtype=np.float64).copy()
        self.v = np.zeros(3, dtype=np.float64) if init_v is None else np.asarray(init_v, dtype=np.float64).copy()
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64) if init_q is None else quat_normalize(init_q)
        self.ba = np.zeros(3, dtype=np.float64) if init_ba is None else np.asarray(init_ba, dtype=np.float64).copy()
        self.bg = np.zeros(3, dtype=np.float64) if init_bg is None else np.asarray(init_bg, dtype=np.float64).copy()

        # Sliding window of cloned camera poses
        self.clones: list[CameraClone] = []
        self.next_clone_id = 0

        # State covariance: 15 (IMU) + 6 * num_clones
        self.P = np.zeros((15, 15), dtype=np.float64)
        self.P[0:3, 0:3] = np.eye(3) * self.config.init_pos_cov
        self.P[3:6, 3:6] = np.eye(3) * self.config.init_vel_cov
        self.P[6:9, 6:9] = np.eye(3) * self.config.init_rot_cov
        self.P[9:12, 9:12] = np.eye(3) * self.config.init_ba_cov
        self.P[12:15, 12:15] = np.eye(3) * self.config.init_bg_cov

    @property
    def R(self) -> np.ndarray:
        return quat_to_rot(self.q)

    @property
    def T_wb(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.p
        return T

    def predict_imu(self, dt: float, acc_m: np.ndarray, gyro_m: np.ndarray) -> None:
        """Propagate nominal IMU kinematics and total covariance matrix."""
        R_k = self.R
        acc_hat = acc_m - self.ba
        gyro_hat = gyro_m - self.bg

        # Nominal kinematics step
        self.p, self.v, self.q, self.ba, self.bg = nominal_imu_step(
            self.p, self.v, self.q, self.ba, self.bg,
            acc_m, gyro_m, dt, self.config.gravity
        )

        # Discrete error transition for IMU block (15 x 15)
        Fx = compute_error_state_jacobian_fx(R_k, acc_hat, gyro_hat, dt)

        # Process noise
        sigma_a2 = self.config.acc_noise_density ** 2
        sigma_g2 = self.config.gyro_noise_density ** 2
        sigma_ba2 = self.config.acc_random_walk ** 2
        sigma_bg2 = self.config.gyro_random_walk ** 2

        Q_imu = np.zeros((15, 15), dtype=np.float64)
        Q_imu[0:3, 0:3] = np.eye(3) * (0.25 * sigma_a2 * (dt ** 3) + 1e-10)
        Q_imu[3:6, 3:6] = np.eye(3) * (sigma_a2 * dt)
        Q_imu[6:9, 6:9] = np.eye(3) * (sigma_g2 * dt)
        Q_imu[9:12, 9:12] = np.eye(3) * (sigma_ba2 * dt)
        Q_imu[12:15, 12:15] = np.eye(3) * (sigma_bg2 * dt)

        # Propagate total covariance:
        # P_II = Fx * P_II * Fx^T + Q_imu
        # P_IC = Fx * P_IC
        # P_CC remains unchanged
        total_dim = self.P.shape[0]
        self.P[0:15, 0:15] = Fx @ self.P[0:15, 0:15] @ Fx.T + Q_imu

        if total_dim > 15:
            self.P[0:15, 15:] = Fx @ self.P[0:15, 15:]
            self.P[15:, 0:15] = self.P[0:15, 15:].T

        self.P = 0.5 * (self.P + self.P.T)

    def augment_state(self, t: float) -> int:
        """Clone current camera pose into the filter state vector."""
        # Camera pose in world frame:
        # p_c = p_b + R_b * p_bc
        # q_c = q_b (x) q_bc
        R_b = self.R
        p_c = self.p + R_b @ self.p_bc
        q_bc = rot_to_quat(self.R_bc)
        q_c = quat_normalize(quat_mult(self.q, q_bc))

        clone_id = self.next_clone_id
        self.next_clone_id += 1
        self.clones.append(CameraClone(clone_id=clone_id, t=t, p=p_c, q=q_c))

        # Augmentation Jacobian (6 x 15):
        # [delta_p_c] = I * delta_p - R_b * [p_bc]x * delta_theta
        # [delta_theta_c] = R_bc^T * delta_theta
        J_clone = np.zeros((6, 15), dtype=np.float64)
        J_clone[0:3, 0:3] = np.eye(3)
        J_clone[0:3, 6:9] = -R_b @ skew(self.p_bc)
        J_clone[3:6, 6:9] = self.R_bc.T

        # Augment covariance matrix
        cur_dim = self.P.shape[0]
        new_P = np.zeros((cur_dim + 6, cur_dim + 6), dtype=np.float64)
        new_P[:cur_dim, :cur_dim] = self.P

        # Cross-covariance between existing states and new clone
        P_cross = self.P[:, :15] @ J_clone.T
        new_P[:cur_dim, cur_dim:] = P_cross
        new_P[cur_dim:, :cur_dim] = P_cross.T

        # Covariance of new clone
        new_P[cur_dim:, cur_dim:] = J_clone @ self.P[:15, :15] @ J_clone.T

        self.P = 0.5 * (new_P + new_P.T)

        # Marginalize oldest clone if window size exceeded
        if len(self.clones) > self.max_clones:
            self._marginalize_oldest_clone()

        return clone_id

    def _marginalize_oldest_clone(self) -> None:
        """Prune oldest camera clone and slice out its rows/cols from covariance."""
        if not self.clones:
            return
        self.clones.pop(0)
        # Oldest clone occupies indices [15:21]
        keep_indices = list(range(0, 15)) + list(range(21, self.P.shape[0]))
        self.P = self.P[np.ix_(keep_indices, keep_indices)]

    def update_multi_view_features(
        self,
        features: list[FeatureObservation],
        meas_noise_std: float = 0.005
    ) -> bool:
        """Perform nullspace projected MSCKF measurement update across all completed feature tracks."""
        if not features or not self.clones:
            return False

        clone_id_to_idx = {c.clone_id: idx for idx, c in enumerate(self.clones)}
        total_dim = self.P.shape[0]

        r_stacked = []
        H_stacked = []

        for feat in features:
            obs = feat.observations
            valid_clone_ids = [cid for cid in obs.keys() if cid in clone_id_to_idx]
            if len(valid_clone_ids) < 3:
                continue

            # 1. Triangulate landmark position p_f in world frame via linear SVD
            A_rows = []
            for cid in valid_clone_ids:
                c = self.clones[clone_id_to_idx[cid]]
                Rc = c.R
                pc = c.p
                uv = obs[cid]
                # Projection matrix row equations:
                # u * (row2) - row0 = 0
                # v * (row2) - row1 = 0
                R_cw = Rc.T
                t_cw = -R_cw @ pc
                P_mat = np.column_stack([R_cw, t_cw])
                A_rows.append(uv[0] * P_mat[2, :] - P_mat[0, :])
                A_rows.append(uv[1] * P_mat[2, :] - P_mat[1, :])

            _, _, Vt = np.linalg.svd(np.array(A_rows))
            X_hom = Vt[-1]
            if np.abs(X_hom[3]) < 1e-8:
                continue
            p_f = (X_hom[:3] / X_hom[3]).reshape(3)

            # 2. Form linearized residuals and Jacobians
            r_feat = []
            H_x_feat = []
            H_f_feat = []

            for cid in valid_clone_ids:
                c_idx = clone_id_to_idx[cid]
                c = self.clones[c_idx]
                Rc = c.R
                pc = c.p

                # Point in camera frame: p_c = Rc^T * (p_f - pc)
                p_c = Rc.T @ (p_f - pc)
                if p_c[2] < 0.2:
                    continue

                z_inv = 1.0 / p_c[2]
                z_inv2 = z_inv * z_inv

                # Normalized coordinates
                z_pred = np.array([p_c[0] * z_inv, p_c[1] * z_inv])
                z_meas = obs[cid]
                residual = z_meas - z_pred
                r_feat.append(residual)

                # Jacobian of normalized coords w.r.t point in camera coords (2 x 3)
                J_proj = np.array([
                    [z_inv, 0.0, -p_c[0] * z_inv2],
                    [0.0, z_inv, -p_c[1] * z_inv2]
                ])

                # Jacobian w.r.t landmark position (2 x 3): J_proj * Rc^T
                H_f_i = J_proj @ Rc.T
                H_f_feat.append(H_f_i)

                # Jacobian w.r.t cloned camera pose [delta_p_ci, delta_theta_ci] (2 x 6):
                # dp_c / d(delta_p_c) = -Rc^T
                # dp_c / d(delta_theta_c) = Rc^T * [p_f - pc]x
                H_c_i = np.zeros((2, 6))
                H_c_i[:, 0:3] = -J_proj @ Rc.T
                H_c_i[:, 3:6] = J_proj @ Rc.T @ skew(p_f - pc)

                # Place into total state Jacobian (2 x total_dim)
                H_x_row = np.zeros((2, total_dim))
                clone_state_start = 15 + c_idx * 6
                H_x_row[:, clone_state_start:clone_state_start + 6] = H_c_i
                H_x_feat.append(H_x_row)

            if len(r_feat) < 3:
                continue

            r_mat = np.concatenate(r_feat)         # 2M x 1
            H_x_mat = np.vstack(H_x_feat)          # 2M x total_dim
            H_f_mat = np.vstack(H_f_feat)          # 2M x 3

            # 3. QR Nullspace Projection eliminating landmark Jacobian H_f
            # H_f = [Q1, Q2] * [R1; 0]
            # Q2^T * H_f = 0
            Q, _ = np.linalg.qr(H_f_mat, mode="complete")
            Q2 = Q[:, 3:]  # Left nullspace of H_f (2M x (2M - 3))

            r_proj = Q2.T @ r_mat                  # (2M - 3)
            H_proj = Q2.T @ H_x_mat                # (2M - 3) x total_dim

            r_stacked.append(r_proj)
            H_stacked.append(H_proj)

        if not r_stacked:
            return False

        r_total = np.concatenate(r_stacked)
        H_total = np.vstack(H_stacked)

        # Measurement noise for projected residual: R_proj = sigma^2 * I
        R_noise = (meas_noise_std ** 2) * np.eye(len(r_total))

        # Innovation covariance
        S = H_total @ self.P @ H_total.T + R_noise
        S = 0.5 * (S + S.T)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ H_total.T @ S_inv
        delta_x = K @ r_total

        # 4. Error state injection:
        # Update IMU state
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        dq = so3_exp_quat(delta_x[6:9])
        self.q = quat_normalize(quat_mult(self.q, dq))
        self.ba += delta_x[9:12]
        self.bg += delta_x[12:15]

        # Update cloned camera poses
        for idx, c in enumerate(self.clones):
            c_start = 15 + idx * 6
            c.p += delta_x[c_start:c_start + 3]
            dq_c = so3_exp_quat(delta_x[c_start + 3:c_start + 6])
            c.q = quat_normalize(quat_mult(c.q, dq_c))

        # 5. Joseph-form covariance update
        IKH = np.eye(total_dim) - K @ H_total
        self.P = IKH @ self.P @ IKH.T + K @ R_noise @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True
