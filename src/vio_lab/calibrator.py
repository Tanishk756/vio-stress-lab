"""22-State Online Camera-IMU Temporal & Spatial Calibrator (Augmented ESKF).

Estimates online:
- 15-State Kinematics & Biases: [p(3), v(3), q(4), ba(3), bg(3)]
- 6-State Extrinsic Spatial Transform T_BC: [p_bc(3), q_bc(4)]
- 1-State Camera-to-IMU Time Delay: t_d (seconds)

Error State (22-D):
    delta_x = [delta_p(3), delta_v(3), delta_theta(3), delta_ba(3), delta_bg(3),
               delta_p_bc(3), delta_theta_bc(3), delta_td(1)]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from vio_lab.eskf import EskfConfig
from vio_lab.se3_math import (
    compute_error_state_jacobian_fx,
    nominal_imu_step,
    quat_mult,
    quat_normalize,
    quat_to_rot,
    rot_to_quat,
    skew,
    so3_exp,
    so3_exp_quat,
    so3_log,
)


@dataclass
class CalibratorConfig(EskfConfig):
    """Configuration including extrinsic and time-delay process noises and initial covariances."""
    init_extrinsic_pos_cov: float = 1e-4
    init_extrinsic_rot_cov: float = 1e-4
    init_td_cov: float = 1e-4

    extrinsic_pos_walk: float = 1e-6    # m / s / sqrt(Hz) (rigidity drift)
    extrinsic_rot_walk: float = 1e-6    # rad / s / sqrt(Hz)
    td_random_walk: float = 1e-5        # s / s / sqrt(Hz) (clock drift)


class AugmentedEskf:
    """22-State Error-State Kalman Filter with Online Extrinsics & Time-Delay Estimation."""

    def __init__(
        self,
        config: Optional[CalibratorConfig] = None,
        init_p: Optional[np.ndarray] = None,
        init_v: Optional[np.ndarray] = None,
        init_q: Optional[np.ndarray] = None,
        init_ba: Optional[np.ndarray] = None,
        init_bg: Optional[np.ndarray] = None,
        init_p_bc: Optional[np.ndarray] = None,
        init_q_bc: Optional[np.ndarray] = None,
        init_td: float = 0.0,
    ):
        self.config = config if config is not None else CalibratorConfig()

        # 1. Kinematics state
        self.p = np.zeros(3, dtype=np.float64) if init_p is None else np.asarray(init_p, dtype=np.float64).copy()
        self.v = np.zeros(3, dtype=np.float64) if init_v is None else np.asarray(init_v, dtype=np.float64).copy()
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64) if init_q is None else quat_normalize(init_q)
        self.ba = np.zeros(3, dtype=np.float64) if init_ba is None else np.asarray(init_ba, dtype=np.float64).copy()
        self.bg = np.zeros(3, dtype=np.float64) if init_bg is None else np.asarray(init_bg, dtype=np.float64).copy()

        # 2. Calibration state
        self.p_bc = np.zeros(3, dtype=np.float64) if init_p_bc is None else np.asarray(init_p_bc, dtype=np.float64).copy()
        self.q_bc = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64) if init_q_bc is None else quat_normalize(init_q_bc)
        self.td = float(init_td)

        # Last raw IMU acceleration and gyro for time-offset coupling
        self.last_acc_hat = np.zeros(3, dtype=np.float64)
        self.last_gyro_hat = np.zeros(3, dtype=np.float64)

        # 3. 22x22 Covariance matrix
        self.P = np.zeros((22, 22), dtype=np.float64)
        self.P[0:3, 0:3] = np.eye(3) * self.config.init_pos_cov
        self.P[3:6, 3:6] = np.eye(3) * self.config.init_vel_cov
        self.P[6:9, 6:9] = np.eye(3) * self.config.init_rot_cov
        self.P[9:12, 9:12] = np.eye(3) * self.config.init_ba_cov
        self.P[12:15, 12:15] = np.eye(3) * self.config.init_bg_cov
        self.P[15:18, 15:18] = np.eye(3) * self.config.init_extrinsic_pos_cov
        self.P[18:21, 18:21] = np.eye(3) * self.config.init_extrinsic_rot_cov
        self.P[21, 21] = self.config.init_td_cov

    @property
    def R(self) -> np.ndarray:
        """Body-to-world rotation matrix (R_wb)."""
        return quat_to_rot(self.q)

    @property
    def R_bc(self) -> np.ndarray:
        """Cam-to-body rotation matrix (R_bc)."""
        return quat_to_rot(self.q_bc)

    @property
    def T_wb(self) -> np.ndarray:
        """4x4 body pose in world frame."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.p
        return T

    @property
    def T_bc(self) -> np.ndarray:
        """4x4 camera-to-body extrinsic transform."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R_bc
        T[:3, 3] = self.p_bc
        return T

    @property
    def T_wc(self) -> np.ndarray:
        """4x4 camera pose in world frame: T_wc = T_wb * T_bc."""
        return self.T_wb @ self.T_bc

    def predict_imu(self, dt: float, acc_m: np.ndarray, gyro_m: np.ndarray) -> None:
        """Propagate nominal kinematics and 22x22 error covariance."""
        acc_m = np.asarray(acc_m, dtype=np.float64).reshape(3)
        gyro_m = np.asarray(gyro_m, dtype=np.float64).reshape(3)

        R_k = self.R
        acc_hat = acc_m - self.ba
        gyro_hat = gyro_m - self.bg
        self.last_acc_hat = acc_hat
        self.last_gyro_hat = gyro_hat

        # 1. Propagate nominal state
        self.p, self.v, self.q, self.ba, self.bg = nominal_imu_step(
            self.p, self.v, self.q, self.ba, self.bg,
            acc_m, gyro_m, dt, self.config.gravity
        )

        # 2. 22x22 Discrete state transition Jacobian
        Fx = np.eye(22, dtype=np.float64)
        Fx_15 = compute_error_state_jacobian_fx(R_k, acc_hat, gyro_hat, dt)
        Fx[0:15, 0:15] = Fx_15

        # 3. Process noise covariance Q_k
        sigma_a2 = self.config.acc_noise_density ** 2
        sigma_g2 = self.config.gyro_noise_density ** 2
        sigma_ba2 = self.config.acc_random_walk ** 2
        sigma_bg2 = self.config.gyro_random_walk ** 2

        Qk = np.zeros((22, 22), dtype=np.float64)
        Qk[0:3, 0:3] = np.eye(3) * (0.25 * sigma_a2 * (dt ** 3) + 1e-10)
        Qk[3:6, 3:6] = np.eye(3) * (sigma_a2 * dt)
        Qk[6:9, 6:9] = np.eye(3) * (sigma_g2 * dt)
        Qk[9:12, 9:12] = np.eye(3) * (sigma_ba2 * dt)
        Qk[12:15, 12:15] = np.eye(3) * (sigma_bg2 * dt)
        Qk[15:18, 15:18] = np.eye(3) * ((self.config.extrinsic_pos_walk ** 2) * dt)
        Qk[18:21, 18:21] = np.eye(3) * ((self.config.extrinsic_rot_walk ** 2) * dt)
        Qk[21, 21] = (self.config.td_random_walk ** 2) * dt

        # 4. Covariance update
        self.P = Fx @ self.P @ Fx.T + Qk
        self.P = 0.5 * (self.P + self.P.T)

    def update_camera_pose(
        self,
        T_wc_meas: np.ndarray,
        R_cov: np.ndarray,
        chi2_threshold: Optional[float] = None
    ) -> tuple[bool, float]:
        """Measurement update observing 6-DOF camera pose in world frame.

        Calculates Jacobians with respect to body pose, extrinsics, and time delay.
        """
        if chi2_threshold is None:
            chi2_threshold = self.config.chi2_threshold_6dof

        R_wb = self.R
        R_bc = self.R_bc

        # Predicted camera position at exposure time (t + td):
        # p_c_pred = p + v * td + R_wb * p_bc
        p_c_pred = self.p + self.v * self.td + R_wb @ self.p_bc
        # Predicted camera rotation:
        # R_c_pred = R_wb @ Exp(omega_b * td) @ R_bc
        R_wb_td = R_wb @ so3_exp(self.last_gyro_hat * self.td)
        R_c_pred = R_wb_td @ R_bc

        # Measured camera pose
        p_c_meas = T_wc_meas[:3, 3]
        R_c_meas = T_wc_meas[:3, :3]

        # Residuals
        r_p = p_c_meas - p_c_pred
        delta_R = R_c_pred.T @ R_c_meas
        r_theta = so3_log(delta_R)
        r = np.concatenate([r_p, r_theta])

        # 6 x 22 Measurement Jacobian H:
        # State: [delta_p(0:3), delta_v(3:6), delta_theta(6:9), delta_ba(9:12), delta_bg(12:15),
        #         delta_p_bc(15:18), delta_theta_bc(18:21), delta_td(21)]
        H = np.zeros((6, 22), dtype=np.float64)

        # Position residual derivatives:
        H[0:3, 0:3] = np.eye(3)                                 # d(r_p)/d(delta_p)
        H[0:3, 3:6] = np.eye(3) * self.td                       # d(r_p)/d(delta_v)
        H[0:3, 6:9] = -R_wb @ skew(self.p_bc)                   # d(r_p)/d(delta_theta)
        H[0:3, 15:18] = R_wb                                    # d(r_p)/d(delta_p_bc)
        H[0:3, 21] = self.v                                     # d(r_p)/d(delta_td)

        # Orientation residual derivatives:
        # delta_theta_c ~= R_bc^T * R_td^T * delta_theta + delta_theta_bc
        R_td = so3_exp(self.last_gyro_hat * self.td)
        H[3:6, 6:9] = R_bc.T @ R_td.T                           # d(r_theta)/d(delta_theta)
        H[3:6, 18:21] = np.eye(3)                               # d(r_theta)/d(delta_theta_bc)
        H[3:6, 21] = R_bc.T @ self.last_gyro_hat                # d(r_theta)/d(delta_td)


        # Innovation covariance
        S = H @ self.P @ H.T + R_cov
        S = 0.5 * (S + S.T)

        try:
            S_inv = np.linalg.inv(S)
            d2 = float(r.T @ S_inv @ r)
        except np.linalg.LinAlgError:
            return False, 999.0

        if d2 > chi2_threshold:
            return False, d2

        # Kalman gain (22 x 6)
        K = self.P @ H.T @ S_inv
        delta_x = K @ r

        # Inject into nominal state
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        dq = so3_exp_quat(delta_x[6:9])
        self.q = quat_normalize(quat_mult(self.q, dq))
        self.ba += delta_x[9:12]
        self.bg += delta_x[12:15]

        # Extrinsics update
        self.p_bc += delta_x[15:18]
        dq_bc = so3_exp_quat(delta_x[18:21])
        self.q_bc = quat_normalize(quat_mult(self.q_bc, dq_bc))

        # Time delay update
        self.td += float(delta_x[21])

        # Joseph-form covariance update
        IKH = np.eye(22, dtype=np.float64) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_cov @ K.T

        # Reset Jacobian for orientation manifolds
        G = np.eye(22, dtype=np.float64)
        G[6:9, 6:9] = np.eye(3) - 0.5 * skew(delta_x[6:9])
        G[18:21, 18:21] = np.eye(3) - 0.5 * skew(delta_x[18:21])
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)

        return True, d2
