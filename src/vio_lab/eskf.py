"""15-State Error-State Kalman Filter (ESKF) for Visual-Inertial Odometry.

Formulation follows Joan Solà (2017): "Quaternion kinematics for the error-state Kalman filter".
Nominal State (16-D):
    x = [p (3), v (3), q (4), ba (3), bg (3)]
Error State (15-D):
    delta_x = [delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), delta_bg (3)]
Local orientation error convention:
    R_true = R @ Exp(delta_theta)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

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
class EskfConfig:
    """Noise parameters and initial covariances for ESKF."""
    acc_noise_density: float = 2.0e-3     # sigma_a [m/s^2 / sqrt(Hz)]
    gyro_noise_density: float = 1.6968e-4  # sigma_g [rad/s / sqrt(Hz)]
    acc_random_walk: float = 3.0e-3       # sigma_ba [m/s^3 / sqrt(Hz)]
    gyro_random_walk: float = 1.9393e-5   # sigma_bg [rad/s^2 / sqrt(Hz)]

    init_pos_cov: float = 1e-4
    init_vel_cov: float = 1e-3
    init_rot_cov: float = 1e-4
    init_ba_cov: float = 1e-3
    init_bg_cov: float = 1e-4

    chi2_threshold_6dof: float = 12.592    # 95% threshold for 6-DOF Chi-Square
    gravity: np.ndarray = None

    def __post_init__(self):
        if self.gravity is None:
            self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)


class Eskf:
    """15-State Error-State Kalman Filter fusing 200 Hz IMU with 20 Hz visual pose updates."""

    def __init__(
        self,
        config: Optional[EskfConfig] = None,
        init_p: Optional[np.ndarray] = None,
        init_v: Optional[np.ndarray] = None,
        init_q: Optional[np.ndarray] = None,
        init_ba: Optional[np.ndarray] = None,
        init_bg: Optional[np.ndarray] = None,
    ):
        self.config = config if config is not None else EskfConfig()

        # Nominal state: p, v, q, ba, bg
        self.p = np.zeros(3, dtype=np.float64) if init_p is None else np.asarray(init_p, dtype=np.float64).copy()
        self.v = np.zeros(3, dtype=np.float64) if init_v is None else np.asarray(init_v, dtype=np.float64).copy()
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64) if init_q is None else quat_normalize(init_q)
        self.ba = np.zeros(3, dtype=np.float64) if init_ba is None else np.asarray(init_ba, dtype=np.float64).copy()
        self.bg = np.zeros(3, dtype=np.float64) if init_bg is None else np.asarray(init_bg, dtype=np.float64).copy()

        # 15x15 Error-state covariance matrix
        self.P = np.zeros((15, 15), dtype=np.float64)
        self.P[0:3, 0:3] = np.eye(3) * self.config.init_pos_cov
        self.P[3:6, 3:6] = np.eye(3) * self.config.init_vel_cov
        self.P[6:9, 6:9] = np.eye(3) * self.config.init_rot_cov
        self.P[9:12, 9:12] = np.eye(3) * self.config.init_ba_cov
        self.P[12:15, 12:15] = np.eye(3) * self.config.init_bg_cov

        self.last_t: Optional[float] = None

    @property
    def R(self) -> np.ndarray:
        """Current 3x3 rotation matrix of body in world frame (R_wb)."""
        return quat_to_rot(self.q)

    @property
    def T_wb(self) -> np.ndarray:
        """Current 4x4 SE(3) pose matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.p
        return T

    def predict_imu(self, dt: float, acc_m: np.ndarray, gyro_m: np.ndarray) -> None:
        """Propagate nominal kinematics and error-state covariance over time step dt.

        Args:
            dt: Time delta in seconds.
            acc_m: Raw 3D accelerometer measurement [m/s^2].
            gyro_m: Raw 3D gyroscope measurement [rad/s].
        """
        acc_m = np.asarray(acc_m, dtype=np.float64).reshape(3)
        gyro_m = np.asarray(gyro_m, dtype=np.float64).reshape(3)

        R_k = self.R
        acc_hat = acc_m - self.ba
        gyro_hat = gyro_m - self.bg

        # 1. Propagate nominal state
        self.p, self.v, self.q, self.ba, self.bg = nominal_imu_step(
            self.p, self.v, self.q, self.ba, self.bg,
            acc_m, gyro_m, dt, self.config.gravity
        )

        # 2. Discrete state transition Jacobian Fx
        Fx = compute_error_state_jacobian_fx(R_k, acc_hat, gyro_hat, dt)

        # 3. Discrete process noise covariance Q_k
        # Continuous spectral densities:
        # sigma_a^2, sigma_g^2, sigma_ba^2, sigma_bg^2
        sigma_a2 = self.config.acc_noise_density ** 2
        sigma_g2 = self.config.gyro_noise_density ** 2
        sigma_ba2 = self.config.acc_random_walk ** 2
        sigma_bg2 = self.config.gyro_random_walk ** 2

        Qk = np.zeros((15, 15), dtype=np.float64)
        # Position noise: 1/4 sigma_a^2 dt^4 + small regularizer
        Qk[0:3, 0:3] = np.eye(3) * (0.25 * sigma_a2 * (dt ** 3) + 1e-10)
        # Velocity noise: sigma_a^2 dt
        Qk[3:6, 3:6] = np.eye(3) * (sigma_a2 * dt)
        # Orientation noise: sigma_g^2 dt
        Qk[6:9, 6:9] = np.eye(3) * (sigma_g2 * dt)
        # Bias random walk: sigma_b^2 dt
        Qk[9:12, 9:12] = np.eye(3) * (sigma_ba2 * dt)
        Qk[12:15, 12:15] = np.eye(3) * (sigma_bg2 * dt)

        # 4. Covariance propagation: P = Fx * P * Fx^T + Qk
        self.P = Fx @ self.P @ Fx.T + Qk
        # Enforce exact symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def update_pose(
        self,
        T_wb_meas: np.ndarray,
        R_cov: np.ndarray,
        chi2_threshold: Optional[float] = None
    ) -> tuple[bool, float]:
        """Perform 6-DOF pose measurement update with Chi-Square Mahalanobis gating.

        Args:
            T_wb_meas: 4x4 measured pose matrix.
            R_cov: 6x6 measurement covariance matrix.
            chi2_threshold: Chi-Square gating threshold (defaults to config value).

        Returns:
            Tuple of (accepted: bool, mahalanobis_d2: float).
        """
        if chi2_threshold is None:
            chi2_threshold = self.config.chi2_threshold_6dof

        p_meas = T_wb_meas[:3, 3]
        R_meas = T_wb_meas[:3, :3]

        # Innovation:
        # Position residual: r_p = p_meas - p
        r_p = p_meas - self.p

        # Orientation residual: R_meas = R @ Exp(delta_theta) -> delta_theta = Log(R^T @ R_meas)
        R_k = self.R
        delta_R = R_k.T @ R_meas
        r_theta = so3_log(delta_R)

        # 6-D residual vector
        r = np.concatenate([r_p, r_theta])

        # Measurement Jacobian H (6 x 15):
        # r_p depends on delta_p; r_theta depends on delta_theta
        H = np.zeros((6, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3, dtype=np.float64)
        H[3:6, 6:9] = np.eye(3, dtype=np.float64)

        # Innovation covariance: S = H P H^T + R_cov
        S = H @ self.P @ H.T + R_cov
        S = 0.5 * (S + S.T)

        # Mahalanobis distance: d^2 = r^T * S^-1 * r
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(r.T @ S_inv @ r)
        except np.linalg.LinAlgError:
            return False, 999.0

        # Outlier rejection check
        if d2 > chi2_threshold:
            return False, d2

        # Kalman Gain: K = P * H^T * S^-1 (15 x 6)
        K = self.P @ H.T @ S_inv

        # Error state correction
        delta_x = K @ r  # 15-D

        # 1. Error state injection into nominal state
        delta_p = delta_x[0:3]
        delta_v = delta_x[3:6]
        delta_theta = delta_x[6:9]
        delta_ba = delta_x[9:12]
        delta_bg = delta_x[12:15]

        self.p += delta_p
        self.v += delta_v

        # Manifold orientation update: q <- q (x) Exp(delta_theta)
        dq = so3_exp_quat(delta_theta)
        self.q = quat_normalize(quat_mult(self.q, dq))

        self.ba += delta_ba
        self.bg += delta_bg

        # 2. Reset Jacobian G:
        # Because delta_x is reset to 0 after injection:
        # G = I_15 with G[6:9, 6:9] = I - 0.5 * skew(delta_theta)
        G = np.eye(15, dtype=np.float64)
        G[6:9, 6:9] = np.eye(3, dtype=np.float64) - 0.5 * skew(delta_theta)

        # 3. Joseph-form covariance update:
        # P = G * [(I - K*H) * P * (I - K*H)^T + K * R * K^T] * G^T
        IKH = np.eye(15, dtype=np.float64) - K @ H
        P_joseph = IKH @ self.P @ IKH.T + K @ R_cov @ K.T
        self.P = G @ P_joseph @ G.T
        self.P = 0.5 * (self.P + self.P.T)

        return True, d2

    def get_3sigma_position_ellipsoid(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the 3-sigma position uncertainty semi-axes and rotation matrix.

        Returns:
            Tuple of (radii: np.ndarray [3], rot_matrix: np.ndarray [3, 3])
        """
        P_pos = self.P[0:3, 0:3]
        eigenvalues, eigenvectors = np.linalg.eigh(P_pos)
        # Ensure positive
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        # 3-sigma semi-axis lengths: 3 * sqrt(lambda)
        radii = 3.0 * np.sqrt(eigenvalues)
        return radii, eigenvectors
