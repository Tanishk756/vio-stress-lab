"""3D Rigid Body Kinematics & Lie Group SO(3) / SE(3) Math.

Following Joan Solà (2017): "Quaternion kinematics for the error-state Kalman filter".
Tangent space perturbations use the local (right) body-frame formulation:
    R_true = R @ Exp(delta_theta)
    q_true = q (x) Exp(delta_theta / 2)
"""

from __future__ import annotations
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Compute the 3x3 skew-symmetric matrix [v]x of a 3D vector v."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) Lie algebra (R^3) to SO(3) Lie group (3x3 rotation matrix).

    Uses Rodrigues' formula with high-order Taylor series for small angles to prevent
    numerical cancellation and division by zero.
    """
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta_sq = float(np.dot(phi, phi))
    theta = np.sqrt(theta_sq)
    K = skew(phi)

    if theta < 1e-7:
        # Taylor series:
        # sin(theta)/theta = 1 - theta^2 / 6 + theta^4 / 120
        # (1 - cos(theta))/theta^2 = 1/2 - theta^2 / 24 + theta^4 / 720
        a = 1.0 - theta_sq / 6.0 + (theta_sq**2) / 120.0
        b = 0.5 - theta_sq / 24.0 + (theta_sq**2) / 720.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / theta_sq

    return np.eye(3, dtype=np.float64) + a * K + b * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map from SO(3) (3x3 rotation matrix) to so(3) (R^3 vector).

    Accurately handles theta -> 0 and theta -> pi singularities.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    # cos(theta) = (tr(R) - 1) / 2
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-7:
        # Small angle approximation:
        # (R - R^T) / (2 * sin(theta)) * theta -> (R - R^T) / 2 * (1 + theta^2 / 6)
        scale = 0.5 * (1.0 + theta * theta / 6.0)
        return np.array([
            (R[2, 1] - R[1, 2]) * scale,
            (R[0, 2] - R[2, 0]) * scale,
            (R[1, 0] - R[0, 1]) * scale
        ], dtype=np.float64)

    if np.pi - theta < 1e-4:
        # Singularity near pi: (R + I) / 2 = u u^T
        # Find column of (R + I) with largest diagonal entry
        B = (R + np.eye(3, dtype=np.float64)) * 0.5
        diag_idx = int(np.argmax(np.diag(B)))
        v = B[:, diag_idx]
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-8:
            axis = v / v_norm
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return axis * theta

    scale = theta / (2.0 * np.sin(theta))
    return np.array([
        (R[2, 1] - R[1, 2]) * scale,
        (R[0, 2] - R[2, 0]) * scale,
        (R[1, 0] - R[0, 1]) * scale
    ], dtype=np.float64)


def left_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3): J_l(phi).

    Maps tangent space variations to Lie algebra: Exp(phi + delta_phi) ~= Exp(J_l * delta_phi) @ Exp(phi)
    """
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta_sq = float(np.dot(phi, phi))
    theta = np.sqrt(theta_sq)
    K = skew(phi)

    if theta < 1e-7:
        a = 0.5 - theta_sq / 24.0
        b = 1.0 / 6.0 - theta_sq / 120.0
    else:
        a = (1.0 - np.cos(theta)) / theta_sq
        b = (theta - np.sin(theta)) / (theta_sq * theta)

    return np.eye(3, dtype=np.float64) + a * K + b * (K @ K)


def right_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3): J_r(phi) = J_l(-phi)."""
    return left_jacobian_so3(-np.asarray(phi, dtype=np.float64).reshape(3))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion [x, y, z, w]. Ensure w >= 0 for uniqueness."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q_norm = q / norm
    if q_norm[3] < 0.0:
        q_norm = -q_norm
    return q_norm


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q = q1 (x) q2 with convention [x, y, z, w]."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)

    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dtype=np.float64)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse / conjugate of unit quaternion [x, y, z, w]."""
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [x, y, z, w] to 3x3 rotation matrix R."""
    q = quat_normalize(q)
    x, y, z, w = q

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)]
    ], dtype=np.float64)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix R to unit quaternion [x, y, z, w] via Shepperd's algorithm."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = np.trace(R)

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0  # S = 4 * w
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # S = 4 * x
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # S = 4 * y
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # S = 4 * z
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return quat_normalize(np.array([x, y, z, w], dtype=np.float64))


def so3_exp_quat(phi: np.ndarray) -> np.ndarray:
    """Exponential map directly to unit quaternion [x, y, z, w]."""
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(phi)
    if theta < 1e-7:
        scale = 0.5 - (theta**2) / 48.0
        w = 1.0 - (theta**2) / 8.0
        return quat_normalize(np.array([phi[0] * scale, phi[1] * scale, phi[2] * scale, w], dtype=np.float64))
    else:
        half_theta = 0.5 * theta
        scale = np.sin(half_theta) / theta
        return quat_normalize(np.array([phi[0] * scale, phi[1] * scale, phi[2] * scale, np.cos(half_theta)], dtype=np.float64))


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Transform 3D point p by 4x4 matrix T: p' = R @ p + t."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return T[:3, :3] @ p + T[:3, 3]


def nominal_imu_step(
    p: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    ba: np.ndarray,
    bg: np.ndarray,
    acc_m: np.ndarray,
    gyro_m: np.ndarray,
    dt: float,
    gravity: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=np.float64)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate nominal state [p, v, q, ba, bg] by dt using IMU readings."""
    acc_hat = acc_m - ba
    gyro_hat = gyro_m - bg

    R = quat_to_rot(q)
    acc_world = R @ acc_hat + gravity

    # Nominal kinematics integration
    p_next = p + v * dt + 0.5 * acc_world * (dt**2)
    v_next = v + acc_world * dt

    # Orientation integration via quaternion product
    delta_q = so3_exp_quat(gyro_hat * dt)
    q_next = quat_normalize(quat_mult(q, delta_q))

    # Biases follow random walk; nominal change is 0
    ba_next = ba.copy()
    bg_next = bg.copy()

    return p_next, v_next, q_next, ba_next, bg_next


def compute_error_state_jacobian_fx(
    R: np.ndarray,
    acc_hat: np.ndarray,
    gyro_hat: np.ndarray,
    dt: float
) -> np.ndarray:
    """Compute 15x15 error-state discrete transition Jacobian F_x.

    State: [delta_p(3), delta_v(3), delta_theta(3), delta_ba(3), delta_bg(3)]
    Convention: local body-frame error R_true = R @ Exp(delta_theta).
    """
    Fx = np.eye(15, dtype=np.float64)

    # delta_p row
    Fx[0:3, 3:6] = np.eye(3, dtype=np.float64) * dt
    Fx[0:3, 6:9] = -0.5 * R @ skew(acc_hat) * (dt**2)
    Fx[0:3, 9:12] = -0.5 * R * (dt**2)

    # delta_v row
    Fx[3:6, 6:9] = -R @ skew(acc_hat) * dt
    Fx[3:6, 9:12] = -R * dt

    # delta_theta row: Exp(-gyro_hat * dt)
    # Note: Exp(-w*dt) = Exp(w*dt)^T
    delta_R = so3_exp(-gyro_hat * dt)
    Fx[6:9, 6:9] = delta_R
    # d(delta_theta_next) / d(delta_bg) = -J_r(gyro_hat * dt) * dt
    Fx[6:9, 12:15] = -right_jacobian_so3(gyro_hat * dt) * dt

    # delta_ba, delta_bg rows are identity (already set by eye(15))
    return Fx


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert 4x4 rigid transformation matrix: T^-1 = [R^T, -R^T @ t; 0, 1]."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


