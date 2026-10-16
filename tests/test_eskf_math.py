"""Unit tests for SO(3) Lie algebra, Hamilton quaternions, and ESKF error-state Jacobians."""

import numpy as np
import pytest
from vio_lab.se3_math import (
    skew,
    so3_exp,
    so3_log,
    left_jacobian_so3,
    right_jacobian_so3,
    quat_normalize,
    quat_mult,
    quat_inv,
    quat_to_rot,
    rot_to_quat,
    so3_exp_quat,
    nominal_imu_step,
    compute_error_state_jacobian_fx,
)


def test_skew_properties():
    """Verify skew-symmetric matrix properties."""
    v = np.array([1.2, -3.4, 5.6])
    K = skew(v)

    # Antisymmetry: K^T = -K
    np.testing.assert_allclose(K.T, -K, atol=1e-12)

    # Cross product equivalence: K @ u == cross(v, u)
    u = np.array([0.7, 2.1, -1.9])
    np.testing.assert_allclose(K @ u, np.cross(v, u), atol=1e-12)


def test_so3_exp_log_roundtrip():
    """Verify log(exp(phi)) == phi across small, medium, and large rotations."""
    test_vectors = [
        np.array([1e-10, 0.0, 0.0]),
        np.array([0.0, 1e-7, -1e-7]),
        np.array([1e-4, -2e-4, 3e-4]),
        np.array([0.1, -0.2, 0.3]),
        np.array([1.0, 1.2, -0.8]),
        np.array([2.5, -0.5, 0.2]),
        np.array([np.pi - 1e-3, 0.0, 0.0]),
    ]

    for phi in test_vectors:
        R = so3_exp(phi)
        # Verify R is orthogonal and det(R) == 1
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

        phi_recovered = so3_log(R)
        np.testing.assert_allclose(phi_recovered, phi, atol=1e-6)


def test_quaternion_rot_conversions():
    """Verify quaternion <-> rotation matrix conversion consistency."""
    test_rotations = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.2, -0.4, 0.8]),
        np.array([-1.5, 0.3, 0.9]),
    ]

    for phi in test_rotations:
        R_true = so3_exp(phi)
        q = rot_to_quat(R_true)
        R_recovered = quat_to_rot(q)
        np.testing.assert_allclose(R_recovered, R_true, atol=1e-9)

        q_exp = so3_exp_quat(phi)
        np.testing.assert_allclose(quat_to_rot(q_exp), R_true, atol=1e-9)


def test_jacobians_so3():
    """Verify left and right SO(3) Jacobians."""
    phi = np.array([0.3, -0.5, 0.4])
    Jl = left_jacobian_so3(phi)
    Jr = right_jacobian_so3(phi)

    # Jr(phi) == Jl(-phi)
    np.testing.assert_allclose(Jr, left_jacobian_so3(-phi), atol=1e-10)

    # Jl(phi) * phi == phi
    np.testing.assert_allclose(Jl @ phi, phi, atol=1e-10)
    np.testing.assert_allclose(Jr @ phi, phi, atol=1e-10)


def test_error_state_jacobian_fx_finite_differences():
    """Verify analytical F_x matches numerical central finite differences to < 1e-6.

    Perturb nominal state with delta_x, propagate both forward, and verify
    the error state matches F_x @ delta_x.
    """
    np.random.seed(42)
    dt = 0.005  # 200 Hz IMU step

    # Nominal states
    p0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v0 = np.array([0.5, -0.3, 1.2], dtype=np.float64)
    phi0 = np.array([0.1, -0.15, 0.2], dtype=np.float64)
    q0 = so3_exp_quat(phi0)
    ba0 = np.array([0.02, -0.01, 0.03], dtype=np.float64)
    bg0 = np.array([0.001, -0.002, 0.001], dtype=np.float64)

    # Measurements
    acc_m = np.array([0.4, 0.8, 9.5], dtype=np.float64)
    gyro_m = np.array([0.05, -0.08, 0.12], dtype=np.float64)
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    R0 = quat_to_rot(q0)
    acc_hat = acc_m - ba0
    gyro_hat = gyro_m - bg0

    # 1. Analytical Jacobian
    Fx_analytical = compute_error_state_jacobian_fx(R0, acc_hat, gyro_hat, dt)

    # Nominal step
    p_nom, v_nom, q_nom, ba_nom, bg_nom = nominal_imu_step(
        p0, v0, q0, ba0, bg0, acc_m, gyro_m, dt, gravity
    )
    R_nom = quat_to_rot(q_nom)

    # 2. Numerical Central Differences
    h = 1e-7
    Fx_num = np.zeros((15, 15), dtype=np.float64)

    def state_to_perturbed(dx: np.ndarray):
        p_p = p0 + dx[0:3]
        v_p = v0 + dx[3:6]
        # Local rotation perturbation: R_p = R0 @ Exp(delta_theta)
        delta_q = so3_exp_quat(dx[6:9])
        q_p = quat_normalize(quat_mult(q0, delta_q))
        ba_p = ba0 + dx[9:12]
        bg_p = bg0 + dx[12:15]
        return p_p, v_p, q_p, ba_p, bg_p

    def extract_error(p_p, v_p, q_p, ba_p, bg_p):
        dp = p_p - p_nom
        dv = v_p - v_nom
        # Local orientation error: R_p = R_nom @ Exp(dtheta) -> dtheta = log(R_nom^T @ R_p)
        R_p = quat_to_rot(q_p)
        dtheta = so3_log(R_nom.T @ R_p)
        dba = ba_p - ba_nom
        dbg = bg_p - bg_nom
        return np.concatenate([dp, dv, dtheta, dba, dbg])

    for i in range(15):
        dx_plus = np.zeros(15, dtype=np.float64)
        dx_plus[i] = h
        p_plus, v_plus, q_plus, ba_plus, bg_plus = nominal_imu_step(
            *state_to_perturbed(dx_plus), acc_m, gyro_m, dt, gravity
        )
        err_plus = extract_error(p_plus, v_plus, q_plus, ba_plus, bg_plus)

        dx_minus = np.zeros(15, dtype=np.float64)
        dx_minus[i] = -h
        p_minus, v_minus, q_minus, ba_minus, bg_minus = nominal_imu_step(
            *state_to_perturbed(dx_minus), acc_m, gyro_m, dt, gravity
        )
        err_minus = extract_error(p_minus, v_minus, q_minus, ba_minus, bg_minus)

        Fx_num[:, i] = (err_plus - err_minus) / (2.0 * h)

    # Check that analytical Jacobian matches numerical finite differences to < 1e-6
    max_err = np.max(np.abs(Fx_analytical - Fx_num))
    print(f"Max difference between analytical F_x and numerical finite diff: {max_err:.2e}")
    np.testing.assert_allclose(Fx_analytical, Fx_num, atol=1e-6)


def test_stationary_zero_drift_propagation():
    """Stationary IMU with gravity cancelled should produce zero drift in nominal kinematics."""
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.0])
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    ba = np.zeros(3)
    bg = np.zeros(3)

    gravity = np.array([0.0, 0.0, -9.81])
    # Accelerometer measures specific force: a_m = R^T (a - g) = -R^T g = [0, 0, +9.81]
    acc_m = np.array([0.0, 0.0, 9.81])
    gyro_m = np.zeros(3)

    p, v, q, _, _ = p0, v0, q0, ba, bg
    dt = 0.005  # 200 Hz
    for _ in range(200):  # 1 second
        p, v, q, _, _ = nominal_imu_step(p, v, q, ba, bg, acc_m, gyro_m, dt, gravity)

    np.testing.assert_allclose(p, p0, atol=1e-12)
    np.testing.assert_allclose(v, v0, atol=1e-12)
    np.testing.assert_allclose(q, q0, atol=1e-12)
