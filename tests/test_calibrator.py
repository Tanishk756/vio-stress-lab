"""Unit tests for 22-state online camera-IMU calibration and Jacobians."""

import numpy as np
import pytest
from vio_lab.calibrator import AugmentedEskf, CalibratorConfig
from vio_lab.se3_math import quat_mult, quat_normalize, quat_to_rot, rot_to_quat, so3_exp, so3_exp_quat, so3_log


def test_augmented_eskf_dimensions():
    """Verify 22-state covariance structure and initializations."""
    filter = AugmentedEskf(
        init_p=np.zeros(3),
        init_v=np.zeros(3),
        init_p_bc=np.array([0.05, 0.0, 0.0]),
        init_td=0.015
    )
    assert filter.P.shape == (22, 22)
    assert filter.p_bc.shape == (3,)
    assert filter.td == 0.015
    assert np.all(np.linalg.eigvalsh(filter.P) > 0)


def test_camera_measurement_jacobian_finite_differences():
    """Verify analytical measurement Jacobian H matches numerical central finite differences."""
    filter = AugmentedEskf(
        init_p=np.array([1.0, 2.0, 3.0]),
        init_v=np.array([0.5, -0.2, 0.8]),
        init_q=so3_exp_quat(np.array([0.1, -0.05, 0.2])),
        init_p_bc=np.array([0.05, -0.02, 0.01]),
        init_q_bc=so3_exp_quat(np.array([0.02, 0.04, -0.01])),
        init_td=0.010
    )
    filter.last_gyro_hat = np.array([0.05, -0.02, 0.08])

    R_wb = filter.R
    R_bc = filter.R_bc

    # Compute analytical H
    H_analytical = np.zeros((6, 22), dtype=np.float64)
    H_analytical[0:3, 0:3] = np.eye(3)
    H_analytical[0:3, 3:6] = np.eye(3) * filter.td
    from vio_lab.se3_math import skew
    H_analytical[0:3, 6:9] = -R_wb @ skew(filter.p_bc)
    H_analytical[0:3, 15:18] = R_wb
    H_analytical[0:3, 21] = filter.v

    R_td = so3_exp(filter.last_gyro_hat * filter.td)
    H_analytical[3:6, 6:9] = R_bc.T @ R_td.T
    H_analytical[3:6, 18:21] = np.eye(3)
    H_analytical[3:6, 21] = R_bc.T @ filter.last_gyro_hat


    # Numerical verification
    h = 1e-7
    H_num = np.zeros((6, 22), dtype=np.float64)

    def predict_camera_pose(p, v, q, p_bc, q_bc, td):
        R_w = quat_to_rot(q)
        R_b = quat_to_rot(q_bc)
        p_c = p + v * td + R_w @ p_bc
        R_c = R_w @ so3_exp(filter.last_gyro_hat * td) @ R_b
        return p_c, R_c

    p_nom, R_c_nom = predict_camera_pose(filter.p, filter.v, filter.q, filter.p_bc, filter.q_bc, filter.td)

    for i in range(22):
        dx_plus = np.zeros(22)
        dx_plus[i] = h
        dx_minus = np.zeros(22)
        dx_minus[i] = -h

        def apply_delta(dx):
            p = filter.p + dx[0:3]
            v = filter.v + dx[3:6]
            q = quat_normalize(quat_mult(filter.q, so3_exp_quat(dx[6:9])))
            p_bc = filter.p_bc + dx[15:18]
            q_bc = quat_normalize(quat_mult(filter.q_bc, so3_exp_quat(dx[18:21])))
            td = filter.td + dx[21]
            return p, v, q, p_bc, q_bc, td

        p_plus, Rc_plus = predict_camera_pose(*apply_delta(dx_plus))
        p_minus, Rc_minus = predict_camera_pose(*apply_delta(dx_minus))

        dp = (p_plus - p_minus) / (2.0 * h)
        dtheta_plus = so3_log(R_c_nom.T @ Rc_plus)
        dtheta_minus = so3_log(R_c_nom.T @ Rc_minus)
        dtheta = (dtheta_plus - dtheta_minus) / (2.0 * h)

        H_num[0:3, i] = dp
        H_num[3:6, i] = dtheta

    np.testing.assert_allclose(H_analytical, H_num, atol=1e-5)


def test_online_extrinsic_calibration_convergence():
    """Verify that AugmentedEskf estimates online camera-IMU lever-arm offset."""
    true_p_bc = np.array([0.08, -0.03, 0.02])
    true_q_bc = np.array([0.0, 0.0, 0.0, 1.0])

    # Filter starts with erroneous extrinsic guess (0 offset)
    filter = AugmentedEskf(
        init_p=np.zeros(3),
        init_v=np.array([1.0, 0.0, 0.0]),
        init_q=np.array([0.0, 0.0, 0.0, 1.0]),
        init_p_bc=np.zeros(3),  # Error is 8 cm on X!
        init_q_bc=true_q_bc,
        init_td=0.0
    )
    filter.P[15:18, 15:18] = np.eye(3) * 0.05  # High initial extrinsic uncertainty

    R_cov = np.eye(6) * 1e-4

    # Perform a sequence of camera updates with body motion
    for step in range(25):
        filter.predict_imu(dt=0.05, acc_m=np.array([0.0, 0.0, 9.81]), gyro_m=np.zeros(3))
        # Ground truth camera pose: T_wc = T_wb * T_bc_true
        T_wc_true = np.eye(4)
        T_wc_true[:3, :3] = filter.R @ quat_to_rot(true_q_bc)
        T_wc_true[:3, 3] = filter.p + filter.R @ true_p_bc

        filter.update_camera_pose(T_wc_true, R_cov, chi2_threshold=100.0)

    # Check that estimated p_bc converged towards true_p_bc
    extrinsic_err = np.linalg.norm(filter.p_bc - true_p_bc)
    print(f"\nInitial extrinsic error: {np.linalg.norm(true_p_bc):.4f} m")
    assert extrinsic_err < 0.015, f"Extrinsic error {extrinsic_err:.4f}m exceeds 1.5 cm threshold"


def test_augmented_eskf_zupt():
    """Verify that ZUPT works on 22-state AugmentedEskf filter."""
    filter = AugmentedEskf(
        init_v=np.array([0.20, -0.15, 0.30]),
        init_p_bc=np.array([0.05, 0.0, 0.0])
    )
    v_norm_before = np.linalg.norm(filter.v)
    cov_v_before = np.trace(filter.P[3:6, 3:6])

    filter.update_zero_velocity(vel_meas_std=0.01)

    v_norm_after = np.linalg.norm(filter.v)
    cov_v_after = np.trace(filter.P[3:6, 3:6])

    assert v_norm_after < 0.05
    assert cov_v_after < cov_v_before
    assert np.all(np.linalg.eigvalsh(filter.P) > 0)

