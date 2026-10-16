"""Unit tests for the Multi-State Constraint Kalman Filter (MSCKF)."""

import numpy as np
import pytest

from vio_lab.eskf import EskfConfig
from vio_lab.msckf import CameraClone, FeatureObservation, MsckfFilter
from vio_lab.se3_math import quat_to_rot, rot_to_quat, skew


def test_msckf_augmentation_and_marginalization():
    """Verify state cloning, dimension growth, and sliding-window marginalization."""
    max_clones = 3
    filt = MsckfFilter(max_clones=max_clones)
    assert filt.P.shape == (15, 15)
    assert len(filt.clones) == 0

    # Augment 1st clone
    id0 = filt.augment_state(t=0.1)
    assert id0 == 0
    assert filt.P.shape == (21, 21)
    assert len(filt.clones) == 1

    # Augment 2nd and 3rd clones
    filt.augment_state(t=0.2)
    filt.augment_state(t=0.3)
    assert filt.P.shape == (33, 33)
    assert len(filt.clones) == 3

    # Augment 4th clone -> should prune oldest clone and keep max_clones = 3
    id3 = filt.augment_state(t=0.4)
    assert len(filt.clones) == 3
    assert filt.P.shape == (33, 33)
    # The oldest clone remaining should now have clone_id 1
    assert filt.clones[0].clone_id == 1
    assert filt.clones[-1].clone_id == id3

    # Check symmetry and positive semi-definiteness (clones are linear projections of IMU state)
    sym_err = np.max(np.abs(filt.P - filt.P.T))
    assert sym_err < 1e-12
    eigvals = np.linalg.eigvalsh(filt.P)
    assert np.all(eigvals >= -1e-12)


def test_msckf_nullspace_projection_property():
    """Verify that QR decomposition gives an exact left nullspace of H_f: Q2^T * H_f = 0."""
    # Synthesize 4 camera views observing a landmark at [1.0, 0.5, 4.0]
    p_f = np.array([1.0, 0.5, 4.0])
    H_f_rows = []

    cam_positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.0, 0.0]),
        np.array([0.2, 0.05, 0.0]),
        np.array([0.3, 0.05, 0.02]),
    ]

    for p_c in cam_positions:
        # Camera looking along +Z
        R_c = np.eye(3)
        p_in_c = R_c.T @ (p_f - p_c)
        z = p_in_c[2]
        J_proj = np.array([
            [1.0 / z, 0.0, -p_in_c[0] / (z * z)],
            [0.0, 1.0 / z, -p_in_c[1] / (z * z)],
        ])
        H_f_i = J_proj @ R_c.T  # 2 x 3
        H_f_rows.append(H_f_i)

    H_f = np.vstack(H_f_rows)  # 8 x 3
    assert H_f.shape == (8, 3)

    # Complete QR
    Q, _ = np.linalg.qr(H_f, mode="complete")
    Q2 = Q[:, 3:]  # 8 x 5

    # Check Q2^T * H_f = 0
    nullspace_residual = Q2.T @ H_f
    norm_err = np.linalg.norm(nullspace_residual)
    assert norm_err < 1e-12, f"Nullspace residual too large: {norm_err}"


def test_msckf_multi_view_update():
    """Verify that multi-view feature observations reduce state uncertainty in MSCKF."""
    # Synthetic setup: body moves linearly along +X
    filt = MsckfFilter(
        max_clones=5,
        init_p=np.array([0.0, 0.0, 0.0]),
        init_v=np.array([1.0, 0.0, 0.0]),
    )

    true_landmarks = [
        np.array([0.5, -0.2, 3.0]),
        np.array([1.0, 0.3, 3.5]),
        np.array([1.5, -0.1, 4.0]),
    ]

    # Create 4 clones along trajectory
    clone_ids = []
    for step in range(4):
        t = step * 0.1
        # Step IMU prediction
        filt.predict_imu(dt=0.1, acc_m=np.array([0.0, 0.0, 9.81]), gyro_m=np.zeros(3))
        cid = filt.augment_state(t=t)
        clone_ids.append(cid)

    # Synthesize noise-free multi-view observations
    features = []
    for fid, lmk in enumerate(true_landmarks):
        obs_dict = {}
        for c in filt.clones:
            p_in_c = c.R.T @ (lmk - c.p)
            if p_in_c[2] > 0.5:
                u = p_in_c[0] / p_in_c[2]
                v = p_in_c[1] / p_in_c[2]
                obs_dict[c.clone_id] = np.array([u, v])
        if len(obs_dict) >= 3:
            features.append(FeatureObservation(feature_id=fid, observations=obs_dict))

    assert len(features) >= 3

    # Record trace of covariance before update
    cov_trace_before = np.trace(filt.P)

    # Perform MSCKF nullspace update
    success = filt.update_multi_view_features(features, meas_noise_std=0.01)
    assert success is True

    # Uncertainty must strictly decrease
    cov_trace_after = np.trace(filt.P)
    assert cov_trace_after < cov_trace_before

    # Covariance must remain symmetric and positive semi-definite
    assert np.max(np.abs(filt.P - filt.P.T)) < 1e-10
    eigvals = np.linalg.eigvalsh(filt.P)
    assert np.all(eigvals >= -1e-12)
