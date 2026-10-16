"""Integration test verifying ESKF VIO convergence and baseline comparisons."""

import numpy as np
import pytest
from vio_lab.dataset import SyntheticVioDataset
from vio_lab.eskf import Eskf, EskfConfig
from vio_lab.frontend import StereoKltFrontend
from vio_lab.pipeline import VioPipeline


def test_eskf_vio_synthetic_convergence():
    """Verify ESKF VIO converges on synthetic figure-eight sequence.

    Requirements:
    1. ESKF position RMSE APE < 0.15 m.
    2. ESKF outperforms pure IMU dead-reckoning by > 10x.
    """
    duration_s = 20.0  # Sufficient duration to observe multi-meter IMU drift
    dataset = SyntheticVioDataset(
        duration_s=duration_s,
        imu_rate_hz=200.0,
        cam_rate_hz=20.0,
        num_landmarks=300,
        seed=42
    )

    init_gt = dataset.gt_samples[0]

    # Initialize ESKF with initial ground truth state
    eskf_cfg = EskfConfig(
        acc_noise_density=0.002,
        gyro_noise_density=0.0002,
        acc_random_walk=0.0001,
        gyro_random_walk=0.00001,
        chi2_threshold_6dof=15.0
    )
    eskf = Eskf(
        config=eskf_cfg,
        init_p=init_gt.p,
        init_v=init_gt.v,
        init_q=init_gt.q,
        init_ba=init_gt.ba,
        init_bg=init_gt.bg
    )

    frontend = StereoKltFrontend(
        K=dataset.K,
        baseline=dataset.baseline,
        T_bc=dataset.T_bc0,
        max_features=250,
        min_inliers_for_update=12,
        base_pos_std=0.03,
        base_rot_std=0.025
    )

    pipeline = VioPipeline(eskf=eskf, frontend=frontend)
    # Pure IMU dead reckoning realistically does not have pre-calibrated online biases
    pipeline.imu_only_ba = np.zeros(3)
    pipeline.imu_only_bg = np.zeros(3)

    for event in dataset.events():
        pipeline.process_event(event)

    records = pipeline.records
    assert len(records.poses_eskf) > 100

    # Calculate position errors
    p_eskf = np.array([T[:3, 3] for T in records.poses_eskf])
    p_gt = np.array([T[:3, 3] for T in records.poses_gt])
    p_imu = np.array([T[:3, 3] for T in records.poses_imu_only])

    eskf_errors = np.linalg.norm(p_eskf - p_gt, axis=1)
    imu_errors = np.linalg.norm(p_imu - p_gt, axis=1)

    rmse_eskf = np.sqrt(np.mean(eskf_errors**2))
    rmse_imu = np.sqrt(np.mean(imu_errors**2))

    print(f"\n--- Synthetic Integration Results ({duration_s}s) ---")
    print(f"ESKF VIO Position RMSE:     {rmse_eskf:.4f} m")
    print(f"IMU Dead-Reckoning RMSE:    {rmse_imu:.4f} m")
    print(f"Improvement Factor:         {rmse_imu / max(rmse_eskf, 1e-6):.1f}x")

    assert rmse_eskf < 0.15, f"ESKF RMSE {rmse_eskf:.4f}m exceeds 0.15m limit"
    assert rmse_imu / rmse_eskf > 10.0, f"ESKF improvement {rmse_imu/rmse_eskf:.1f}x is less than 10x"
