"""Unit and stress-testing verification for sensor degradation fault injection."""

import numpy as np
import pytest
from vio_lab.dataset import ImuSample, StereoFrame, SyntheticVioDataset
from vio_lab.degradation import DegradationConfig, FaultInjectionEngine
from vio_lab.eskf import Eskf, EskfConfig
from vio_lab.frontend import StereoKltFrontend
from vio_lab.pipeline import VioPipeline


def test_blackout_fault_injection():
    """Verify camera blackout zeroes images and features only within the window."""
    cfg = DegradationConfig(blackout_enabled=True, blackout_t_start=5.0, blackout_t_end=8.0)
    engine = FaultInjectionEngine(cfg)

    # Frame before blackout
    f_before = StereoFrame(
        t=4.0,
        img_l=np.ones((10, 10), dtype=np.uint8) * 100,
        img_r=np.ones((10, 10), dtype=np.uint8) * 100,
        visible_landmarks={1: np.zeros(3)},
        visible_keypoints={1: np.zeros(2)}
    )
    out_before = engine.process_event(f_before)
    assert np.all(out_before.img_l == 100)
    assert len(out_before.visible_landmarks) == 1

    # Frame during blackout
    f_during = StereoFrame(
        t=6.5,
        img_l=np.ones((10, 10), dtype=np.uint8) * 100,
        img_r=np.ones((10, 10), dtype=np.uint8) * 100,
        visible_landmarks={1: np.zeros(3)},
        visible_keypoints={1: np.zeros(2)}
    )
    out_during = engine.process_event(f_during)
    assert np.all(out_during.img_l == 0)
    assert len(out_during.visible_landmarks) == 0
    assert len(out_during.visible_keypoints) == 0

    # Frame after blackout
    f_after = StereoFrame(
        t=9.0,
        img_l=np.ones((10, 10), dtype=np.uint8) * 100,
        img_r=np.ones((10, 10), dtype=np.uint8) * 100,
        visible_landmarks={1: np.zeros(3)},
        visible_keypoints={1: np.zeros(2)}
    )
    out_after = engine.process_event(f_after)
    assert np.all(out_after.img_l == 100)
    assert len(out_after.visible_landmarks) == 1


def test_imu_bias_step_injection():
    """Verify IMU bias shock step impulse is correctly applied at t >= shock_t."""
    cfg = DegradationConfig(
        imu_bias_shock_enabled=True,
        bias_shock_t=10.0,
        acc_bias_step=np.array([0.5, 0.0, 0.0]),
        gyro_bias_step=np.array([0.0, 0.05, 0.0])
    )
    engine = FaultInjectionEngine(cfg)

    imu_pre = ImuSample(t=9.9, acc=np.array([1.0, 2.0, 3.0]), gyro=np.zeros(3))
    out_pre = engine.process_event(imu_pre)
    np.testing.assert_allclose(out_pre.acc, [1.0, 2.0, 3.0])

    imu_post = ImuSample(t=10.1, acc=np.array([1.0, 2.0, 3.0]), gyro=np.zeros(3))
    out_post = engine.process_event(imu_post)
    np.testing.assert_allclose(out_post.acc, [1.5, 2.0, 3.0])
    np.testing.assert_allclose(out_post.gyro, [0.0, 0.05, 0.0])


def test_blackout_dead_reckoning_bridge():
    """Verify ESKF VIO bridges 3-second camera blackout with calibrated IMU coasting."""
    duration_s = 18.0
    dataset = SyntheticVioDataset(duration_s=duration_s, seed=42)
    init_gt = dataset.gt_samples[0]

    eskf = Eskf(
        init_p=init_gt.p, init_v=init_gt.v, init_q=init_gt.q,
        init_ba=init_gt.ba, init_bg=init_gt.bg
    )
    frontend = StereoKltFrontend(
        K=dataset.K, baseline=dataset.baseline, T_bc=dataset.T_bc0,
        base_pos_std=0.03, base_rot_std=0.02
    )

    # 3-second blackout from 10.0s to 13.0s
    fault_cfg = DegradationConfig(
        blackout_enabled=True,
        blackout_t_start=10.0,
        blackout_t_end=13.0
    )
    engine = FaultInjectionEngine(fault_cfg)

    pipeline = VioPipeline(eskf=eskf, frontend=frontend, degradation_hook=engine.process_event)

    for event in dataset.events():
        pipeline.process_event(event)

    rec = pipeline.records
    p_eskf = np.array([T[:3, 3] for T in rec.poses_eskf])
    p_gt = np.array([T[:3, 3] for T in rec.poses_gt])
    t_arr = np.array(rec.timestamps)

    # Check that covariance expands during blackout
    cov_diag = np.array([np.trace(P[0:3, 0:3]) for P in rec.covariances])
    idx_before = np.argmin(np.abs(t_arr - 9.8))
    idx_end_blackout = np.argmin(np.abs(t_arr - 12.9))
    idx_after = np.argmin(np.abs(t_arr - 14.5))

    # Position covariance should inflate during blackout
    assert cov_diag[idx_end_blackout] > cov_diag[idx_before] * 1.5, (
        f"Covariance did not inflate: before={cov_diag[idx_before]:.2e}, end={cov_diag[idx_end_blackout]:.2e}"
    )

    # After vision returns, covariance snaps back
    assert cov_diag[idx_after] < cov_diag[idx_end_blackout], (
        f"Covariance did not snap back: end={cov_diag[idx_end_blackout]:.2e}, after={cov_diag[idx_after]:.2e}"
    )

    # Drift during blackout remains bounded (< 0.35 m)
    blackout_mask = (t_arr >= 10.0) & (t_arr <= 13.0)
    blackout_errors = np.linalg.norm(p_eskf[blackout_mask] - p_gt[blackout_mask], axis=1)
    max_blackout_err = np.max(blackout_errors)
    print(f"\nMax position error during 3-second blackout: {max_blackout_err:.4f} m")
    assert max_blackout_err < 0.35, f"Blackout drift {max_blackout_err:.3f}m exceeded 0.35m threshold"
