"""Tests for SyntheticVioDataset and EuRoC dataset loaders."""

from pathlib import Path
import numpy as np
import pytest
from vio_lab.dataset import SyntheticVioDataset, EurocAslDataset, ImuSample, StereoFrame
from scripts.download_euroc_sample import generate_offline_synthetic_dataset


def test_synthetic_dataset_generation():
    """Verify SyntheticVioDataset generates correctly timestamped IMU and Stereo events."""
    duration = 1.0  # 1 second
    ds = SyntheticVioDataset(duration_s=duration, imu_rate_hz=200.0, cam_rate_hz=20.0)

    imu_count = 0
    cam_count = 0
    prev_t = -1.0

    for ev in ds.events():
        # Verify chronological ordering
        assert ev.t >= prev_t
        prev_t = ev.t

        if isinstance(ev, ImuSample):
            imu_count += 1
            assert ev.acc.shape == (3,)
            assert ev.gyro.shape == (3,)
            # Stationary approx norm of acc should be around 9.81 m/s^2
            assert 8.0 < np.linalg.norm(ev.acc) < 12.0
        elif isinstance(ev, StereoFrame):
            cam_count += 1
            assert ev.img_l.shape == (480, 752)
            assert ev.img_r.shape == (480, 752)
            assert ev.T_wb_gt is not None
            assert ev.visible_landmarks is not None
            assert len(ev.visible_landmarks) > 10

    # For 1.0s: ~200 IMU samples and ~20 camera frames
    assert 190 <= imu_count <= 210
    assert 19 <= cam_count <= 22


def test_euroc_asl_loader(tmp_path: Path):
    """Verify EuRoC ASL dataset parser loads generated files correctly."""
    sample_dir = tmp_path / "euroc_test"
    generate_offline_synthetic_dataset(sample_dir, duration_s=0.5)

    loader = EurocAslDataset(sample_dir)
    gt = loader.load_ground_truth()
    assert len(gt) > 50

    events = list(loader.events())
    assert len(events) > 50

    imu_events = [e for e in events if isinstance(e, ImuSample)]
    cam_events = [e for e in events if isinstance(e, StereoFrame)]
    assert len(imu_events) > 80
    assert len(cam_events) >= 8
