"""Trajectory Evaluation using evo programmatic API and TUM format exporters."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evo.core import metrics, sync, trajectory
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from vio_lab.pipeline import TrajectoryRecord
from vio_lab.se3_math import rot_to_quat


@dataclass
class ScenarioMetrics:
    """Quantitative performance score card for a single scenario."""
    scenario_name: str
    num_poses: int
    duration_s: float
    # ESKF VIO APE
    eskf_ape_rmse: float
    eskf_ape_mean: float
    eskf_ape_max: float
    eskf_ape_std: float
    # Pure Vision APE
    vis_ape_rmse: float
    # Pure IMU APE
    imu_ape_rmse: float
    # RPE (Relative Pose Error)
    eskf_rpe_rmse: float


def export_tum_trajectory(timestamps: list[float], poses: list[np.ndarray], output_file: Path) -> Path:
    """Export list of 4x4 SE(3) poses to standard TUM trajectory format.

    Format: timestamp tx ty tz qx qy qz qw
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for t, T in zip(timestamps, poses):
            p = T[:3, 3]
            q = rot_to_quat(T[:3, :3])  # [qx, qy, qz, qw]
            f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    return output_file


def record_to_evo_trajectory(timestamps: list[float], poses: list[np.ndarray]) -> PoseTrajectory3D:
    """Convert timestamps and 4x4 poses to evo PoseTrajectory3D object."""
    xyz = np.array([T[:3, 3] for T in poses], dtype=np.float64)
    quats = np.array([rot_to_quat(T[:3, :3]) for T in poses], dtype=np.float64)
    # evo quat format: [qw, qx, qy, qz]
    quats_wxyz = np.column_stack([quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]])
    return PoseTrajectory3D(
        positions_xyz=xyz,
        orientations_quat_wxyz=quats_wxyz,
        timestamps=np.array(timestamps, dtype=np.float64)
    )



def evaluate_trajectory_record(
    record: TrajectoryRecord,
    scenario_name: str = "nominal",
    output_dir: Optional[Path] = None
) -> ScenarioMetrics:
    """Compute APE and RPE metrics against ground truth via evo API."""
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_tum_trajectory(record.timestamps, record.poses_gt, output_dir / f"{scenario_name}_gt.tum")
        export_tum_trajectory(record.timestamps, record.poses_eskf, output_dir / f"{scenario_name}_eskf.tum")
        export_tum_trajectory(record.timestamps, record.poses_vis_only, output_dir / f"{scenario_name}_vis.tum")
        export_tum_trajectory(record.timestamps, record.poses_imu_only, output_dir / f"{scenario_name}_imu.tum")

    traj_gt = record_to_evo_trajectory(record.timestamps, record.poses_gt)
    traj_eskf = record_to_evo_trajectory(record.timestamps, record.poses_eskf)
    traj_vis = record_to_evo_trajectory(record.timestamps, record.poses_vis_only)
    traj_imu = record_to_evo_trajectory(record.timestamps, record.poses_imu_only)

    # Synchronize and align
    traj_eskf_sync, traj_gt_sync = sync.associate_trajectories(traj_eskf, traj_gt, max_diff=0.05)
    traj_vis_sync, _ = sync.associate_trajectories(traj_vis, traj_gt, max_diff=0.05)
    traj_imu_sync, _ = sync.associate_trajectories(traj_imu, traj_gt, max_diff=0.05)

    # 1. ESKF APE (Translation part)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_gt_sync, traj_eskf_sync))
    ape_stats = ape_metric.get_all_statistics()

    # 2. Vision-only APE
    ape_vis = metrics.APE(metrics.PoseRelation.translation_part)
    ape_vis.process_data((traj_gt_sync, traj_vis_sync))
    vis_stats = ape_vis.get_all_statistics()

    # 3. IMU-only APE
    ape_imu = metrics.APE(metrics.PoseRelation.translation_part)
    ape_imu.process_data((traj_gt_sync, traj_imu_sync))
    imu_stats = ape_imu.get_all_statistics()

    # 4. ESKF RPE (Relative Pose Error)
    rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta=1.0, delta_unit=metrics.Unit.frames)
    rpe_metric.process_data((traj_gt_sync, traj_eskf_sync))
    rpe_stats = rpe_metric.get_all_statistics()

    duration = record.timestamps[-1] - record.timestamps[0] if record.timestamps else 0.0

    return ScenarioMetrics(
        scenario_name=scenario_name,
        num_poses=len(record.poses_eskf),
        duration_s=duration,
        eskf_ape_rmse=float(ape_stats["rmse"]),
        eskf_ape_mean=float(ape_stats["mean"]),
        eskf_ape_max=float(ape_stats["max"]),
        eskf_ape_std=float(ape_stats["std"]),
        vis_ape_rmse=float(vis_stats["rmse"]),
        imu_ape_rmse=float(imu_stats["rmse"]),
        eskf_rpe_rmse=float(rpe_stats["rmse"]),
    )


def generate_benchmark_summary_markdown(
    results: list[ScenarioMetrics],
    output_path: Path
) -> str:
    """Generate professional Markdown benchmark table for portfolio README and reports."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# VIO Failure-Mode Stress Benchmark Report",
        "",
        "Quantitative evaluation comparing **15-State ESKF VIO** against **Pure Vision** and **IMU Dead-Reckoning** across nominal and corrupted sensor regimes.",
        "",
        "| Scenario | Fault Injection Parameters | Vision-Only APE (m) | IMU Dead-Reckoning (m) | **ESKF VIO APE (m)** | **ESKF RPE (m)** | Drift Reduction |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    fault_descriptions = {
        "nominal": "Zero faults; calibrated sensor noise",
        "blackout_3s": "Visual loss: camera blackout for $t \\in [10, 13]\\text{ s}$",
        "motion_blur": "$15\\times 15$ directional blur + $35\\%$ keypoint outliers",
        "bias_shock": "Step impulse: $+0.35\\text{ m/s}^2$ accel, $+0.03\\text{ rad/s}$ gyro",
        "time_desync_25ms": "Hardware trigger latency: $+25\\text{ ms}$ camera delay"
    }

    for r in results:
        desc = fault_descriptions.get(r.scenario_name, "Stress test")
        ratio = r.imu_ape_rmse / max(r.eskf_ape_rmse, 1e-4)
        lines.append(
            f"| **{r.scenario_name}** | {desc} | {r.vis_ape_rmse:.4f} | {r.imu_ape_rmse:.4f} | **{r.eskf_ape_rmse:.4f}** | **{r.eskf_rpe_rmse:.4f}** | **{ratio:.1f}x** |"
        )

    lines.extend([
        "",
        "### Key Architectural Insights:",
        "1. **Camera Blackout Resilience:** During the 3.0s visual blackout, pure vision tracking fails, while ESKF coasting on online-calibrated IMU biases bounds position drift and seamlessly re-locks upon feature recovery.",
        "2. **$\chi^2$ Outlier Rejection:** Gross visual mismatches and severe motion blur are intercepted by the 6-DOF Mahalanobis gate ($d^2 > 12.592$), preventing filter divergence.",
        "3. **Online Bias Tracking:** The 15-state filter dynamically estimates accelerometer and gyroscope bias walks, suppressing exponential dead-reckoning divergence.",
        ""
    ])

    md_content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return md_content


def plot_trajectory_comparison(
    record: TrajectoryRecord,
    title: str = "VIO Trajectory Comparison",
    output_path: Optional[Path] = None
) -> None:
    """Generate 2D Top-Down (X-Y) and Height (Z) trajectory comparison plot."""
    t_arr = np.array(record.timestamps)
    p_gt = np.array([T[:3, 3] for T in record.poses_gt])
    p_eskf = np.array([T[:3, 3] for T in record.poses_eskf])
    p_vis = np.array([T[:3, 3] for T in record.poses_vis_only])
    p_imu = np.array([T[:3, 3] for T in record.poses_imu_only])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Top-Down X-Y Plane
    ax1 = axes[0]
    ax1.plot(p_gt[:, 0], p_gt[:, 1], "g-", linewidth=2.0, label="Ground Truth")
    ax1.plot(p_eskf[:, 0], p_eskf[:, 1], "c--", linewidth=2.0, label="ESKF VIO (Ours)")
    ax1.plot(p_vis[:, 0], p_vis[:, 1], "m:", linewidth=1.5, alpha=0.7, label="Vision-Only")
    # Subsample IMU for clarity if diverging
    ax1.plot(p_imu[:, 0], p_imu[:, 1], "r-.", linewidth=1.2, alpha=0.6, label="IMU Dead-Reckoning")
    ax1.set_xlabel("X Position [m]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Y Position [m]", fontsize=11, fontweight="bold")
    ax1.set_title(f"{title} (Top-Down X-Y)", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", frameon=True)
    ax1.axis("equal")

    # Plot 2: Absolute Position Error Over Time
    ax2 = axes[1]
    err_eskf = np.linalg.norm(p_eskf - p_gt, axis=1)
    err_vis = np.linalg.norm(p_vis - p_gt, axis=1)
    err_imu = np.linalg.norm(p_imu - p_gt, axis=1)

    ax2.plot(t_arr, err_eskf, "c-", linewidth=2.0, label=f"ESKF APE (RMSE: {np.sqrt(np.mean(err_eskf**2)):.3f}m)")
    ax2.plot(t_arr, err_vis, "m:", linewidth=1.5, alpha=0.8, label=f"Vision APE (RMSE: {np.sqrt(np.mean(err_vis**2)):.3f}m)")
    ax2.plot(t_arr, err_imu, "r-.", linewidth=1.2, alpha=0.6, label=f"IMU APE (RMSE: {np.sqrt(np.mean(err_imu**2)):.3f}m)")
    ax2.set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Position Error [m]", fontsize=11, fontweight="bold")
    ax2.set_title("Absolute Pose Error (APE) Profile", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", frameon=True)

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Comparison plot saved to {output_path}")
    plt.close()


def plot_benchmark_summary(
    results: list[ScenarioMetrics],
    output_path: Path
) -> None:
    """Generate comparative summary bar chart across all evaluated scenarios."""
    scenarios = [r.scenario_name for r in results]
    eskf_rmse = [r.eskf_ape_rmse for r in results]
    vis_rmse = [r.vis_ape_rmse for r in results]
    imu_rmse = [r.imu_ape_rmse for r in results]

    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    ax.bar(x - width, imu_rmse, width, label="IMU Dead-Reckoning", color="#e53935", alpha=0.85)
    ax.bar(x, vis_rmse, width, label="Pure Vision", color="#fb8c00", alpha=0.85)
    ax.bar(x + width, eskf_rmse, width, label="ESKF VIO (Ours)", color="#00acc1", alpha=0.95)

    ax.set_ylabel("Absolute Pose Error (APE) RMSE [m] (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("VIO Stress Lab: Failure-Mode Robustness Benchmark Matrix", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Benchmark summary plot saved to {output_path}")
    plt.close()

