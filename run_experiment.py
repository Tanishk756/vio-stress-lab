"""Master CLI Entrypoint: VIO Failure-Mode Stress Lab & 3D ESKF Debugger."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

from vio_lab.dataset import SyntheticVioDataset, EurocAslDataset
from vio_lab.degradation import DegradationConfig, FaultInjectionEngine
from vio_lab.eskf import Eskf, EskfConfig
from vio_lab.evaluate import (
    ScenarioMetrics,
    evaluate_trajectory_record,
    generate_benchmark_summary_markdown,
    plot_trajectory_comparison,
    plot_benchmark_summary,
)

from vio_lab.frontend import StereoKltFrontend
from vio_lab.pipeline import VioPipeline
from vio_lab.visualizer import RerunVisualizer


SCENARIOS = {
    "nominal": DegradationConfig(),
    "blackout_3s": DegradationConfig(
        blackout_enabled=True,
        blackout_t_start=10.0,
        blackout_t_end=13.0
    ),
    "motion_blur": DegradationConfig(
        motion_blur_enabled=True,
        blur_t_start=7.0,
        blur_t_end=10.0,
        outlier_ratio=0.35
    ),
    "bias_shock": DegradationConfig(
        imu_bias_shock_enabled=True,
        bias_shock_t=12.0,
        acc_bias_step=np.array([0.35, -0.20, 0.15]),
        gyro_bias_step=np.array([0.03, -0.02, 0.015])
    ),
    "time_desync_25ms": DegradationConfig(
        time_desync_enabled=True,
        cam_time_offset_s=0.025
    )
}


def build_pipeline_for_dataset(
    dataset: SyntheticVioDataset | EurocAslDataset,
    degradation_cfg: DegradationConfig
) -> VioPipeline:
    """Instantiate ESKF, Stereo Frontend, and Pipeline for given dataset."""
    if isinstance(dataset, SyntheticVioDataset):
        init_gt = dataset.gt_samples[0]
        eskf = Eskf(
            init_p=init_gt.p, init_v=init_gt.v, init_q=init_gt.q,
            init_ba=init_gt.ba, init_bg=init_gt.bg
        )
        frontend = StereoKltFrontend(
            K=dataset.K,
            baseline=dataset.baseline,
            T_bc=dataset.T_bc0,
            base_pos_std=0.03,
            base_rot_std=0.02
        )
    else:
        # EuRoC ASL Dataset
        gt_samples = dataset.load_ground_truth()
        init_gt = gt_samples[0] if gt_samples else None
        p0 = init_gt.p if init_gt else np.zeros(3)
        q0 = init_gt.q if init_gt else np.array([0, 0, 0, 1.0])
        v0 = init_gt.v if init_gt else np.zeros(3)
        ba0 = init_gt.ba if init_gt else np.zeros(3)
        bg0 = init_gt.bg if init_gt else np.zeros(3)

        eskf = Eskf(init_p=p0, init_v=v0, init_q=q0, init_ba=ba0, init_bg=bg0)
        # EuRoC cam0 intrinsics & extrinsics
        K_euroc = np.array([
            [458.654, 0.0, 367.215],
            [0.0, 437.283, 248.375],
            [0.0, 0.0, 1.0]
        ])
        T_bc0_euroc = np.array([
            [0.01486554, -0.99988093, 0.0041403, -0.02164015],
            [0.99955725, 0.01496721, 0.02571553, -0.06467699],
            [-0.02577444, 0.00375619, 0.99966073, 0.00981073],
            [0.0, 0.0, 0.0, 1.0]
        ])
        frontend = StereoKltFrontend(
            K=K_euroc,
            baseline=0.11007,
            T_bc=T_bc0_euroc,
            base_pos_std=0.04,
            base_rot_std=0.025
        )

    fault_engine = FaultInjectionEngine(degradation_cfg)
    pipeline = VioPipeline(eskf=eskf, frontend=frontend, degradation_hook=fault_engine.process_event)
    # Cold start pure IMU dead-reckoning (realistic drift baseline)
    pipeline.imu_only_ba = np.zeros(3)
    pipeline.imu_only_bg = np.zeros(3)

    return pipeline


def run_single_scenario(
    scenario_name: str,
    degradation_cfg: DegradationConfig,
    dataset: SyntheticVioDataset | EurocAslDataset,
    output_dir: Path,
    save_rrd: bool = False,
    spawn_viewer: bool = False
) -> ScenarioMetrics:
    """Execute pipeline for one scenario and compute evaluation metrics."""
    print(f"\n=======================================================")
    print(f" Running Scenario: {scenario_name}")
    print(f"=======================================================")

    pipeline = build_pipeline_for_dataset(dataset, degradation_cfg)

    # Process all events in stream
    for ev in dataset.events():
        pipeline.process_event(ev)

    records = pipeline.records
    print(f"Processed {len(records.timestamps)} visual frames.")

    # Compute evo metrics
    metrics = evaluate_trajectory_record(records, scenario_name=scenario_name, output_dir=output_dir / "trajectories")

    print(f"--- Results for {scenario_name} ---")
    print(f"ESKF VIO APE RMSE:       {metrics.eskf_ape_rmse:.4f} m")
    print(f"Vision-Only APE RMSE:    {metrics.vis_ape_rmse:.4f} m")
    print(f"IMU Dead-Reckoning RMSE: {metrics.imu_ape_rmse:.4f} m")
    print(f"Improvement Factor:      {metrics.imu_ape_rmse / max(metrics.eskf_ape_rmse, 1e-4):.1f}x")

    # Plot trajectory
    plot_trajectory_comparison(
        records,
        title=f"VIO Stress Lab - {scenario_name}",
        output_path=output_dir / f"plots/{scenario_name}_comparison.png"
    )

    # Log to Rerun if requested
    if save_rrd or spawn_viewer:
        rrd_file = output_dir / f"{scenario_name}_telemetry.rrd"
        viz = RerunVisualizer(
            app_id=f"vio-{scenario_name}",
            save_path=rrd_file if save_rrd else None,
            spawn_viewer=spawn_viewer
        )
        viz.log_pipeline_record(records, save_path=rrd_file if save_rrd else None)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="VIO Failure-Mode Stress Lab & 3D ESKF Debugger")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic 6-DOF trajectory generator")
    parser.add_argument("--euroc-path", type=str, default=None, help="Path to EuRoC ASL dataset (e.g., data/MH_01_easy)")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds for evaluation")
    parser.add_argument("--scenario", type=str, default="nominal", choices=list(SCENARIOS.keys()), help="Scenario to run")
    parser.add_argument("--all-scenarios", action="store_true", help="Execute complete benchmark stress matrix")
    parser.add_argument("--save-rrd", action="store_true", help="Save Rerun 3D recording file (.rrd)")
    parser.add_argument("--view", action="store_true", help="Spawn live Rerun viewer")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save benchmark reports and plots")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine dataset source
    if args.euroc_path is not None and Path(args.euroc_path).exists():
        print(f"Loading EuRoC ASL Dataset from {args.euroc_path}...")
        dataset = EurocAslDataset(args.euroc_path)
    else:
        print(f"Using Synthetic 6-DOF Figure-Eight Trajectory ({args.duration}s)...")
        dataset = SyntheticVioDataset(duration_s=args.duration, seed=42)

    if args.all_scenarios:
        # Run entire stress benchmark matrix
        all_metrics: list[ScenarioMetrics] = []
        for name, cfg in SCENARIOS.items():
            # Regenerate fresh dataset stream for each scenario
            if isinstance(dataset, SyntheticVioDataset):
                ds = SyntheticVioDataset(duration_s=args.duration, seed=42)
            else:
                ds = EurocAslDataset(args.euroc_path)

            save_demo_rrd = args.save_rrd and (name == "blackout_3s")
            m = run_single_scenario(
                scenario_name=name,
                degradation_cfg=cfg,
                dataset=ds,
                output_dir=output_dir,
                save_rrd=args.save_rrd,
                spawn_viewer=args.view and (name == "nominal")
            )
            all_metrics.append(m)

        # Generate summary markdown table and summary comparison plot
        summary_md_path = output_dir / "benchmark_table.md"
        generate_benchmark_summary_markdown(all_metrics, summary_md_path)
        plot_benchmark_summary(all_metrics, output_dir / "ape_comparison.png")
        print(f"\n=======================================================")
        print(f" Benchmark Suite Completed! Summary written to:")
        print(f" -> {summary_md_path}")
        print(f" -> {output_dir / 'ape_comparison.png'}")
        print(f"=======================================================\n")


        # Create demo rrd link
        if args.save_rrd:
            demo_rrd = output_dir / "vio_stress_demo.rrd"
            src_rrd = output_dir / "blackout_3s_telemetry.rrd"
            if src_rrd.exists():
                import shutil
                shutil.copyfile(src_rrd, demo_rrd)
                print(f"Primary demo recording saved: {demo_rrd}")

    else:
        cfg = SCENARIOS[args.scenario]
        m = run_single_scenario(
            scenario_name=args.scenario,
            degradation_cfg=cfg,
            dataset=dataset,
            output_dir=output_dir,
            save_rrd=args.save_rrd,
            spawn_viewer=args.view
        )
        summary_md_path = output_dir / f"{args.scenario}_benchmark.md"
        generate_benchmark_summary_markdown([m], summary_md_path)


if __name__ == "__main__":
    main()
