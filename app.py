"""Interactive Web UI & Degrader Studio for VIO Stress Lab.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from vio_lab.calibrator import AugmentedEskf
from vio_lab.dataset import SyntheticVioDataset, EurocAslDataset
from vio_lab.degradation import DegradationConfig, FaultInjectionEngine
from vio_lab.eskf import Eskf, EskfConfig
from vio_lab.evaluate import evaluate_trajectory_record
from vio_lab.frontend import StereoKltFrontend
from vio_lab.msckf import MsckfFilter
from vio_lab.pipeline import VioPipeline
from vio_lab.se3_math import rot_to_quat


# Configure page layout & metadata
st.set_page_config(
    page_title="VIO Failure-Mode Stress Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern dark aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00E5FF, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #9E9E9E;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1E1E2F;
        border: 1px solid #2D2D44;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
        background-color: #2D2D44;
        color: #00E5FF;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-title">🔬 VIO Failure-Mode Stress Lab & 3D ESKF Debugger</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Interactive Sensor Fault Injection, 15/22-State ESKF, and Multi-State Constraint (MSCKF) Verification Studio</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <span class="badge">15-State ESKF</span>
        <span class="badge">22-State Online Calibrator</span>
        <span class="badge">MSCKF Nullspace Projection</span>
        <span class="badge">Chi-Square Outlier Gating</span>
        <span class="badge">Stereo KLT Frontend</span>
        <span class="badge">Real-time Fault Injection</span>
    </div>
    """, unsafe_allow_html=True)

    # ==========================
    # SIDEBAR CONTROLS
    # ==========================
    st.sidebar.header("⚙️ Simulation Settings")

    dataset_type = st.sidebar.selectbox(
        "Trajectory / Dataset Source",
        options=["Synthetic 6-DOF (Figure-Eight)", "EuRoC MAV Dataset (ASL format)"],
        index=0
    )

    duration_s = st.sidebar.slider(
        "Simulation Duration (seconds)",
        min_value=5.0,
        max_value=30.0,
        value=15.0,
        step=1.0,
        help="Higher durations demonstrate compound dead-reckoning drift during visual outages."
    )

    filter_mode = st.sidebar.selectbox(
        "State Estimation Filter Engine",
        options=[
            "15-State ESKF (Standard)",
            "22-State Online Calibrator (Extrinsics & Latency)",
            "MSCKF (Sliding-Window Feature Nullspace)"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🚨 Sensor Degradation & Faults")

    # 1. Camera Blackout
    blackout_enabled = st.sidebar.checkbox("⬛ Camera Visual Blackout", value=True)
    if blackout_enabled:
        c1, c2 = st.sidebar.columns(2)
        b_start = c1.number_input("Start (s)", min_value=1.0, max_value=float(duration_s - 1.0), value=6.0, step=0.5)
        b_end = c2.number_input("End (s)", min_value=float(b_start + 0.5), max_value=float(duration_s), value=9.0, step=0.5)
    else:
        b_start, b_end = 6.0, 9.0

    # 2. Motion Blur & Outliers
    blur_enabled = st.sidebar.checkbox("🌀 High-Speed Motion Blur & Outliers", value=False)
    if blur_enabled:
        blur_kernel = st.sidebar.slider("Motion Blur Kernel (px)", min_value=5, max_value=25, value=15, step=2)
        outlier_ratio = st.sidebar.slider("Feature Outlier Mismatch Ratio", min_value=0.1, max_value=0.7, value=0.35, step=0.05)
    else:
        blur_kernel = 15
        outlier_ratio = 0.35

    # 3. IMU Bias Step Shock
    bias_shock_enabled = st.sidebar.checkbox("⚡ IMU Bias Step Shock (Mechanical/Thermal)", value=False)
    if bias_shock_enabled:
        shock_t = st.sidebar.slider("Shock Timestamp (s)", min_value=1.0, max_value=float(duration_s - 1.0), value=7.0, step=0.5)
        acc_shock = st.sidebar.slider("Acc Bias Step (m/s²)", min_value=0.05, max_value=1.0, value=0.35, step=0.05)
        gyro_shock = st.sidebar.slider("Gyro Bias Step (rad/s)", min_value=0.005, max_value=0.1, value=0.03, step=0.005)
    else:
        shock_t = 7.0
        acc_shock = 0.35
        gyro_shock = 0.03

    # 4. Camera-IMU Desync
    desync_enabled = st.sidebar.checkbox("⏱️ Hardware Trigger Time Desynchronization", value=False)
    if desync_enabled:
        cam_latency_ms = st.sidebar.slider("Camera Latency (ms)", min_value=5, max_value=100, value=25, step=5)
    else:
        cam_latency_ms = 25

    # Construct DegradationConfig
    degradation_cfg = DegradationConfig(
        blackout_enabled=blackout_enabled,
        blackout_t_start=float(b_start),
        blackout_t_end=float(b_end),
        motion_blur_enabled=blur_enabled,
        blur_t_start=4.0,
        blur_t_end=7.0,
        blur_kernel_size=blur_kernel,
        outlier_ratio=outlier_ratio,
        imu_bias_shock_enabled=bias_shock_enabled,
        bias_shock_t=float(shock_t),
        acc_bias_step=np.array([acc_shock, -acc_shock * 0.5, acc_shock * 0.3]),
        gyro_bias_step=np.array([gyro_shock, -gyro_shock * 0.6, gyro_shock * 0.4]),
        time_desync_enabled=desync_enabled,
        cam_time_offset_s=float(cam_latency_ms) / 1000.0
    )

    run_btn = st.sidebar.button("🚀 Run VIO Stress Simulation", type="primary", use_container_width=True)

    if not run_btn and "last_results" not in st.session_state:
        st.info("👈 Configure your trajectory & fault-injection profile in the sidebar and click **'Run VIO Stress Simulation'** to execute.")
        # Render architecture diagram
        st.markdown("### System Architecture")
        st.markdown("""
        ```
        +------------------------------------------------------------------------------------+
        |                               VIO-STRESS-LAB ENGINE                                |
        +------------------------------------------------------------------------------------+
                   |                                                    |
         [IMU 200 Hz Stream]                                  [Stereo 20 Hz Stream]
                   |                                                    |
        (Fault Injection Engine:                              (Fault Injection Engine:
         Bias Shock / Temp Drift)                              Blackout / Blur / Outliers)
                   |                                                    |
                   v                                                    v
         [Nominal IMU Propagator]                             [Stereo KLT Optical Flow]
         dt = 5ms Kinematics Step                                       |
                   |                                                    v
                   |                                          [EPnP + RANSAC Pose Engine]
                   |                                                    |
                   |                                                    v
                   +-------------------> [ESKF / MSCKF] <---------------+
                                         - 15/22-State Error State
                                         - Chi-Square Innovation Gate
                                         - Covariance Swell Tracking
        ```
        """)
        return

    if run_btn:
        with st.spinner("Executing Visual-Inertial Odometry and Sensor Fault Injection..."):
            start_wall_time = time.time()

            # Initialize dataset
            if "EuRoC" in dataset_type:
                euroc_dir = Path("data/MH_01_easy")
                if not euroc_dir.exists():
                    st.warning("EuRoC sequence not found locally in `data/MH_01_easy`. Generating offline ASL dataset...")
                    from scripts.download_euroc_sample import generate_offline_euroc_asl_sample
                    generate_offline_euroc_asl_sample(euroc_dir, duration_s=duration_s)
                dataset = EurocAslDataset(euroc_dir)
            else:
                dataset = SyntheticVioDataset(duration_s=duration_s, seed=42)

            # Build filter according to selected mode
            gt_samples = dataset.gt_samples
            if gt_samples:
                init_gt = gt_samples[0]
                p0, v0, q0 = init_gt.p.copy(), init_gt.v.copy(), init_gt.q.copy()
                ba0, bg0 = init_gt.ba.copy(), init_gt.bg.copy()
            else:
                p0, v0 = np.zeros(3), np.zeros(3)
                q0 = np.array([0.0, 0.0, 0.0, 1.0])
                ba0, bg0 = np.zeros(3), np.zeros(3)

            if "22-State" in filter_mode:
                filt = AugmentedEskf(
                    init_p=p0,
                    init_v=v0,
                    init_q=q0,
                    init_ba=ba0,
                    init_bg=bg0,
                    init_p_bc=dataset.T_bc0[:3, 3],
                    init_q_bc=rot_to_quat(dataset.T_bc0[:3, :3])
                )
            elif "MSCKF" in filter_mode:
                filt = MsckfFilter(
                    init_p=p0,
                    init_v=v0,
                    init_q=q0,
                    init_ba=ba0,
                    init_bg=bg0,
                    T_bc=dataset.T_bc0,
                    max_clones=8
                )
            else:
                filt = Eskf(
                    init_p=p0,
                    init_v=v0,
                    init_q=q0,
                    init_ba=ba0,
                    init_bg=bg0
                )

            frontend = StereoKltFrontend(
                K=dataset.K,
                baseline=dataset.baseline,
                T_bc=dataset.T_bc0,
                base_pos_std=0.03,
                base_rot_std=0.02
            )
            fault_engine = FaultInjectionEngine(degradation_cfg)
            pipeline = VioPipeline(
                eskf=filt,
                frontend=frontend,
                degradation_hook=fault_engine.process_event
            )
            pipeline.imu_only_ba = np.zeros(3)
            pipeline.imu_only_bg = np.zeros(3)

            # Run stream
            events_gen = dataset.events(max_seconds=duration_s) if isinstance(dataset, EurocAslDataset) else dataset.events()
            for ev in events_gen:
                pipeline.process_event(ev)

            records = pipeline.records
            if len(records.timestamps) < 2:
                st.error("No valid frames were processed from dataset.")
                return

            scenario_name = "custom_stress"
            metrics = evaluate_trajectory_record(records, scenario_name=scenario_name)
            elapsed_time = time.time() - start_wall_time

            st.session_state["last_results"] = {
                "records": records,
                "metrics": metrics,
                "elapsed": elapsed_time,
                "degradation_cfg": degradation_cfg,
                "duration": duration_s,
                "filter_mode": filter_mode
            }

    # Fetch results from session state
    res = st.session_state["last_results"]
    records = res["records"]
    metrics = res["metrics"]
    degradation_cfg = res["degradation_cfg"]

    # ==========================
    # METRIC KPI CARDS
    # ==========================
    st.markdown("### 📊 Performance Scorecard")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    improvement = metrics.imu_ape_rmse / max(metrics.eskf_ape_rmse, 1e-4)

    with kpi1:
        st.metric(
            label="ESKF VIO APE RMSE",
            value=f"{metrics.eskf_ape_rmse:.3f} m",
            delta=f"-{metrics.imu_ape_rmse - metrics.eskf_ape_rmse:.2f} m vs IMU",
            delta_color="inverse"
        )
    with kpi2:
        st.metric(
            label="Max Tracking Error",
            value=f"{metrics.eskf_ape_max:.3f} m",
            help="Maximum deviation along entire trajectory."
        )
    with kpi3:
        st.metric(
            label="Pure IMU Drift RMSE",
            value=f"{metrics.imu_ape_rmse:.2f} m",
            help="Compound unconstrained quadratic dead-reckoning drift."
        )
    with kpi4:
        st.metric(
            label="Improvement Factor",
            value=f"{improvement:.1f}x",
            delta="Robust to outages" if improvement > 5.0 else "Vulnerable"
        )
    with kpi5:
        num_rejected = sum(1 for acc in records.chi2_accepted if not acc)
        st.metric(
            label="Rejected Outliers",
            value=f"{num_rejected}",
            help="Visual measurements rejected by Chi-Square innovation gating."
        )

    # ==========================
    # 3D INTERACTIVE TRAJECTORY
    # ==========================
    st.markdown("### 🌐 3D Spatial Trajectory Comparison")

    p_gt = np.array([T[:3, 3] for T in records.poses_gt])
    p_eskf = np.array([T[:3, 3] for T in records.poses_eskf])
    p_imu = np.array([T[:3, 3] for T in records.poses_imu_only])
    p_vis = np.array([T[:3, 3] for T in records.poses_vis_only])

    fig_3d = go.Figure()

    # Ground Truth
    fig_3d.add_trace(go.Scatter3d(
        x=p_gt[:, 0], y=p_gt[:, 1], z=p_gt[:, 2],
        mode='lines',
        name='Ground Truth',
        line=dict(color='#00E676', width=5)
    ))

    # ESKF Estimate
    fig_3d.add_trace(go.Scatter3d(
        x=p_eskf[:, 0], y=p_eskf[:, 1], z=p_eskf[:, 2],
        mode='lines+markers',
        marker=dict(size=2, color='#00E5FF'),
        name='15-State ESKF VIO',
        line=dict(color='#00E5FF', width=4)
    ))

    # Vision-Only
    if len(p_vis) > 0:
        fig_3d.add_trace(go.Scatter3d(
            x=p_vis[:, 0], y=p_vis[:, 1], z=p_vis[:, 2],
            mode='lines',
            name='Vision-Only (PnP)',
            line=dict(color='#FF9100', width=2, dash='dot')
        ))

    # Pure IMU
    fig_3d.add_trace(go.Scatter3d(
        x=p_imu[:, 0], y=p_imu[:, 1], z=p_imu[:, 2],
        mode='lines',
        name='Pure IMU Dead-Reckoning',
        line=dict(color='#FF1744', width=3, dash='dash')
    ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (meters)',
            bgcolor='#11111E',
        ),
        paper_bgcolor='#11111E',
        font=dict(color='#E0E0E0'),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(30,30,47,0.7)'),
        height=550
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # ==========================
    # 2D TIME-SERIES TELEMETRY & UNCERTAINTY
    # ==========================
    st.markdown("### 📈 Error Evolution & 3-Sigma Covariance Dynamics")

    timestamps = np.array(records.timestamps)
    eskf_errors = np.linalg.norm(p_eskf - p_gt, axis=1)
    imu_errors = np.linalg.norm(p_imu - p_gt, axis=1)

    # Compute 3-sigma position envelope from filter covariances
    three_sigma = []
    for cov in records.covariances:
        pos_cov = cov[:3, :3]
        eigvals = np.linalg.eigvalsh(pos_cov)
        max_std = np.sqrt(max(float(np.max(eigvals)), 1e-12))
        three_sigma.append(3.0 * max_std)
    three_sigma = np.array(three_sigma)

    fig_telemetry = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Absolute Position Error (APE) Over Time", "3-Sigma Position Uncertainty Swell Envelope")
    )

    # Subplot 1: APE Errors
    fig_telemetry.add_trace(
        go.Scatter(x=timestamps, y=eskf_errors, mode='lines', name='ESKF Error (m)', line=dict(color='#00E5FF', width=2.5)),
        row=1, col=1
    )
    fig_telemetry.add_trace(
        go.Scatter(x=timestamps, y=imu_errors, mode='lines', name='IMU Error (m)', line=dict(color='#FF1744', width=1.5, dash='dash')),
        row=1, col=1
    )

    # Highlight Blackout Window if enabled
    if degradation_cfg.blackout_enabled:
        fig_telemetry.add_vrect(
            x0=degradation_cfg.blackout_t_start,
            x1=degradation_cfg.blackout_t_end,
            fillcolor="red",
            opacity=0.25,
            line_width=1,
            line_color="red",
            annotation_text="Visual Blackout",
            annotation_position="top left",
            row=1, col=1
        )
        fig_telemetry.add_vrect(
            x0=degradation_cfg.blackout_t_start,
            x1=degradation_cfg.blackout_t_end,
            fillcolor="red",
            opacity=0.25,
            line_width=1,
            line_color="red",
            row=2, col=1
        )

    # Subplot 2: 3-Sigma Covariance Swell
    fig_telemetry.add_trace(
        go.Scatter(
            x=timestamps, y=three_sigma,
            mode='lines',
            name='±3σ Position Bound (m)',
            line=dict(color='#FFD600', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 214, 0, 0.15)'
        ),
        row=2, col=1
    )

    fig_telemetry.update_layout(
        paper_bgcolor='#11111E',
        plot_bgcolor='#1E1E2F',
        font=dict(color='#E0E0E0'),
        height=500,
        margin=dict(l=40, r=40, b=40, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_telemetry.update_xaxes(title_text="Time (seconds)", row=2, col=1, gridcolor="#2D2D44")
    fig_telemetry.update_yaxes(title_text="Error (m)", row=1, col=1, gridcolor="#2D2D44")
    fig_telemetry.update_yaxes(title_text="3-Sigma (m)", row=2, col=1, gridcolor="#2D2D44")

    st.plotly_chart(fig_telemetry, use_container_width=True)

    # ==========================
    # BIAS & CALIBRATION DYNAMICS
    # ==========================
    with st.expander("🔬 Online IMU Bias & Calibration Telemetry", expanded=False):
        if len(records.ba_estimates) > 0:
            ba_arr = np.array(records.ba_estimates)
            bg_arr = np.array(records.bg_estimates)

            fig_bias = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Estimated Accelerometer Biases (m/s²)", "Estimated Gyroscope Biases (rad/s)")
            )

            fig_bias.add_trace(go.Scatter(x=timestamps, y=ba_arr[:, 0], mode='lines', name='b_ax', line=dict(color='#FF5252')), row=1, col=1)
            fig_bias.add_trace(go.Scatter(x=timestamps, y=ba_arr[:, 1], mode='lines', name='b_ay', line=dict(color='#69F0AE')), row=1, col=1)
            fig_bias.add_trace(go.Scatter(x=timestamps, y=ba_arr[:, 2], mode='lines', name='b_az', line=dict(color='#448AFF')), row=1, col=1)

            fig_bias.add_trace(go.Scatter(x=timestamps, y=bg_arr[:, 0], mode='lines', name='b_gx', line=dict(color='#FF5252', dash='dot')), row=1, col=2)
            fig_bias.add_trace(go.Scatter(x=timestamps, y=bg_arr[:, 1], mode='lines', name='b_gy', line=dict(color='#69F0AE', dash='dot')), row=1, col=2)
            fig_bias.add_trace(go.Scatter(x=timestamps, y=bg_arr[:, 2], mode='lines', name='b_gz', line=dict(color='#448AFF', dash='dot')), row=1, col=2)

            fig_bias.update_layout(
                paper_bgcolor='#11111E',
                plot_bgcolor='#1E1E2F',
                font=dict(color='#E0E0E0'),
                height=320,
                margin=dict(l=40, r=40, b=30, t=30)
            )
            fig_bias.update_xaxes(title_text="Time (s)", gridcolor="#2D2D44")
            fig_bias.update_yaxes(gridcolor="#2D2D44")
            st.plotly_chart(fig_bias, use_container_width=True)

    # ==========================
    # TELEMETRY DATA EXPORT
    # ==========================
    st.markdown("### 💾 Export Telemetry & Benchmark Data")
    d1, d2 = st.columns(2)

    # Export TUM string
    tum_lines = []
    for t, T in zip(records.timestamps, records.poses_eskf):
        p = T[:3, 3]
        q = rot_to_quat(T[:3, :3])
        tum_lines.append(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    tum_data = "\n".join(tum_lines)

    with d1:
        st.download_button(
            label="📥 Download TUM Trajectory File (.tum)",
            data=tum_data,
            file_name="eskf_vio_trajectory.tum",
            mime="text/plain",
            use_container_width=True
        )

    # Export CSV metrics
    df_metrics = pd.DataFrame([{
        "scenario": "interactive_run",
        "eskf_ape_rmse_m": metrics.eskf_ape_rmse,
        "eskf_ape_max_m": metrics.eskf_ape_max,
        "imu_ape_rmse_m": metrics.imu_ape_rmse,
        "vis_ape_rmse_m": metrics.vis_ape_rmse,
        "improvement_factor": improvement,
        "num_poses": len(records.poses_eskf)
    }])
    csv_data = df_metrics.to_csv(index=False)

    with d2:
        st.download_button(
            label="📥 Download Metrics Summary (.csv)",
            data=csv_data,
            file_name="vio_benchmark_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
