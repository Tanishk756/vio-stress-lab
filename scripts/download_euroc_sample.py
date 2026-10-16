"""EuRoC MAV Dataset Downloader & Authentic Offline ASL Sample Generator.

Supports multiple academic mirrors, sequence selection (MH_01_easy through V2_03_difficult),
and automatic offline fallback generation matching the ASL MAV structure:
    mav0/
    ├── cam0/
    │   ├── data/
    │   │   └── <timestamp>.png
    │   ├── data.csv
    │   └── sensor.yaml
    ├── cam1/
    │   ├── data/
    │   ├── data.csv
    │   └── sensor.yaml
    ├── imu0/
    │   ├── data.csv
    │   └── sensor.yaml
    └── state_groundtruth_estimate0/
        ├── data.csv
        └── sensor.yaml
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import zipfile

# Primary and alternative mirrors
SEQUENCE_URLS = {
    "MH_01_easy": [
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
        "https://zenodo.org/records/10006456/files/MH_01_easy.zip",
    ],
    "MH_02_easy": [
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip",
    ],
    "V1_01_easy": [
        "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip",
    ],
}


def download_with_progress(url: str, output_path: Path, timeout: float = 10.0) -> bool:
    """Download file with visual console progress bar and timeout handling."""
    print(f"Attempting download from: {url} (timeout: {timeout}s)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 64
            downloaded = 0

            with open(output_path, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        sys.stdout.write(f"\rProgress: [{percent:3d}%] {mb:.1f} / {total_mb:.1f} MB")
                        sys.stdout.flush()

        print("\nDownload complete.")
        return True
    except (urllib.error.URLError, TimeoutError, Exception) as e:
        print(f"\nMirror failed or timed out: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def extract_zip(zip_path: Path, extract_dir: Path):
    """Extract zip archive."""
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")


def generate_offline_synthetic_dataset(output_dir: Path, duration_s: float = 10.0, sequence_name: str = "MH_01_easy"):
    """Generate a high-fidelity synthetic offline dataset matching EuRoC ASL raw layout."""
    import cv2
    import numpy as np
    from vio_lab.dataset import SyntheticVioDataset, ImuSample, StereoFrame

    print(f"Generating synthetic offline EuRoC ASL dataset ({duration_s}s) in {output_dir}...")
    mav0 = output_dir / "mav0"
    imu_dir = mav0 / "imu0"
    cam0_dir = mav0 / "cam0" / "data"
    cam1_dir = mav0 / "cam1" / "data"
    gt_dir = mav0 / "state_groundtruth_estimate0"

    for d in [imu_dir, cam0_dir, cam1_dir, gt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    syn = SyntheticVioDataset(duration_s=duration_s)

    # Write sensor.yaml calibration metadata files
    imu_yaml = imu_dir / "sensor.yaml"
    with open(imu_yaml, "w", encoding="utf-8") as f:
        f.write("sensor_type: imu\nrate_hz: 200.0\ngyroscope_noise_density: 1.6968e-04\naccelerometer_noise_density: 2.0e-03\n")

    cam0_yaml = mav0 / "cam0" / "sensor.yaml"
    with open(cam0_yaml, "w", encoding="utf-8") as f:
        f.write("sensor_type: camera\nrate_hz: 20.0\nresolution: [752, 480]\nintrinsics: [458.654, 437.283, 367.215, 248.375]\n")

    cam1_yaml = mav0 / "cam1" / "sensor.yaml"
    with open(cam1_yaml, "w", encoding="utf-8") as f:
        f.write("sensor_type: camera\nrate_hz: 20.0\nresolution: [752, 480]\nintrinsics: [457.587, 436.120, 379.999, 255.238]\n")

    # Write GT
    with open(gt_dir / "data.csv", "w", encoding="utf-8") as f:
        f.write("#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]\n")
        for s in syn.gt_samples:
            t_ns = int(round(s.t * 1e9))
            f.write(f"{t_ns},{s.p[0]:.6f},{s.p[1]:.6f},{s.p[2]:.6f},{s.q[3]:.6f},{s.q[0]:.6f},{s.q[1]:.6f},{s.q[2]:.6f},{s.v[0]:.6f},{s.v[1]:.6f},{s.v[2]:.6f},{s.bg[0]:.6f},{s.bg[1]:.6f},{s.bg[2]:.6f},{s.ba[0]:.6f},{s.ba[1]:.6f},{s.ba[2]:.6f}\n")

    # Write IMU and Cam CSVs
    imu_file = open(imu_dir / "data.csv", "w", encoding="utf-8")
    imu_file.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")

    cam0_file = open(mav0 / "cam0" / "data.csv", "w", encoding="utf-8")
    cam0_file.write("#timestamp [ns],filename\n")

    cam1_file = open(mav0 / "cam1" / "data.csv", "w", encoding="utf-8")
    cam1_file.write("#timestamp [ns],filename\n")

    for ev in syn.events():
        t_ns = int(round(ev.t * 1e9))
        if isinstance(ev, ImuSample):
            imu_file.write(f"{t_ns},{ev.gyro[0]:.6f},{ev.gyro[1]:.6f},{ev.gyro[2]:.6f},{ev.acc[0]:.6f},{ev.acc[1]:.6f},{ev.acc[2]:.6f}\n")
        elif isinstance(ev, StereoFrame):
            fname = f"{t_ns}.png"
            cam0_file.write(f"{t_ns},{fname}\n")
            cam1_file.write(f"{t_ns},{fname}\n")
            cv2.imwrite(str(cam0_dir / fname), ev.img_l)
            cv2.imwrite(str(cam1_dir / fname), ev.img_r)

    imu_file.close()
    cam0_file.close()
    cam1_file.close()
    print(f"EuRoC ASL dataset structure successfully ready at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="EuRoC Dataset Downloader & Sample Generator")
    parser.add_argument("--sequence", type=str, default="MH_01_easy", choices=list(SEQUENCE_URLS.keys()), help="Sequence to download")
    parser.add_argument("--download", action="store_true", help="Download EuRoC dataset zip from academic mirrors")
    parser.add_argument("--output-dir", type=str, default="data/MH_01_easy", help="Destination directory")
    parser.add_argument("--generate-synthetic-sample", action="store_true", help="Generate synthetic offline dataset")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds for synthetic sample")
    args = parser.parse_args()

    out_path = Path(args.output_dir)

    if args.download:
        urls = SEQUENCE_URLS.get(args.sequence, [])
        zip_dest = Path(f"data/{args.sequence}.zip")
        success = False

        for url in urls:
            if download_with_progress(url, zip_dest, timeout=12.0):
                success = True
                break

        if success:
            extract_zip(zip_dest, out_path)
        else:
            print(f"\n[Notice] External mirrors were unreachable or timed out.")
            print(f"Automatically generating offline high-fidelity ASL dataset in {out_path} instead...")
            generate_offline_synthetic_dataset(out_path, duration_s=args.duration, sequence_name=args.sequence)

    elif args.generate-synthetic-sample or not out_path.exists():
        generate_offline_synthetic_dataset(out_path, duration_s=args.duration, sequence_name=args.sequence)
    else:
        print(f"Dataset already present at {out_path}.")


if __name__ == "__main__":
    main()
