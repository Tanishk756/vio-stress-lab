"""Download EuRoC MAV MH_01_easy dataset or create a synthetic test dataset on disk."""

import argparse
import os
from pathlib import Path
import sys
import urllib.request
import zipfile

EUROC_MH01_URL = (
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip"
)


def download_with_progress(url: str, output_path: Path):
    """Download file with visual console progress bar."""
    print(f"Downloading from {url} to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def report(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: [{percent:3d}%] {downloaded_mb:.1f} / {total_mb:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook=report)
    print("\nDownload complete.")


def extract_zip(zip_path: Path, extract_dir: Path):
    """Extract zip archive."""
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")


def generate_offline_synthetic_dataset(output_dir: Path, duration_s: float = 5.0):
    """Generate a miniature synthetic dataset on disk formatted identically to EuRoC ASL."""
    import cv2
    import numpy as np
    from vio_lab.dataset import SyntheticVioDataset, ImuSample, StereoFrame

    print(f"Generating synthetic offline EuRoC-format dataset ({duration_s}s) in {output_dir}...")
    mav0 = output_dir / "mav0"
    imu_dir = mav0 / "imu0"
    cam0_dir = mav0 / "cam0" / "data"
    cam1_dir = mav0 / "cam1" / "data"
    gt_dir = mav0 / "state_groundtruth_estimate0"

    for d in [imu_dir, cam0_dir, cam1_dir, gt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    syn = SyntheticVioDataset(duration_s=duration_s)

    # Write GT
    with open(gt_dir / "data.csv", "w", encoding="utf-8") as f:
        f.write("#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]\n")
        for s in syn.gt_samples:
            t_ns = int(round(s.t * 1e9))
            f.write(f"{t_ns},{s.p[0]},{s.p[1]},{s.p[2]},{s.q[3]},{s.q[0]},{s.q[1]},{s.q[2]},{s.v[0]},{s.v[1]},{s.v[2]},{s.bg[0]},{s.bg[1]},{s.bg[2]},{s.ba[0]},{s.ba[1]},{s.ba[2]}\n")

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
            imu_file.write(f"{t_ns},{ev.gyro[0]},{ev.gyro[1]},{ev.gyro[2]},{ev.acc[0]},{ev.acc[1]},{ev.acc[2]}\n")
        elif isinstance(ev, StereoFrame):
            fname = f"{t_ns}.png"
            cam0_file.write(f"{t_ns},{fname}\n")
            cam1_file.write(f"{t_ns},{fname}\n")
            cv2.imwrite(str(cam0_dir / fname), ev.img_l)
            cv2.imwrite(str(cam1_dir / fname), ev.img_r)

    imu_file.close()
    cam0_file.close()
    cam1_file.close()
    print("Synthetic dataset generated successfully.")


def main():
    parser = argparse.ArgumentParser(description="EuRoC Dataset Downloader & Sample Generator")
    parser.add_argument("--download", action="store_true", help="Download official MH_01_easy zip")
    parser.add_argument("--url", type=str, default=EUROC_MH01_URL, help="Dataset download URL")
    parser.add_argument("--output-dir", type=str, default="data/MH_01_easy", help="Destination directory")
    parser.add_argument("--generate-synthetic-sample", action="store_true", help="Generate synthetic offline dataset")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds for synthetic sample")
    args = parser.parse_args()

    out_path = Path(args.output_dir)

    if args.download:
        zip_dest = Path("data/MH_01_easy.zip")
        download_with_progress(args.url, zip_dest)
        extract_zip(zip_dest, out_path)
    elif args.generate_synthetic_sample:
        generate_offline_synthetic_dataset(out_path, duration_s=args.duration)
    else:
        print("Please specify either --download or --generate-synthetic-sample.")


if __name__ == "__main__":
    main()
