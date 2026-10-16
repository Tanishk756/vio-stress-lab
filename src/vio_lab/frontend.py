"""Stereo KLT Visual Frontend with Grid Bucketing, Triangulation, and PnP RANSAC."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import cv2
import numpy as np

from vio_lab.se3_math import invert_transform, transform_point


@dataclass
class VisualPoseEstimate:
    """Estimated 6-DOF pose and adaptive measurement covariance."""
    t: float
    T_wb: np.ndarray  # 4x4 SE(3) pose of body in world
    R_cov: np.ndarray  # 6x6 covariance [pos(3), rot(3)]
    num_tracked: int
    num_inliers: int
    inlier_ratio: float
    valid: bool
    tracked_pts_2d: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    tracked_pts_3d: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))


class StereoKltFrontend:
    """Grid-bucketed Stereo Lucas-Kanade Visual Odometry Frontend."""

    def __init__(
        self,
        K: np.ndarray,
        baseline: float,
        T_bc: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        max_features: int = 250,
        grid_rows: int = 6,
        grid_cols: int = 8,
        min_inliers_for_update: int = 15,
        base_pos_std: float = 0.02,
        base_rot_std: float = 0.01,
    ):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.baseline = baseline
        self.T_bc = np.asarray(T_bc, dtype=np.float64).reshape(4, 4)
        self.T_cb = invert_transform(self.T_bc)
        self.dist_coeffs = dist_coeffs

        self.max_features = max_features
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.min_inliers_for_update = min_inliers_for_update
        self.base_pos_std = base_pos_std
        self.base_rot_std = base_rot_std

        # State across frames
        self.prev_img_l: Optional[np.ndarray] = None
        self.prev_pts_l: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.prev_pts_3d_w: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.prev_feature_ids: np.ndarray = np.empty((0,), dtype=np.int64)
        self.next_feature_id = 0

        # Current estimated pose of body in world
        self.current_T_wb: Optional[np.ndarray] = None

    def process_frame(
        self,
        t: float,
        img_l: np.ndarray,
        img_r: np.ndarray,
        prior_T_wb: Optional[np.ndarray] = None,
        synthetic_landmarks: Optional[dict[int, np.ndarray]] = None,
        synthetic_keypoints: Optional[dict[int, np.ndarray]] = None
    ) -> VisualPoseEstimate:
        """Process stereo image pair and estimate 6-DOF camera body pose."""
        # Ensure grayscale
        if len(img_l.shape) == 3:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        if len(img_r.shape) == 3:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        h, w = img_l.shape

        # If subpixel synthetic keypoints and landmarks are provided directly
        if synthetic_landmarks is not None and synthetic_keypoints is not None and len(synthetic_keypoints) >= self.min_inliers_for_update:
            common_ids = [k for k in synthetic_keypoints.keys() if k in synthetic_landmarks]
            pts_2d = np.array([synthetic_keypoints[k] for k in common_ids], dtype=np.float32)
            pts_3d = np.array([synthetic_landmarks[k] for k in common_ids], dtype=np.float64)

            # Solve PnP
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=pts_3d,
                imagePoints=pts_2d,
                cameraMatrix=self.K,
                distCoeffs=self.dist_coeffs,
                iterationsCount=100,
                reprojectionError=2.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP
            )

            if success and inliers is not None and len(inliers) >= self.min_inliers_for_update:
                R_cw, _ = cv2.Rodrigues(rvec)
                R_wc = R_cw.T
                t_wc = -R_wc @ tvec.flatten()
                T_wc = np.eye(4, dtype=np.float64)
                T_wc[:3, :3] = R_wc
                T_wc[:3, 3] = t_wc
                T_wb_est = T_wc @ self.T_cb

                n_inl = len(inliers)
                ratio = float(n_inl) / max(len(pts_2d), 1)
                cov_scale = (1.0 / max(ratio, 0.2)) ** 2 * (30.0 / max(n_inl, 5))
                pos_var = (self.base_pos_std ** 2) * cov_scale
                rot_var = (self.base_rot_std ** 2) * cov_scale

                R_cov = np.zeros((6, 6), dtype=np.float64)
                R_cov[0:3, 0:3] = np.eye(3) * pos_var
                R_cov[3:6, 3:6] = np.eye(3) * rot_var

                self.current_T_wb = T_wb_est
                self.prev_img_l = img_l.copy()
                self.prev_pts_l = pts_2d[inliers.flatten()]
                self.prev_pts_3d_w = pts_3d[inliers.flatten()]
                self.prev_feature_ids = np.array(common_ids, dtype=np.int64)[inliers.flatten()]

                return VisualPoseEstimate(
                    t=t,
                    T_wb=T_wb_est,
                    R_cov=R_cov,
                    num_tracked=len(pts_2d),
                    num_inliers=n_inl,
                    inlier_ratio=ratio,
                    valid=True,
                    tracked_pts_2d=pts_2d[inliers.flatten()],
                    tracked_pts_3d=pts_3d[inliers.flatten()]
                )


        if self.prev_img_l is None:
            # First keyframe bootstrap
            self._bootstrap_keyframe(img_l, img_r, prior_T_wb, synthetic_landmarks)
            init_T_wb = prior_T_wb if prior_T_wb is not None else np.eye(4)
            self.current_T_wb = init_T_wb
            return VisualPoseEstimate(
                t=t,
                T_wb=init_T_wb,
                R_cov=np.eye(6) * 1e-4,
                num_tracked=len(self.prev_pts_l),
                num_inliers=len(self.prev_pts_l),
                inlier_ratio=1.0,
                valid=True,
                tracked_pts_2d=self.prev_pts_l.copy(),
                tracked_pts_3d=self.prev_pts_3d_w.copy()
            )


        # 1. Temporal KLT Tracking from prev_img_l to curr img_l
        if len(self.prev_pts_l) > 0:
            curr_pts_l, valid_temporal = self._track_klt_bidirectional(
                self.prev_img_l, img_l, self.prev_pts_l
            )
        else:
            curr_pts_l = np.empty((0, 2), dtype=np.float32)
            valid_temporal = np.array([], dtype=bool)

        tracked_pts_l = curr_pts_l[valid_temporal]
        tracked_pts_3d_w = self.prev_pts_3d_w[valid_temporal]
        if len(self.prev_feature_ids) != len(valid_temporal):
            self.prev_feature_ids = np.arange(len(valid_temporal), dtype=np.int64)
        tracked_ids = self.prev_feature_ids[valid_temporal]


        # 2. Estimate 6-DOF pose via solvePnPRansac
        T_wb_est = None
        inliers_idx = np.array([], dtype=np.int32)

        if len(tracked_pts_3d_w) >= 6:
            # Initial guess from prior if provided
            use_extrinsic_guess = False
            rvec_guess = np.zeros((3, 1), dtype=np.float64)
            tvec_guess = np.zeros((3, 1), dtype=np.float64)

            if prior_T_wb is not None:
                # Prior camera pose: T_wc = prior_T_wb @ T_bc
                T_wc_prior = prior_T_wb @ self.T_bc
                # T_cw = T_wc^-1
                T_cw_prior = invert_transform(T_wc_prior)
                rvec_guess, _ = cv2.Rodrigues(T_cw_prior[:3, :3])
                tvec_guess = T_cw_prior[:3, 3:4]
                use_extrinsic_guess = True

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=tracked_pts_3d_w,
                imagePoints=tracked_pts_l,
                cameraMatrix=self.K,
                distCoeffs=self.dist_coeffs,
                rvec=rvec_guess,
                tvec=tvec_guess,
                useExtrinsicGuess=use_extrinsic_guess,
                iterationsCount=150,
                reprojectionError=2.5,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success and inliers is not None and len(inliers) >= self.min_inliers_for_update:
                inliers_idx = inliers.flatten()
                R_cw, _ = cv2.Rodrigues(rvec)
                # T_wc = [R_cw^T, -R_cw^T @ tvec; 0, 1]
                R_wc = R_cw.T
                t_wc = -R_wc @ tvec.flatten()
                T_wc = np.eye(4, dtype=np.float64)
                T_wc[:3, :3] = R_wc
                T_wc[:3, 3] = t_wc

                # Body pose: T_wb = T_wc @ T_cb
                T_wb_est = T_wc @ self.T_cb

        # Determine validity and compute adaptive measurement covariance
        num_tracked = len(tracked_pts_l)
        num_inliers = len(inliers_idx)
        inlier_ratio = float(num_inliers) / max(num_tracked, 1)

        is_valid = (T_wb_est is not None) and (num_inliers >= self.min_inliers_for_update)

        if is_valid:
            self.current_T_wb = T_wb_est
            # Keep inlier points
            surviving_pts_l = tracked_pts_l[inliers_idx]
            surviving_pts_3d_w = tracked_pts_3d_w[inliers_idx]
            surviving_ids = tracked_ids[inliers_idx]

            # Scale covariance: less inliers -> higher uncertainty
            # If inlier_ratio is low, multiply covariance
            cov_scale = (1.0 / max(inlier_ratio, 0.15)) ** 2 * (40.0 / max(num_inliers, 5))
            pos_var = (self.base_pos_std ** 2) * cov_scale
            rot_var = (self.base_rot_std ** 2) * cov_scale

            R_cov = np.zeros((6, 6), dtype=np.float64)
            R_cov[0:3, 0:3] = np.eye(3) * pos_var
            R_cov[3:6, 3:6] = np.eye(3) * rot_var
        else:
            # Visual tracking failed or degraded
            T_wb_est = prior_T_wb if prior_T_wb is not None else self.current_T_wb
            if T_wb_est is None:
                T_wb_est = np.eye(4)
            R_cov = np.eye(6, dtype=np.float64) * 1e6
            surviving_pts_l = np.empty((0, 2), dtype=np.float32)
            surviving_pts_3d_w = np.empty((0, 3), dtype=np.float64)
            surviving_ids = np.empty((0,), dtype=np.int64)

        # 3. Replenish features if inliers drop below budget
        if len(surviving_pts_l) < self.max_features * 0.6:
            new_pts_l, new_pts_3d_w, new_ids = self._detect_and_triangulate_new(
                img_l, img_r, surviving_pts_l, self.current_T_wb, synthetic_landmarks
            )
            if len(new_pts_l) > 0:
                surviving_pts_l = np.vstack([surviving_pts_l, new_pts_l]) if len(surviving_pts_l) > 0 else new_pts_l
                surviving_pts_3d_w = np.vstack([surviving_pts_3d_w, new_pts_3d_w]) if len(surviving_pts_3d_w) > 0 else new_pts_3d_w
                surviving_ids = np.concatenate([surviving_ids, new_ids]) if len(surviving_ids) > 0 else new_ids

        # Update historical state for next iteration
        self.prev_img_l = img_l.copy()
        self.prev_pts_l = surviving_pts_l
        self.prev_pts_3d_w = surviving_pts_3d_w
        self.prev_feature_ids = surviving_ids

        return VisualPoseEstimate(
            t=t,
            T_wb=T_wb_est,
            R_cov=R_cov,
            num_tracked=num_tracked,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            valid=is_valid,
            tracked_pts_2d=surviving_pts_l.copy(),
            tracked_pts_3d=surviving_pts_3d_w.copy()
        )

    def _track_klt_bidirectional(
        self, img_prev: np.ndarray, img_curr: np.ndarray, pts_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform forward-backward KLT tracking with consistency threshold (< 1.0 px)."""
        pts_prev = pts_prev.reshape(-1, 1, 2).astype(np.float32)

        # Forward flow: prev -> curr
        pts_fwd, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            img_prev, img_curr, pts_prev, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Backward flow: curr -> prev
        pts_bwd, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            img_curr, img_prev, pts_fwd, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Forward-backward error
        fb_err = np.linalg.norm(pts_prev[:, 0, :] - pts_bwd[:, 0, :], axis=1)

        # Boundary check
        h, w = img_curr.shape
        in_bounds = (
            (pts_fwd[:, 0, 0] >= 5) & (pts_fwd[:, 0, 0] < w - 5) &
            (pts_fwd[:, 0, 1] >= 5) & (pts_fwd[:, 0, 1] < h - 5)
        )

        valid = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1) & (fb_err < 1.0) & in_bounds
        return pts_fwd[:, 0, :], valid

    def _bootstrap_keyframe(
        self,
        img_l: np.ndarray,
        img_r: np.ndarray,
        T_wb_init: Optional[np.ndarray],
        synthetic_landmarks: Optional[dict[int, np.ndarray]]
    ):
        """Initialize features and 3D landmarks for the first frame."""
        T_wb = T_wb_init if T_wb_init is not None else np.eye(4)
        new_pts_l, new_pts_3d_w, new_ids = self._detect_and_triangulate_new(
            img_l, img_r, np.empty((0, 2), dtype=np.float32), T_wb, synthetic_landmarks
        )
        self.prev_img_l = img_l.copy()
        self.prev_pts_l = new_pts_l
        self.prev_pts_3d_w = new_pts_3d_w
        self.prev_feature_ids = new_ids

    def _detect_and_triangulate_new(
        self,
        img_l: np.ndarray,
        img_r: np.ndarray,
        existing_pts_l: np.ndarray,
        T_wb: Optional[np.ndarray],
        synthetic_landmarks: Optional[dict[int, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect grid-bucketed corners on left image, match to right, and triangulate."""
        if T_wb is None:
            T_wb = np.eye(4)

        h, w = img_l.shape
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Mask out regions near existing features
        if len(existing_pts_l) > 0:
            for pt in existing_pts_l:
                cv2.circle(mask, (int(round(pt[0])), int(round(pt[1]))), 10, 0, -1)

        # Detect Shi-Tomasi corners
        corners = cv2.goodFeaturesToTrack(
            img_l,
            maxCorners=self.max_features - len(existing_pts_l),
            qualityLevel=0.01,
            minDistance=12,
            mask=mask,
            blockSize=5
        )

        if corners is None or len(corners) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int64)
            )

        pts_l = corners[:, 0, :]

        # Match to right image via optical flow (stereo epipolar search)
        pts_r, valid_stereo = self._track_klt_bidirectional(img_l, img_r, pts_l)

        # Check horizontal epipolar constraint: |v_l - v_r| < 1.5 px and positive disparity
        v_diff = np.abs(pts_l[:, 1] - pts_r[:, 1])
        disparity = pts_l[:, 0] - pts_r[:, 0]

        valid_match = valid_stereo & (v_diff < 1.5) & (disparity > 1.5)

        good_pts_l = pts_l[valid_match]
        good_disp = disparity[valid_match]

        # Triangulate in camera frame: Z = fx * baseline / disparity
        Z_c = (self.fx * self.baseline) / np.maximum(good_disp, 0.1)
        X_c = (good_pts_l[:, 0] - self.cx) * Z_c / self.fx
        Y_c = (good_pts_l[:, 1] - self.cy) * Z_c / self.fy
        pts_3d_c = np.column_stack([X_c, Y_c, Z_c])

        # If ground truth synthetic landmarks are provided, associate with known 3D points
        if synthetic_landmarks is not None and len(synthetic_landmarks) > 0:
            T_wc_prior = T_wb @ self.T_bc
            R_cw_prior = T_wc_prior[:3, :3].T
            t_cw_prior = -R_cw_prior @ T_wc_prior[:3, 3]

            matched_pts_l = []
            matched_pts_3d_w = []

            if len(good_pts_l) > 0:
                used_indices = set()
                for lm_id, pw in synthetic_landmarks.items():
                    pc = R_cw_prior @ pw + t_cw_prior
                    if pc[2] > 0.5:
                        u = (self.fx * pc[0]) / pc[2] + self.cx
                        v = (self.fy * pc[1]) / pc[2] + self.cy
                        if 25 <= u < w - 25 and 25 <= v < h - 25:
                            dists = np.hypot(good_pts_l[:, 0] - u, good_pts_l[:, 1] - v)
                            min_idx = int(np.argmin(dists))
                            if dists[min_idx] < 8.0 and min_idx not in used_indices:
                                used_indices.add(min_idx)
                                matched_pts_l.append(good_pts_l[min_idx])
                                matched_pts_3d_w.append(pw)

            if len(matched_pts_l) >= 15:
                final_pts_l = np.array(matched_pts_l, dtype=np.float32)
                final_pts_3d_w = np.array(matched_pts_3d_w, dtype=np.float64)
                n_new = len(final_pts_l)
                new_ids = np.arange(self.next_feature_id, self.next_feature_id + n_new, dtype=np.int64)
                self.next_feature_id += n_new
                return final_pts_l, final_pts_3d_w, new_ids


        # Otherwise fallback to stereo triangulation
        valid_depth = (Z_c >= 0.5) & (Z_c <= 35.0)
        final_pts_l = good_pts_l[valid_depth]
        final_pts_3d_c = pts_3d_c[valid_depth]

        # Transform 3D points to world frame: P_w = T_wb * T_bc * P_c
        T_wc = T_wb @ self.T_bc
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]

        final_pts_3d_w = (R_wc @ final_pts_3d_c.T).T + t_wc

        # Assign unique feature IDs
        n_new = len(final_pts_l)
        new_ids = np.arange(self.next_feature_id, self.next_feature_id + n_new, dtype=np.int64)
        self.next_feature_id += n_new

        return final_pts_l, final_pts_3d_w, new_ids

