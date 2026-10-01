import threading
import numpy as np
import cv2
import os
import glob
import json


APP_DIR = os.path.dirname(os.path.abspath(__file__))


class CalibrationThread(threading.Thread):
    """Stereo camera calibration computation thread."""

    def __init__(
        self,
        base_sync_dir,
        square_size=35.0,
        board_size=(11, 8),
        progress_callback=None,
        finished_callback=None,
        failed_callback=None,
    ):
        super().__init__(daemon=True)
        self.base_sync_dir = base_sync_dir
        self.square_size = square_size / 1000.0  # mm -> m
        self.chessboard_size = board_size
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.failed_callback = failed_callback
        self._stop_event = threading.Event()

    def _update_progress(self, message):
        if self.progress_callback:
            self.progress_callback(message)

    def _find_chessboard_corners(self, images):
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2) * self.square_size

        objpoints = []
        imgpoints = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        found_count = 0
        total_count = len(images)
        img_size = None

        for i, img_path in enumerate(images):
            if self._stop_event.is_set():
                return None, None, None

            img = cv2.imread(img_path)
            if img is None:
                self._update_progress(f"Warning: failed to read image {os.path.basename(img_path)}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                found_count += 1
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                objpoints.append(objp)
                self._update_progress(f"Image {i + 1}/{total_count}: corners found ({found_count} valid)")
            else:
                self._update_progress(f"Image {i + 1}/{total_count}: corners not found")

        return objpoints, imgpoints, img_size

    def _calibrate_camera(self, cam_dir, camera_name):
        self._update_progress(f"--- Calibrating {camera_name} ---")

        image_extensions = ("*.png", "*.jpg", "*.jpeg")
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(cam_dir, ext)))
        images.sort()

        if not images:
            raise ValueError(f"{camera_name}: no calibration images found")

        objpoints, imgpoints, img_size = self._find_chessboard_corners(images)
        if objpoints is None or imgpoints is None:
            raise RuntimeError(f"{camera_name}: calibration stopped")
        if not objpoints:
            raise ValueError(f"{camera_name}: no valid chessboard detections in images")
        if img_size is None:
            raise ValueError(f"{camera_name}: failed to infer image size")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        if not ret:
            raise RuntimeError(f"{camera_name}: failed to solve calibration matrix")

        mean_error = 0.0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += float(error)
        mean_error /= max(1, len(objpoints))

        calib_result = {
            "camera_name": camera_name,
            "image_size": list(img_size),
            "camera_matrix": mtx.tolist(),
            "distortion_coeffs": dist.tolist(),
            "reprojection_error": mean_error,
            "successful_images": len(objpoints),
        }

        calib_dir = os.path.join(os.path.dirname(self.base_sync_dir), "results")
        if not os.path.isabs(calib_dir):
            calib_dir = os.path.join(APP_DIR, calib_dir)
        os.makedirs(calib_dir, exist_ok=True)

        out_path = os.path.join(calib_dir, f"{camera_name}_calibration.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(calib_result, f, indent=2)

        self._update_progress(f"{camera_name}: done, reprojection error = {mean_error:.4f}px")
        return calib_result

    def run(self):
        try:
            self._update_progress("Starting stereo calibration...")
            camera_dirs = []

            for i in range(2):
                cam_path = os.path.join(self.base_sync_dir, f"cam{i}")
                if os.path.isdir(cam_path):
                    camera_dirs.append((cam_path, f"camera_{i}"))

            if not camera_dirs:
                raise ValueError("No calibration image folders found (expected calib_images/sync/cam0, cam1)")

            calibration_results = {}
            for cam_dir, cam_name in camera_dirs:
                if self._stop_event.is_set():
                    return
                try:
                    result = self._calibrate_camera(cam_dir, cam_name)
                    if result:
                        calibration_results[cam_name] = result
                except Exception as e:
                    if self.failed_callback:
                        self.failed_callback(f"{cam_name}: calibration failed - {e}")
                    return

            if self.finished_callback:
                self.finished_callback(calibration_results)

        except Exception as e:
            if self.failed_callback:
                self.failed_callback(f"Stereo calibration failed: {e}")
        finally:
            self._update_progress("Stereo calibration thread finished.")

    def stop(self):
        self._stop_event.set()



