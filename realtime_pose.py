import os
import sys
import cv2
import torch
import numpy as np
import threading
import types
import json
import glob
import time
from collections import deque

from gait_analytics import GaitAnalyticsEngine


# Fallback: AlphaPose requires easydict, but environment may not have it installed.
try:
    import easydict  # noqa: F401
except ModuleNotFoundError:
    class EasyDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    easydict_module = types.ModuleType("easydict")
    easydict_module.EasyDict = EasyDict
    sys.modules["easydict"] = easydict_module


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALPHAPOSE_ROOT = os.path.join(CURRENT_DIR, "AlphaPose")

# Ensure OrbbecSDK.dll in project root is discoverable on Windows.
if os.name == "nt":
    os.environ["PATH"] = CURRENT_DIR + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(CURRENT_DIR)
    except (AttributeError, FileNotFoundError, OSError):
        pass

if not os.path.isdir(ALPHAPOSE_ROOT):
    raise FileNotFoundError(f"Cannot find AlphaPose directory: {ALPHAPOSE_ROOT}")

if ALPHAPOSE_ROOT not in sys.path:
    sys.path.insert(0, ALPHAPOSE_ROOT)

# Orbbec SDK
from pyorbbecsdk import Context, Config, OBError, OBSensorType, Pipeline, OBFormat


_ORIGINAL_CWD = os.getcwd()
try:
    os.chdir(ALPHAPOSE_ROOT)
    from AlphaPose.detector.apis import get_detector
    from AlphaPose.alphapose.models import builder
    from AlphaPose.alphapose.utils.config import update_config
    from AlphaPose.alphapose.utils.presets import SimpleTransform
    from AlphaPose.alphapose.utils.transforms import get_func_heatmap_to_coord
    from AlphaPose.alphapose.utils.pPose_nms import pose_nms
    from AlphaPose.alphapose.utils.vis import vis_frame_fast
finally:
    # Do not leak cwd changes to callers (e.g., GUI app).
    os.chdir(_ORIGINAL_CWD)


class Opt:
    # Runtime mode
    sp = True
    detector = "yolo"  # Use existing local weight: detector/yolo/data/yolov3-spp.weights

    # Output and visualization behavior
    save_img = False
    vis = False
    showbox = True
    vis_fast = True
    save_video = False

    # NMS and export
    format = "open"
    min_box_area = 0
    eval = False
    outputpath = "examples/res"

    # Tracking options
    tracking = False
    pose_track = False
    pose_flow = False

    # Batch-ish options expected by detector code
    detbatch = 1
    posebatch = 16
    profile = False

    # Filled at runtime
    device = None
    gpus = None


class OrbbecTimedCameraThread(threading.Thread):
    def __init__(self, device, device_idx, buffer_dict, lock, profile_candidates=None):
        super().__init__(daemon=True)
        self.device = device
        self.device_idx = device_idx
        self.buffer_dict = buffer_dict
        self.lock = lock
        self.profile_candidates = profile_candidates or []
        self.running = True
        self.actual_width = None
        self.actual_height = None
        self.actual_fps = None

    def select_best_profile(self, profile_list):
        for width, height, fps in self.profile_candidates:
            try:
                profile = profile_list.get_video_stream_profile(width, height, OBFormat.MJPG, fps)
                self.actual_width = width
                self.actual_height = height
                self.actual_fps = fps
                return profile
            except Exception:
                continue
        # fallback
        default_profile = profile_list.get_default_video_stream_profile()
        self.actual_width = default_profile.get_width()
        self.actual_height = default_profile.get_height()
        self.actual_fps = default_profile.get_fps()
        return default_profile

    def _extract_frame_timestamp_ms(self, color_frame):
        # Prefer hardware timestamp if available.
        for method in ("get_timestamp", "get_time_stamp", "timestamp", "get_system_timestamp"):
            fn = getattr(color_frame, method, None)
            if fn is None:
                continue
            try:
                val = fn() if callable(fn) else fn
                ts = float(val)
                if ts > 1e12:  # ns/us-like, normalize to ms
                    ts = ts / 1e6
                elif ts > 1e9:  # us-like for recent epoch
                    ts = ts / 1e3
                elif ts < 1e6:  # likely seconds
                    ts = ts * 1000.0
                return ts
            except Exception:
                continue
        # Fallback to host monotonic clock.
        return time.monotonic() * 1000.0

    def run(self):
        config = Config()
        pipeline = None
        try:
            pipeline = Pipeline(self.device)
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = self.select_best_profile(profile_list)
            config.enable_stream(color_profile)
            pipeline.start(config)
            print(f"Camera {self.device_idx} config: {self.actual_width}x{self.actual_height}@{self.actual_fps}")

            while self.running:
                try:
                    frames = pipeline.wait_for_frames(500)
                    if not frames:
                        continue
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    ts_ms = self._extract_frame_timestamp_ms(color_frame)
                    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    if not image.flags.c_contiguous:
                        image = np.ascontiguousarray(image)

                    frame_record = {
                        "ts_ms": ts_ms,
                        "image": image,
                        "host_ms": time.monotonic() * 1000.0,
                    }
                    with self.lock:
                        self.buffer_dict[self.device_idx].append(frame_record)
                except Exception as e:
                    print(f"Camera {self.device_idx} read error: {e}")
                    break
        except Exception as e:
            print(f"Camera {self.device_idx} init failed: {e}")
        finally:
            if pipeline:
                pipeline.stop()
            print(f"Camera {self.device_idx} stopped")


class StereoAlphaPose:
    def __init__(
        self,
        show_3d_window=True,
        show_preview_window=True,
        show_gait_dashboard=True,
        on_3d_canvas=None,
        on_stereo_preview=None,
        on_gait_metrics=None,
        on_gait_dashboard=None,
        on_gait_enable_state=None,
        on_gait_reset_requested=None,
        on_ground_board_calibrate_requested=None,
        on_should_stop=None,
    ):
        self.cfg_path, self.ckpt_path = self._resolve_pose_assets()
        self.cfg = update_config(self.cfg_path)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.opt = Opt()
        self.opt.device = self.device
        self.opt.gpus = [0] if torch.cuda.is_available() else [-1]

        print(f"[AlphaPose] cfg: {self.cfg_path}")
        print(f"[AlphaPose] checkpoint: {self.ckpt_path}")
        print(f"[AlphaPose] device: {self.device}")
        self.show_3d_window = bool(show_3d_window)
        self.show_preview_window = bool(show_preview_window)
        self.on_3d_canvas = on_3d_canvas
        self.on_stereo_preview = on_stereo_preview
        self.on_gait_metrics = on_gait_metrics
        self.on_gait_dashboard = on_gait_dashboard
        self.on_gait_enable_state = on_gait_enable_state
        self.on_gait_reset_requested = on_gait_reset_requested
        self.on_ground_board_calibrate_requested = on_ground_board_calibrate_requested
        self.on_should_stop = on_should_stop

        # Must be ready before _build_detector().
        self.detector_input_dim = 320
        self.detector = self._build_detector()
        self.pose_model = self._build_pose_model()

        self._input_size = self.cfg.DATA_PRESET.IMAGE_SIZE
        self._hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self._norm_type = self.cfg.LOSS.get("NORM_TYPE", None)
        self._heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)
        self._use_heatmap_loss = self.cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss") == "MSELoss"

        self.transformation = self._build_transform()

        # Runtime recovery strategy for long-running sessions.
        self._cuda_error_count = 0
        self._fallback_to_cpu_on_cuda_error = True
        self._cuda_cache_clear_interval = 240

        self.vis_thres = self._build_vis_thresholds(self.cfg)
        self.eval_joints = list(range(self.cfg.DATA_PRESET.NUM_JOINTS))
        # Performance profile for mobile RTX 2060 on dual camera.
        self.max_infer_width = 640
        self.frame_skip = 2 if torch.cuda.is_available() else 3  # Run inference every N frames per camera.
        self.preview_panel_size = (640, 360)  # Per-camera preview (w, h) for lower display overhead.
        self._frame_count = {"CAM0": 0, "CAM1": 0}
        self._last_vis = {"CAM0": None, "CAM1": None}
        # Physical mapping: your current setup has CAM1 on the physical LEFT side.
        self.swap_lr_labels = True
        self.display_label_map = (
            {"CAM0": "RIGHT", "CAM1": "LEFT"}
            if self.swap_lr_labels else
            {"CAM0": "LEFT", "CAM1": "RIGHT"}
        )
        # Prefer lower input resolution first to reduce per-frame decode/inference cost.
        # The thread will auto-fallback to next profile if a mode is unsupported.
        self.capture_profiles = [
            (1280, 720, 60),
            (1280, 720, 30),
            (1280, 720, 15),
            (640, 480, 60),
            (640, 480, 30),
            (640, 480, 15),
            (424, 240, 30),
            (424, 240, 15),
            (320, 240, 30),
            (320, 240, 15),
            (1920, 1080, 15),
            (1920, 1080, 30),
        ]
        # Strict temporal pairing config (for 3D reconstruction safety)
        self.sync_tol_ms = 8.0
        self.sync_buffer_size = 12
        self.sync_drop_left = 0
        self.sync_drop_right = 0
        self.sync_pairs = 0
        self.sync_last_dt = None
        self.latest_kpts = {"CAM0": None, "CAM1": None}
        self.latest_kpt_scores = {"CAM0": None, "CAM1": None}
        self.proj_ready = False
        self.P1 = None
        self.P2 = None
        self._setup_realtime_3d()
        self.gait_analytics = GaitAnalyticsEngine(
            render_dashboard=bool(show_gait_dashboard),
            on_dashboard_frame=self.on_gait_dashboard,
            on_metrics=self.on_gait_metrics,
        )

    @property
    def joint_pairs(self):
        # COCO-17 style pairs; vis_frame only uses this for some model paths.
        return [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    def _resolve_pose_assets(self):
        candidates = [
            (
                "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml",
                "pretrained_models/halpe26_fast_res50_256x192.pth",
            ),
            (
                "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml",
                "pretrained_models/fast_res50_256x192.pth",
            ),
        ]
        for cfg_rel, ckpt_rel in candidates:
            cfg_abs = os.path.join(ALPHAPOSE_ROOT, cfg_rel)
            ckpt_abs = os.path.join(ALPHAPOSE_ROOT, ckpt_rel)
            if os.path.isfile(cfg_abs) and os.path.isfile(ckpt_abs):
                return cfg_abs, ckpt_abs

        raise FileNotFoundError(
            "No valid AlphaPose cfg/checkpoint pair found. "
            "Expected one of: "
            "(configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml + pretrained_models/halpe26_fast_res50_256x192.pth) "
            "or (configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml + pretrained_models/fast_res50_256x192.pth)."
        )

    def _build_detector(self):
        cfg_path = os.path.join(ALPHAPOSE_ROOT, "detector", "yolo", "cfg", "yolov3-spp.cfg")
        weight_path = os.path.join(ALPHAPOSE_ROOT, "detector", "yolo", "data", "yolov3-spp.weights")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"Detector cfg not found: {cfg_path}. "
                "Please prepare detector/yolo/cfg/yolov3-spp.cfg."
            )
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"Detector weight not found: {weight_path}. "
                "Please prepare detector/yolo/data/yolov3-spp.weights or switch detector config."
            )

        # Keep path portable (relative to project root), but resolve to absolute at runtime.
        # This avoids any dependency on current working directory.
        import detector.yolo_cfg as yolo_cfg
        yolo_cfg.cfg.CONFIG = cfg_path
        yolo_cfg.cfg.WEIGHTS = weight_path
        # Shrink detector input to reduce GPU load.
        yolo_cfg.cfg.INP_DIM = int(getattr(self, "detector_input_dim", 416))
        yolo_cfg.cfg.CONFIDENCE = 0.2

        print(f"[AlphaPose] Loading human detector (YOLOv3-SPP, {yolo_cfg.cfg.INP_DIM} input)...")
        return get_detector(self.opt)

    def _setup_realtime_3d(self):
        # Use Halpe-26 joints (exclude face/hand extras by using first 26 directly).
        self.recon_joint_names = [f"J{i}" for i in range(26)]
        self.recon_joint_count = 26

        # Halpe-26 body + foot topology (matches AlphaPose vis for 26 joints).
        # Exclude ears (3/4) in 3D because frontal view makes them unstable.
        self._excluded_joints_3d = {3, 4}
        self.skeleton_edges = [
            (0, 1), (0, 2),
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),
        ]

        self._last_3d_points = None
        self._3d_window_name = "Realtime 3D Skeleton"
        self._3d_canvas_size = (940, 1220)  # H, W; keep height, reduce side whitespace.
        self._3d_center = np.array([700.0, 560.0], dtype=np.float32)
        # Fixed display scale (no auto zoom) for stable depth perception.
        self._3d_scale_x = 130.0
        self._3d_scale_y = 180.0
        self._3d_scale_z = 110.0
        self._3d_margin_px = 80.0
        self._auto_scale_enabled = False
        # Fixed clinical view (right-front) for stable comparison across sessions.
        self._3d_rot_yaw = np.deg2rad(-30.0)
        self._3d_rot_pitch = np.deg2rad(12.0)
        self._current_view_name = "YAW -30 deg"
        self._pts3d_ema = None
        self._ema_alpha = 0.2
        self._save_dir_3d = os.path.join(CURRENT_DIR, "realtime_3d_snapshots")
        os.makedirs(self._save_dir_3d, exist_ok=True)
        self._ground_calib_path = os.path.join(self._save_dir_3d, "ground_calibration.json")
        self._latest_export_points = None
        self._latest_valid_mask = None
        self._latest_export_meta = {}
        self._latest_frames_raw = {"CAM0": None, "CAM1": None}
        self._latest_frames_vis = {"CAM0": None, "CAM1": None}
        self._last_synced_pair = {"CAM0": None, "CAM1": None}
        self._latest_dt_ms = None
        self._last_3d_canvas = None
        self._kpt3d_conf_thresh = 0.25
        self._kpt3d_reproj_thresh_px = 28.0
        self._kpt3d_depth_min_m = 0.4
        self._kpt3d_depth_max_m = 8.0
        self._kpt3d_torso_radius_max_m = 1.8
        self._kpt3d_step_max_m = 0.7
        # Metric axis style in millimeters.
        self._axis_tick_mm = 200
        self._axis_margin_px = 34
        self._axis_bounds_initialized = False
        # When True, axis bounds follow the subject with EMA smoothing.
        self._axis_lock_to_subject = True
        self._axis_follow_alpha = 0.25
        self._axis_span_x_m = 0.95
        self._axis_span_z_m = 1.35
        self._axis_height_m = 2.10
        self._axis_floor_margin_m = 0.08
        self._axis_pad_x_m = 0.14
        self._axis_pad_z_m = 0.20
        self._axis_pad_y_top_m = 0.26
        self._view_zoom = 1.04
        self._hud_height_px = 132
        # Visual compression for depth to avoid an over-stretched Z impression.
        self._axis_visual_x_scale = 0.96
        self._axis_visual_z_scale = 0.72
        # Default to mirrored visual orientation for intuitive left-right viewing.
        self._view_mirror_x = True
        # Fallback bounds before first pose lock.
        self._axis_x_min_m = -4.8
        self._axis_x_max_m = -3.0
        self._axis_y_min_m = 0.8
        self._axis_y_max_m = 3.0
        self._axis_z_min_m = 3.5
        self._axis_z_max_m = 5.7
        self._ground_ready = False
        self._ground_R = np.eye(3, dtype=np.float32)
        self._ground_center = np.zeros((3,), dtype=np.float32)
        self._ground_y_offset = 0.0
        self._ground_method = "none"
        self._ground_board_size = (11, 8)
        self._ground_square_size_m = 0.035
        self._recent_foot_frames = deque(maxlen=600)
        self._ground_min_samples = 50
        self._last_valid_mask_3d = np.ones((self.recon_joint_count,), dtype=bool)
        self._prev_pts3_raw = None

        self._init_projection_from_calibration()
        self._load_ground_calibration()

    def _load_single_calib(self, path):
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        K = np.array(data["camera_matrix"], dtype=np.float64)
        dist = np.array(data["distortion_coeffs"], dtype=np.float64)
        img_size = tuple(int(v) for v in data.get("image_size", [1280, 720]))
        return {"K": K, "dist": dist, "size": img_size}

    def _find_gui_calibration_pair(self):
        # GUI writes to calib_images/results/camera_0_calibration.json etc.
        cands0 = glob.glob(os.path.join(CURRENT_DIR, "**", "camera_0_calibration.json"), recursive=True)
        cands1 = glob.glob(os.path.join(CURRENT_DIR, "**", "camera_1_calibration.json"), recursive=True)
        if not cands0 or not cands1:
            return None, None
        cands0.sort(key=os.path.getmtime, reverse=True)
        cands1.sort(key=os.path.getmtime, reverse=True)
        return cands0[0], cands1[0]

    def _collect_sync_pairs(self):
        cam0_dir = os.path.join(CURRENT_DIR, "calib_images", "sync", "cam0")
        cam1_dir = os.path.join(CURRENT_DIR, "calib_images", "sync", "cam1")
        if not (os.path.isdir(cam0_dir) and os.path.isdir(cam1_dir)):
            return []

        exts = ("*.png", "*.jpg", "*.jpeg")
        left_files = []
        right_files = []
        for ext in exts:
            left_files.extend(glob.glob(os.path.join(cam0_dir, ext)))
            right_files.extend(glob.glob(os.path.join(cam1_dir, ext)))

        left_map = {os.path.basename(p): p for p in left_files}
        right_map = {os.path.basename(p): p for p in right_files}
        common = sorted(set(left_map.keys()) & set(right_map.keys()))
        return [(left_map[name], right_map[name]) for name in common]

    def _estimate_stereo_rt(self, calib0, calib1):
        pairs = self._collect_sync_pairs()
        if len(pairs) < 3:
            raise RuntimeError("Not enough synced chessboard pairs in calib_images/sync/cam0,cam1 (need >=3)")

        board_size = (11, 8)  # keep aligned with your original scripts default
        square_size = 0.035  # meter, aligned with your board square width 35mm
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

        objpoints = []
        imgpoints0 = []
        imgpoints1 = []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for lp, rp in pairs:
            img0 = cv2.imread(lp)
            img1 = cv2.imread(rp)
            if img0 is None or img1 is None:
                continue
            g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            ok0, c0 = cv2.findChessboardCorners(g0, board_size, None)
            ok1, c1 = cv2.findChessboardCorners(g1, board_size, None)
            if not (ok0 and ok1):
                continue
            c0 = cv2.cornerSubPix(g0, c0, (11, 11), (-1, -1), criteria)
            c1 = cv2.cornerSubPix(g1, c1, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints0.append(c0)
            imgpoints1.append(c1)

        if len(objpoints) < 3:
            raise RuntimeError("Not enough valid chessboard detections for stereo R/T estimation (need >=3)")

        img_size = calib0["size"]
        flags = cv2.CALIB_FIX_INTRINSIC
        _, K0, D0, K1, D1, R, T, _, _ = cv2.stereoCalibrate(
            objpoints,
            imgpoints0,
            imgpoints1,
            calib0["K"],
            calib0["dist"],
            calib1["K"],
            calib1["dist"],
            img_size,
            criteria=criteria,
            flags=flags
        )
        return K0, D0, K1, D1, R, T

    def _init_projection_from_calibration(self):
        c0_path, c1_path = self._find_gui_calibration_pair()
        if not c0_path or not c1_path:
            print("[3D] GUI calibration json not found. Please run calibration in run.py first.")
            print(f"[3D] Expected under: {os.path.join(CURRENT_DIR, 'calib_images', 'results')}")
            return

        try:
            calib0 = self._load_single_calib(c0_path)
            calib1 = self._load_single_calib(c1_path)
            K0, D0, K1, D1, R, T = self._estimate_stereo_rt(calib0, calib1)
            self.P1 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
            self.P2 = K1 @ np.hstack([R, T.reshape(3, 1)])
            self.proj_ready = True
            print(f"[3D] Projection ready using {os.path.basename(c0_path)} / {os.path.basename(c1_path)}")
        except Exception as e:
            print(f"[3D] Failed to initialize stereo projection: {e}")
            self.proj_ready = False

    @staticmethod
    def _fit_to_canvas_16_9(frame_bgr, out_w=800, out_h=450):
        if frame_bgr is None:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        src_h, src_w = frame_bgr.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)

        scale = min(out_w / float(src_w), out_h / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        x0 = (out_w - new_w) // 2
        y0 = (out_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _build_pose_model(self):
        print("[AlphaPose] Loading pose model...")
        model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        state_dict = torch.load(self.ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _build_transform(self):
        return SimpleTransform(
            self,
            scale_factor=0,
            add_dpg=False,
            input_size=self._input_size,
            output_size=self._hm_size,
            rot=0,
            sigma=self.cfg.DATA_PRESET.SIGMA,
            train=False,
            gpu_device=self.device,
            loss_type=self.cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss"),
        )

    @staticmethod
    def _is_cuda_runtime_error(exc):
        msg = str(exc).lower()
        cuda_signals = (
            "cuda error",
            "cudnn",
            "cublas",
            "device-side assert",
            "illegal instruction",
            "illegal memory access",
        )
        return any(sig in msg for sig in cuda_signals)

    @staticmethod
    def _make_status_frame(frame_bgr, cam_label, msg, color=(0, 140, 255)):
        draw = frame_bgr.copy()
        if not draw.flags.c_contiguous:
            draw = np.ascontiguousarray(draw)
        cv2.putText(draw, f"{cam_label} | {msg}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return draw

    def _clear_cuda_runtime_state(self):
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def _switch_inference_device(self, device_name):
        target = torch.device(device_name)
        if target.type == self.device.type:
            return
        print(f"[Recovery] Switching inference device: {self.device} -> {target}")
        self.device = target
        self.opt.device = target
        self.opt.gpus = [0] if target.type == "cuda" else [-1]
        self.detector = self._build_detector()
        self.pose_model = self._build_pose_model()
        self.transformation = self._build_transform()

    def _handle_inference_runtime_error(self, exc, cam_id):
        if not self._is_cuda_runtime_error(exc):
            return False
        self._cuda_error_count += 1
        print(f"[Recovery][{cam_id}] CUDA runtime error #{self._cuda_error_count}: {exc}")
        self._clear_cuda_runtime_state()
        if self._fallback_to_cpu_on_cuda_error and self.device.type == "cuda":
            try:
                self._switch_inference_device("cpu")
                print("[Recovery] CUDA fault handled. Switched to CPU inference.")
            except Exception as switch_exc:
                print(f"[Recovery] CPU fallback failed: {switch_exc}")
        return True

    def _rodrigues_align(self, src_n, dst_n):
        src = np.asarray(src_n, dtype=np.float32)
        dst = np.asarray(dst_n, dtype=np.float32)
        src_norm = float(np.linalg.norm(src))
        dst_norm = float(np.linalg.norm(dst))
        if src_norm < 1e-8 or dst_norm < 1e-8:
            return np.eye(3, dtype=np.float32)
        src = src / src_norm
        dst = dst / dst_norm
        v = np.cross(src, dst)
        c = float(np.clip(np.dot(src, dst), -1.0, 1.0))
        s = float(np.linalg.norm(v))
        if s < 1e-8:
            if c > 0.0:
                return np.eye(3, dtype=np.float32)
            # 180-degree rotation: choose any orthogonal axis.
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(src[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            axis = axis - np.dot(axis, src) * src
            axis = axis / (np.linalg.norm(axis) + 1e-9)
            K = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
                dtype=np.float32,
            )
            return np.eye(3, dtype=np.float32) + 2.0 * (K @ K)
        K = np.array(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
            dtype=np.float32,
        )
        R = np.eye(3, dtype=np.float32) + K + (K @ K) * ((1.0 - c) / (s * s + 1e-9))
        return R.astype(np.float32)

    def _fit_plane_svd_robust(self, pts, max_iter=2, inlier_k=2.0):
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape[0] < 3:
            return None, None, None
        inlier_mask = np.ones((pts.shape[0],), dtype=bool)
        center = None
        normal = None
        for _ in range(max_iter):
            use = pts[inlier_mask]
            if use.shape[0] < 3:
                break
            center = np.mean(use, axis=0)
            A = use - center
            _, _, vt = np.linalg.svd(A, full_matrices=False)
            normal = vt[-1]
            nrm = float(np.linalg.norm(normal))
            if nrm < 1e-8:
                break
            normal = normal / nrm
            d = np.abs((pts - center) @ normal)
            med = float(np.median(d))
            sigma = 1.4826 * med + 1e-6
            inlier_mask = d <= (inlier_k * sigma + 0.01)
        if center is None or normal is None:
            return None, None, None
        return center.astype(np.float32), normal.astype(np.float32), inlier_mask

    def _save_ground_calibration(self):
        data = {
            "ground_ready": bool(self._ground_ready),
            "ground_method": str(self._ground_method),
            "board_size": [int(self._ground_board_size[0]), int(self._ground_board_size[1])],
            "square_size_m": float(self._ground_square_size_m),
            "ground_R": self._ground_R.astype(float).tolist(),
            "ground_center": self._ground_center.astype(float).tolist(),
            "ground_y_offset": float(self._ground_y_offset),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self._ground_calib_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_ground_calibration(self):
        if not os.path.isfile(self._ground_calib_path):
            return
        try:
            with open(self._ground_calib_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if bool(data.get("ground_ready", False)):
                self._ground_method = str(data.get("ground_method", "legacy"))
                self._ground_R = np.asarray(data.get("ground_R", np.eye(3)), dtype=np.float32).reshape(3, 3)
                self._ground_center = np.asarray(data.get("ground_center", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
                self._ground_y_offset = float(data.get("ground_y_offset", 0.0))
                self._ground_ready = True
                print(f"[3D] Ground calibration loaded: {self._ground_calib_path}")
        except Exception as e:
            print(f"[3D] Failed to load ground calibration: {e}")

    def _collect_ground_points_from_frame(self, pts_world_raw, valid_mask):
        if pts_world_raw is None or pts_world_raw.shape[0] < 26:
            return
        # Halpe feet endpoints only; avoid knees/ankles biasing the plane.
        foot_ids = [20, 21, 22, 23, 24, 25]
        pts = []
        for jid in foot_ids:
            if jid >= pts_world_raw.shape[0]:
                continue
            if valid_mask is not None and jid < len(valid_mask) and not bool(valid_mask[jid]):
                continue
            p = pts_world_raw[jid]
            if not np.all(np.isfinite(p)):
                continue
            pts.append(p.astype(np.float32))
        if len(pts) >= 3:
            self._recent_foot_frames.append(np.stack(pts, axis=0))

    def _calibrate_ground_from_recent_points(self):
        if len(self._recent_foot_frames) <= 0:
            print(f"[Ground] No foot frame samples yet. Stand still and retry G.")
            return False

        # Use per-frame lowest-contact points to approximate ground contacts.
        frame_lows = []
        for arr in self._recent_foot_frames:
            if arr is None or arr.shape[0] < 3:
                continue
            ys = arr[:, 1]
            if ys.size <= 0:
                continue
            y_thr = float(np.percentile(ys, 45))
            cand = arr[ys <= y_thr]
            if cand.shape[0] >= 1:
                frame_lows.append(cand)
        if len(frame_lows) <= 0:
            print("[Ground] No valid low-foot samples. Stand still and retry G.")
            return False
        pts = np.concatenate(frame_lows, axis=0).astype(np.float32)
        if pts.shape[0] < self._ground_min_samples:
            print(f"[Ground] Not enough samples: {pts.shape[0]}/{self._ground_min_samples}. Stand still and retry G.")
            return False
        center, normal, inlier_mask = self._fit_plane_svd_robust(pts, max_iter=3, inlier_k=2.5)
        if center is None or normal is None:
            print("[Ground] Plane fitting failed.")
            return False
        if normal[1] < 0:
            normal = -normal
        R = self._rodrigues_align(normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        pts_rot = (pts - center) @ R.T
        y_vals = pts_rot[:, 1]
        if inlier_mask is not None and np.any(inlier_mask):
            y0 = float(np.percentile(y_vals[inlier_mask], 10))
            inlier_ratio = float(np.mean(inlier_mask))
        else:
            y0 = float(np.percentile(y_vals, 10))
            inlier_ratio = 0.0

        self._ground_R = R.astype(np.float32)
        self._ground_center = center.astype(np.float32)
        self._ground_y_offset = y0
        self._ground_ready = True
        self._ground_method = "feet"
        self._save_ground_calibration()
        print(
            f"[Ground] Calibrated. low_foot_samples={pts.shape[0]}, inlier_ratio={inlier_ratio:.2f}, "
            f"y_offset={self._ground_y_offset:.4f}m"
        )
        return True

    def _find_chessboard_corners_single(self, gray, board_size):
        corners = None
        ok = False
        # Prefer SB detector when available; fallback to legacy.
        if hasattr(cv2, "findChessboardCornersSB"):
            flags = 0
            for name in ("CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY", "CALIB_CB_NORMALIZE_IMAGE"):
                flags |= int(getattr(cv2, name, 0))
            try:
                ok, corners = cv2.findChessboardCornersSB(gray, board_size, flags=flags)
            except Exception:
                ok = False
                corners = None
        if not ok or corners is None:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ok, corners = cv2.findChessboardCorners(gray, board_size, flags)
            if ok and corners is not None:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return bool(ok), corners

    def _calibrate_ground_from_board_pair(self, frame_cam0_bgr, frame_cam1_bgr):
        if not self.proj_ready or self.P1 is None or self.P2 is None:
            print("[GroundBoard] Projection is not ready. Run camera calibration first.")
            return False
        if frame_cam0_bgr is None or frame_cam1_bgr is None:
            print("[GroundBoard] Missing camera frames.")
            return False

        board_size = self._ground_board_size
        sq = float(self._ground_square_size_m)
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * sq

        g0 = cv2.cvtColor(frame_cam0_bgr, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frame_cam1_bgr, cv2.COLOR_BGR2GRAY)
        ok0, c0 = self._find_chessboard_corners_single(g0, board_size)
        ok1, c1 = self._find_chessboard_corners_single(g1, board_size)
        if not (ok0 and ok1):
            print("[GroundBoard] Chessboard corners not found in both cameras.")
            return False

        pts_l = c0.reshape(-1, 2).T.astype(np.float64)
        pts_r = c1.reshape(-1, 2).T.astype(np.float64)
        pts4 = cv2.triangulatePoints(self.P1, self.P2, pts_l, pts_r)
        w = pts4[3, :]
        w[np.abs(w) < 1e-9] = 1e-9
        pts3 = (pts4[:3, :] / w).T.astype(np.float32)
        pts_world = self._to_world_axes(pts3)

        center, normal, inlier_mask = self._fit_plane_svd_robust(pts_world, max_iter=3, inlier_k=2.5)
        if center is None or normal is None:
            print("[GroundBoard] Plane fitting failed from board points.")
            return False
        if normal[1] < 0:
            normal = -normal
        R = self._rodrigues_align(normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        pts_rot = (pts_world - center) @ R.T
        y_vals = pts_rot[:, 1]
        if inlier_mask is not None and np.any(inlier_mask):
            y0 = float(np.median(y_vals[inlier_mask]))
            inlier_ratio = float(np.mean(inlier_mask))
        else:
            y0 = float(np.median(y_vals))
            inlier_ratio = 0.0

        self._ground_R = R.astype(np.float32)
        self._ground_center = center.astype(np.float32)
        self._ground_y_offset = y0
        self._ground_ready = True
        self._ground_method = "board"
        self._save_ground_calibration()
        print(
            f"[GroundBoard] Calibrated from chessboard. points={pts_world.shape[0]}, "
            f"inlier_ratio={inlier_ratio:.2f}, y_offset={self._ground_y_offset:.4f}m"
        )
        return True

    def _apply_ground_transform(self, pts_world):
        if pts_world is None:
            return None
        pts = np.asarray(pts_world, dtype=np.float32)
        if not self._ground_ready:
            return pts
        pts_t = (pts - self._ground_center) @ self._ground_R.T
        pts_t[:, 1] -= self._ground_y_offset
        return pts_t.astype(np.float32)

    def _reset_ground_calibration(self):
        self._ground_ready = False
        self._ground_method = "none"
        self._ground_R = np.eye(3, dtype=np.float32)
        self._ground_center = np.zeros((3,), dtype=np.float32)
        self._ground_y_offset = 0.0
        self._recent_foot_frames.clear()
        print("[Ground] Calibration reset. Press G to recalibrate ground.")

    @staticmethod
    def _build_vis_thresholds(cfg):
        num_joints = cfg.DATA_PRESET.NUM_JOINTS
        loss_type = cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss")
        if loss_type == "MSELoss":
            return [0.4] * num_joints
        if "JointRegression" in loss_type:
            return [0.05] * num_joints
        if loss_type == "Combined":
            hand_face_num = 42 if num_joints == 68 else 110
            return [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num
        return [0.4] * num_joints

    def _run_detection(self, frame_bgr):
        img_k = self.detector.image_preprocess(frame_bgr)

        if isinstance(img_k, np.ndarray):
            img_k = torch.from_numpy(img_k)
        if img_k.dim() == 3:
            img_k = img_k.unsqueeze(0)

        im_dim = (frame_bgr.shape[1], frame_bgr.shape[0])
        im_dim_list = torch.FloatTensor(im_dim).repeat(1, 2)

        with torch.no_grad():
            dets = self.detector.images_detection(img_k, im_dim_list)

        if isinstance(dets, int) or dets is None:
            return None

        if isinstance(dets, np.ndarray):
            dets = torch.from_numpy(dets)

        if dets.shape[0] == 0:
            return None

        dets = dets.cpu()
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]
        ids = torch.zeros(scores.shape)

        mask = dets[:, 0] == 0
        boxes_k = boxes[mask]
        scores_k = scores[mask]
        ids_k = ids[mask]

        if boxes_k.nelement() == 0:
            return None

        return boxes_k, scores_k, ids_k

    def _run_pose(self, frame_bgr, det_out):
        boxes, scores, ids = det_out

        # Keep the same color convention as AlphaPose webcam loader.
        orig_img_rgb = frame_bgr[:, :, ::-1]

        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)

        for i, box in enumerate(boxes):
            inps[i], cropped_box = self.transformation.test_transform(orig_img_rgb, box)
            cropped_boxes[i] = torch.FloatTensor(cropped_box)

        inps = inps.to(self.device)
        with torch.no_grad():
            hm = self.pose_model(inps)
        if hm.is_cuda:
            hm = hm.cpu()

        pose_coords = []
        pose_scores = []

        face_hand_num = 110
        for i in range(hm.shape[0]):
            bbox = cropped_boxes[i].tolist()

            if isinstance(self._heatmap_to_coord, list):
                if hm.size(1) == 68:
                    face_hand_num = 42

                body_coords, body_scores = self._heatmap_to_coord[0](
                    hm[i][self.eval_joints[:-face_hand_num]], bbox, hm_shape=self._hm_size, norm_type=self._norm_type
                )
                face_coords, face_scores = self._heatmap_to_coord[1](
                    hm[i][self.eval_joints[-face_hand_num:]], bbox, hm_shape=self._hm_size, norm_type=self._norm_type
                )
                pose_coord = np.concatenate((body_coords, face_coords), axis=0)
                pose_score = np.concatenate((body_scores, face_scores), axis=0)
            else:
                pose_coord, pose_score = self._heatmap_to_coord(
                    hm[i][self.eval_joints], bbox, hm_shape=self._hm_size, norm_type=self._norm_type
                )

            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)

        boxes_nms, scores_nms, ids_nms, preds_img_nms, preds_scores_nms, _ = pose_nms(
            boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area, use_heatmap_loss=self._use_heatmap_loss
        )

        result_items = []
        for k in range(len(scores_nms)):
            result_items.append(
                {
                    "keypoints": preds_img_nms[k],
                    "kp_score": preds_scores_nms[k],
                    "proposal_score": torch.mean(preds_scores_nms[k]) + scores_nms[k] + 1.25 * max(preds_scores_nms[k]),
                    "idx": ids_nms[k],
                    "box": [
                        boxes_nms[k][0],
                        boxes_nms[k][1],
                        boxes_nms[k][2] - boxes_nms[k][0],
                        boxes_nms[k][3] - boxes_nms[k][1],
                    ],
                }
            )

        # Select one person (highest proposal score) for 3D triangulation stability.
        kpts_xy = None
        kpts_score = None
        if result_items:
            best = max(result_items, key=lambda x: float(x["proposal_score"]))
            kpts = best["keypoints"]
            ksc = best["kp_score"]
            if torch.is_tensor(kpts):
                kpts_xy = kpts[:, :2].cpu().numpy()
            else:
                kpts_xy = np.asarray(kpts)[:, :2]
            if torch.is_tensor(ksc):
                kpts_score = ksc.squeeze(-1).cpu().numpy()
            else:
                kpts_score = np.asarray(ksc).reshape(-1)
        return orig_img_rgb, {"imgname": "frame.jpg", "result": result_items}, kpts_xy, kpts_score

    def _triangulate_keypoints(self, kpts_l, kpts_r, s_l=None, s_r=None):
        if not self.proj_ready or kpts_l is None or kpts_r is None:
            return None, None
        n = min(len(kpts_l), len(kpts_r), self.recon_joint_count)
        if n < self.recon_joint_count:
            return None, None
        pts_l = kpts_l[:self.recon_joint_count].T.astype(np.float64)
        pts_r = kpts_r[:self.recon_joint_count].T.astype(np.float64)
        pts4 = cv2.triangulatePoints(self.P1, self.P2, pts_l, pts_r)
        w = pts4[3, :]
        w[np.abs(w) < 1e-9] = 1e-9
        pts3 = (pts4[:3, :] / w).T
        if s_l is not None and s_r is not None:
            conf = np.minimum(np.asarray(s_l[:self.recon_joint_count]), np.asarray(s_r[:self.recon_joint_count]))
            valid = conf >= self._kpt3d_conf_thresh
        else:
            valid = np.ones((self.recon_joint_count,), dtype=bool)
        valid = self._refine_3d_valid_mask(pts3, valid, pts_l.T, pts_r.T)
        for jid in self._excluded_joints_3d:
            if 0 <= jid < len(valid):
                valid[jid] = False
        return pts3, valid

    def _refine_3d_valid_mask(self, pts3, valid, pts2_l=None, pts2_r=None):
        if pts3 is None:
            return valid
        valid = np.asarray(valid, dtype=bool).copy()
        n = min(len(valid), pts3.shape[0], self.recon_joint_count)
        valid = valid[:n]
        p = pts3[:n]

        finite = np.all(np.isfinite(p), axis=1)
        valid &= finite
        if not np.any(valid):
            return valid

        # Depth sanity in camera frame.
        z = p[:, 2]
        valid &= (z >= self._kpt3d_depth_min_m) & (z <= self._kpt3d_depth_max_m)

        # Reprojection error sanity (pixel domain).
        if pts2_l is not None and pts2_r is not None:
            pts_h = np.hstack([p, np.ones((n, 1), dtype=np.float64)]).T
            proj_l = self.P1 @ pts_h
            proj_r = self.P2 @ pts_h
            eps = 1e-9
            proj_l_xy = (proj_l[:2, :] / np.clip(proj_l[2:3, :], eps, None)).T
            proj_r_xy = (proj_r[:2, :] / np.clip(proj_r[2:3, :], eps, None)).T
            e_l = np.linalg.norm(proj_l_xy - np.asarray(pts2_l[:n], dtype=np.float64), axis=1)
            e_r = np.linalg.norm(proj_r_xy - np.asarray(pts2_r[:n], dtype=np.float64), axis=1)
            reproj_ok = np.maximum(e_l, e_r) <= self._kpt3d_reproj_thresh_px
            valid &= reproj_ok

        # Torso-radius sanity in world axes to remove explosive outliers.
        if n > 19 and np.any(valid):
            pw = self._to_world_axes(p)
            torso_ids = [5, 6, 11, 12, 18, 19]
            torso_ids = [i for i in torso_ids if i < n and valid[i]]
            if len(torso_ids) >= 3:
                center = np.mean(pw[torso_ids], axis=0)
                r = np.linalg.norm(pw - center, axis=1)
                valid &= (r <= self._kpt3d_torso_radius_max_m)

        # Temporal jump sanity for each joint to reduce frame-to-frame spikes.
        if self._prev_pts3_raw is not None and self._prev_pts3_raw.shape[0] >= n:
            step = np.linalg.norm(p - self._prev_pts3_raw[:n], axis=1)
            valid &= (step <= self._kpt3d_step_max_m)
        self._prev_pts3_raw = p.copy()

        return valid

    def _to_world_axes(self, pts3):
        # x=lateral, y=height, z=depth
        return np.stack([pts3[:, 0], -pts3[:, 1], pts3[:, 2]], axis=1).astype(np.float32)

    def _upright_and_ground(self, pts_world):
        return self._apply_ground_transform(pts_world)

    def _save_current_snapshot_bundle(self):
        if self._latest_export_points is None:
            print("[SAVE] No valid 3D points to save yet.")
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        bundle_dir = os.path.join(self._save_dir_3d, f"snapshot_{ts}")
        os.makedirs(bundle_dir, exist_ok=True)

        # 1) Save 3D coordinates
        pts3d = self._latest_export_points.astype(np.float64)
        csv3d = os.path.join(bundle_dir, "pose3d_world.csv")
        npy3d = os.path.join(bundle_dir, "pose3d_world.npy")
        rows3d = np.column_stack([np.arange(pts3d.shape[0]), pts3d])
        np.savetxt(csv3d, rows3d, delimiter=",", header="joint_id,x_lateral,y_height,z_depth", comments="")
        np.save(npy3d, pts3d)
        if self._latest_valid_mask is not None:
            mask_path = os.path.join(bundle_dir, "pose3d_valid_mask.csv")
            rows_mask = np.column_stack([np.arange(len(self._latest_valid_mask)), self._latest_valid_mask.astype(np.int32)])
            np.savetxt(mask_path, rows_mask, delimiter=",", header="joint_id,valid", comments="")

        # 2) Save 2D coordinates for both cameras (original-frame pixel coordinates)
        for cam in ("CAM0", "CAM1"):
            kpts2d = self.latest_kpts.get(cam, None)
            if kpts2d is not None:
                kpts2d = np.asarray(kpts2d, dtype=np.float64)
                disp = self.display_label_map[cam].lower()
                csv2d = os.path.join(bundle_dir, f"pose2d_{disp}_from_{cam.lower()}.csv")
                rows2d = np.column_stack([np.arange(kpts2d.shape[0]), kpts2d[:, :2]])
                np.savetxt(csv2d, rows2d, delimiter=",", header="joint_id,x,y", comments="")
                np.save(os.path.join(bundle_dir, f"pose2d_{disp}_from_{cam.lower()}.npy"), kpts2d[:, :2])

        # 3) Save screenshots / frames
        for cam in ("CAM0", "CAM1"):
            disp = self.display_label_map[cam].lower()
            if self._latest_frames_raw[cam] is not None:
                cv2.imwrite(os.path.join(bundle_dir, f"frame_{disp}_raw_from_{cam.lower()}.png"), self._latest_frames_raw[cam])
            if self._latest_frames_vis[cam] is not None:
                cv2.imwrite(os.path.join(bundle_dir, f"frame_{disp}_vis_from_{cam.lower()}.png"), self._latest_frames_vis[cam])
        if self._last_3d_canvas is not None:
            cv2.imwrite(os.path.join(bundle_dir, "realtime_3d_canvas.png"), self._last_3d_canvas)

        # 4) Save metadata
        meta = {
            "timestamp": ts,
            "sync_dt_ms": self._latest_dt_ms,
            "num_3d_joints": int(pts3d.shape[0]),
            "display_label_map": self.display_label_map,
            "ground_calibration": {
                "ready": bool(self._ground_ready),
                "method": str(self._ground_method),
                "center": self._ground_center.astype(float).tolist(),
                "y_offset": float(self._ground_y_offset),
                "board_size": [int(self._ground_board_size[0]), int(self._ground_board_size[1])],
                "square_size_m": float(self._ground_square_size_m),
            },
            "profile": {
                "scale_x": self._3d_scale_x,
                "scale_y": self._3d_scale_y,
                "scale_z": self._3d_scale_z,
                "axis_visual_x_scale": self._axis_visual_x_scale,
                "axis_visual_z_scale": self._axis_visual_z_scale,
                "ema_alpha": self._ema_alpha,
                "sync_tol_ms": self.sync_tol_ms,
                "detector_input_dim": int(self.detector_input_dim),
                "max_infer_width": int(self.max_infer_width),
                "frame_skip": int(self.frame_skip),
                "capture_profiles": [list(p) for p in self.capture_profiles],
                "preview_panel_size": [int(self.preview_panel_size[0]), int(self.preview_panel_size[1])],
                "canvas_size_hw": [int(self._3d_canvas_size[0]), int(self._3d_canvas_size[1])],
                "excluded_joints_3d": sorted(int(i) for i in self._excluded_joints_3d),
                "view_yaw_deg": float(np.degrees(self._3d_rot_yaw)),
                "view_pitch_deg": float(np.degrees(self._3d_rot_pitch)),
                "view_mirror_x": bool(self._view_mirror_x),
            },
            "gait_live": self.gait_analytics.live_metrics,
            "gait_periodic_report": self.gait_analytics.periodic_report,
        }
        with open(os.path.join(bundle_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"[SAVE] Snapshot bundle saved: {bundle_dir}")

    def _project_3d_to_canvas(self, pts3, valid_mask=None):
        # pts3: (N, 3), in camera metric space.
        pts_world = self._to_world_axes(pts3)
        pts_world = self._upright_and_ground(pts_world)

        cy = np.cos(self._3d_rot_yaw)
        sy = np.sin(self._3d_rot_yaw)
        cp = np.cos(self._3d_rot_pitch)
        sp = np.sin(self._3d_rot_pitch)

        # Yaw around vertical Y axis, then pitch around X.
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
        rot = rx @ ry

        pts = pts_world @ rot.T
        # Normalize around pelvis/torso center for stable viewport.
        if valid_mask is None:
            valid_mask = np.ones((pts.shape[0],), dtype=bool)
        torso_ids = [11, 12, 18, 19, 5, 6]
        center_ids = [i for i in torso_ids if i < len(valid_mask) and bool(valid_mask[i])]
        if len(center_ids) >= 2:
            root = np.mean(pts[center_ids], axis=0)
        else:
            root = np.mean(pts[np.asarray(valid_mask, dtype=bool)], axis=0) if np.any(valid_mask) else np.zeros((3,), dtype=np.float32)
        pts = pts - root
        uv = np.zeros((pts.shape[0], 2), dtype=np.int32)
        x2 = pts[:, 0] * self._3d_scale_x
        y2 = pts[:, 1] * self._3d_scale_y
        z2 = pts[:, 2] * self._3d_scale_z
        uv[:, 0] = (self._3d_center[0] + x2).astype(np.int32)
        uv[:, 1] = (self._3d_center[1] - y2).astype(np.int32)
        return uv

    def _update_auto_scale(self, pts3):
        if pts3 is None or pts3.shape[0] == 0 or not self._auto_scale_enabled:
            return
        pts_world = np.stack([pts3[:, 0], -pts3[:, 1], pts3[:, 2]], axis=1).astype(np.float32)
        cy = np.cos(self._3d_rot_yaw)
        sy = np.sin(self._3d_rot_yaw)
        cp = np.cos(self._3d_rot_pitch)
        sp = np.sin(self._3d_rot_pitch)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
        rot = rx @ ry
        pts = pts_world @ rot.T
        root = (pts[11] + pts[12]) * 0.5 if pts.shape[0] > 12 else np.mean(pts, axis=0)
        pts = pts - root

        max_abs_x = float(np.max(np.abs(pts[:, 0])) + 1e-6)
        max_abs_y = float(np.max(np.abs(pts[:, 1])) + 1e-6)
        h, w = self._3d_canvas_size
        avail_x = max((w * 0.5) - self._3d_margin_px, 20.0)
        avail_y = max((h * 0.5) - self._3d_margin_px, 20.0)
        scale_x = avail_x / max_abs_x
        scale_y = avail_y / max_abs_y
        target_scale = max(20.0, min(scale_x, scale_y))
        # legacy path (disabled by default)
        self._3d_scale_x = 0.85 * self._3d_scale_x + 0.15 * target_scale
        self._3d_scale_y = 0.85 * self._3d_scale_y + 0.15 * target_scale

    def _get_view_rot_matrix(self):
        cy = np.cos(self._3d_rot_yaw)
        sy = np.sin(self._3d_rot_yaw)
        cp = np.cos(self._3d_rot_pitch)
        sp = np.sin(self._3d_rot_pitch)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
        return rx @ ry

    def _try_initialize_axis_bounds(self, pts_world, valid_mask):
        if pts_world is None:
            return
        n = pts_world.shape[0]
        vm = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else np.ones((n,), dtype=bool)
        if vm.shape[0] > n:
            vm = vm[:n]
        elif vm.shape[0] < n:
            vm = np.pad(vm, (0, n - vm.shape[0]), constant_values=False)
        if not np.any(vm):
            return
        p = pts_world[:n][vm]
        if p.shape[0] < 4:
            return

        foot_ids = [20, 21, 22, 23, 24, 25, 15, 16]
        valid_foot = [i for i in foot_ids if i < n and vm[i]]
        if valid_foot:
            y_floor = float(np.min(pts_world[valid_foot, 1])) - self._axis_floor_margin_m
        else:
            y_floor = float(np.min(p[:, 1])) - self._axis_floor_margin_m

        x_min_t = float(np.min(p[:, 0]) - self._axis_pad_x_m)
        x_max_t = float(np.max(p[:, 0]) + self._axis_pad_x_m)
        z_min_t = float(np.min(p[:, 2]) - self._axis_pad_z_m)
        z_max_t = float(np.max(p[:, 2]) + self._axis_pad_z_m)
        y_min_t = y_floor
        y_max_t = float(np.max(p[:, 1]) + self._axis_pad_y_top_m)

        x_center = 0.5 * (x_min_t + x_max_t)
        z_center = 0.5 * (z_min_t + z_max_t)
        x_span = max(self._axis_span_x_m, x_max_t - x_min_t, 1e-3)
        z_span = max(self._axis_span_z_m, z_max_t - z_min_t, 1e-3)
        y_span = max(self._axis_height_m, y_max_t - y_min_t, 1e-3)
        x_min_t = x_center - 0.5 * x_span
        x_max_t = x_center + 0.5 * x_span
        z_min_t = z_center - 0.5 * z_span
        z_max_t = z_center + 0.5 * z_span
        y_max_t = y_min_t + y_span

        if not self._axis_bounds_initialized:
            self._axis_x_min_m = x_min_t
            self._axis_x_max_m = x_max_t
            self._axis_y_min_m = y_min_t
            self._axis_y_max_m = y_max_t
            self._axis_z_min_m = z_min_t
            self._axis_z_max_m = z_max_t
            self._axis_bounds_initialized = True
            return

        if not self._axis_lock_to_subject:
            return

        a = float(np.clip(self._axis_follow_alpha, 0.0, 1.0))
        ia = 1.0 - a
        self._axis_x_min_m = ia * self._axis_x_min_m + a * x_min_t
        self._axis_x_max_m = ia * self._axis_x_max_m + a * x_max_t
        self._axis_y_min_m = ia * self._axis_y_min_m + a * y_min_t
        self._axis_y_max_m = ia * self._axis_y_max_m + a * y_max_t
        self._axis_z_min_m = ia * self._axis_z_min_m + a * z_min_t
        self._axis_z_max_m = ia * self._axis_z_max_m + a * z_max_t

    def _make_bbox_corners_world(self):
        xs = [self._axis_x_min_m, self._axis_x_max_m]
        ys = [self._axis_y_min_m, self._axis_y_max_m]
        zs = [self._axis_z_min_m, self._axis_z_max_m]
        corners = []
        for x in xs:
            for y in ys:
                for z in zs:
                    corners.append([x, y, z])
        return np.asarray(corners, dtype=np.float32)

    def _build_view_params(self):
        corners = self._make_bbox_corners_world()
        center = np.array(
            [
                0.5 * (self._axis_x_min_m + self._axis_x_max_m),
                0.5 * (self._axis_y_min_m + self._axis_y_max_m),
                0.5 * (self._axis_z_min_m + self._axis_z_max_m),
            ],
            dtype=np.float32,
        )
        rot = self._get_view_rot_matrix()
        c_view = self._apply_visual_axis_scale(corners, center) @ rot.T
        min_x = float(np.min(c_view[:, 0]))
        max_x = float(np.max(c_view[:, 0]))
        min_y = float(np.min(c_view[:, 1]))
        max_y = float(np.max(c_view[:, 1]))
        h, w = self._3d_canvas_size
        hud_h = int(max(0, min(self._hud_height_px, h - 20)))
        avail_w = max(10.0, float(w - 2 * self._axis_margin_px))
        avail_h = max(10.0, float(h - hud_h - 2 * self._axis_margin_px))
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)
        s = min(avail_w / span_x, avail_h / span_y)
        s *= float(max(self._view_zoom, 0.1))
        ox = (w - s * (min_x + max_x)) * 0.5
        y_top = float(hud_h + self._axis_margin_px)
        y_bottom = float(h - self._axis_margin_px)
        y_center = 0.5 * (y_top + y_bottom)
        oy = y_center + 0.5 * s * (min_y + max_y)
        return center, rot, s, ox, oy

    def _apply_visual_axis_scale(self, pts_world, center):
        pv = np.asarray(pts_world, dtype=np.float32) - center
        pv = pv.copy()
        if self._view_mirror_x:
            pv[:, 0] *= -1.0
        pv[:, 0] *= float(self._axis_visual_x_scale)
        pv[:, 2] *= float(self._axis_visual_z_scale)
        return pv

    def _project_world_to_canvas(self, pts_world, center, rot, scale, ox, oy):
        pv = self._apply_visual_axis_scale(pts_world, center) @ rot.T
        uv = np.zeros((pv.shape[0], 2), dtype=np.int32)
        uv[:, 0] = np.round(pv[:, 0] * scale + ox).astype(np.int32)
        uv[:, 1] = np.round(oy - pv[:, 1] * scale).astype(np.int32)
        return uv

    def _draw_metric_axes(self, canvas, center, rot, scale, ox, oy):
        # Draw only metric axis lines and ticks to avoid frame edges occluding the skeleton.
        tick_m = self._axis_tick_mm / 1000.0
        x0, x1 = self._axis_x_min_m, self._axis_x_max_m
        y0, y1 = self._axis_y_min_m, self._axis_y_max_m
        z0, z1 = self._axis_z_min_m, self._axis_z_max_m

        axis_color = (115, 115, 115)
        tick_color = (145, 145, 145)
        text_color = (70, 70, 70)

        def draw_line(pw1, pw2, color, th=1):
            pts = np.asarray([pw1, pw2], dtype=np.float32)
            uv = self._project_world_to_canvas(pts, center, rot, scale, ox, oy)
            cv2.line(canvas, tuple(uv[0]), tuple(uv[1]), color, th, lineType=cv2.LINE_AA)

        x_ticks = np.arange(np.floor(x0 / tick_m) * tick_m, x1 + 0.5 * tick_m, tick_m)
        z_ticks = np.arange(np.floor(z0 / tick_m) * tick_m, z1 + 0.5 * tick_m, tick_m)
        y_ticks = np.arange(np.floor(y0 / tick_m) * tick_m, y1 + 0.5 * tick_m, tick_m)

        # Main axis lines from one origin corner.
        draw_line([x0, y0, z0], [x1, y0, z0], axis_color, 2)  # X axis
        draw_line([x0, y0, z0], [x0, y1, z0], axis_color, 2)  # Y axis
        draw_line([x0, y0, z0], [x0, y0, z1], axis_color, 2)  # Z axis

        tick_len = max(0.03, 0.05 * min(max(x1 - x0, 1e-3), max(y1 - y0, 1e-3), max(z1 - z0, 1e-3)))

        # Tick marks + labels on each axis.
        for xt in x_ticks:
            draw_line([xt, y0, z0], [xt, y0 + tick_len, z0], tick_color, 1)
            p = self._project_world_to_canvas(np.asarray([[xt, y0 + tick_len, z0]], dtype=np.float32), center, rot, scale, ox, oy)[0]
            cv2.putText(canvas, f"{int(round(xt * 1000.0))}", (p[0] - 18, p[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        for zt in z_ticks:
            draw_line([x0, y0, zt], [x0, y0 + tick_len, zt], tick_color, 1)
            p = self._project_world_to_canvas(np.asarray([[x0, y0 + tick_len, zt]], dtype=np.float32), center, rot, scale, ox, oy)[0]
            cv2.putText(canvas, f"{int(round(zt * 1000.0))}", (p[0] + 4, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        for yt in y_ticks:
            draw_line([x0, yt, z0], [x0 + tick_len, yt, z0], tick_color, 1)
            p = self._project_world_to_canvas(np.asarray([[x0 + tick_len, yt, z0]], dtype=np.float32), center, rot, scale, ox, oy)[0]
            cv2.putText(canvas, f"{int(round(yt * 1000.0))}", (p[0] + 6, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        # Axis titles
        p_x = self._project_world_to_canvas(np.asarray([[x1, y0, z0]], dtype=np.float32), center, rot, scale, ox, oy)[0]
        p_z = self._project_world_to_canvas(np.asarray([[x0, y0, z1]], dtype=np.float32), center, rot, scale, ox, oy)[0]
        p_y = self._project_world_to_canvas(np.asarray([[x0, y1, z0]], dtype=np.float32), center, rot, scale, ox, oy)[0]
        cv2.putText(canvas, "Spatial X (mm)", (p_x[0] + 8, p_x[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Depth Z (mm)", (p_z[0] + 8, p_z[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 2, cv2.LINE_AA)
        y_label_x = int(p_y[0] + 8)
        y_label_y = int(max(p_y[1] - 8, 172))
        cv2.putText(canvas, "Height Y (mm)", (y_label_x, y_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 2, cv2.LINE_AA)

    def _draw_3d_hud(self, canvas):
        hud_h = int(max(0, min(self._hud_height_px, canvas.shape[0] - 20)))
        # Draw a solid panel to keep axis/ticks from occluding the HUD text.
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], hud_h), (242, 242, 242), -1)
        cv2.line(canvas, (0, hud_h), (canvas.shape[1], hud_h), (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Realtime 3D Skeleton (Metric Axes)", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (28, 28, 28), 2)
        cv2.putText(canvas, "G: feet ground | B: board ground | R: reset | P: save", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 50, 50), 1)
        status = "ON" if self._ground_ready else "OFF"
        cv2.putText(
            canvas,
            f"GroundCalib: {status} ({self._ground_method}) | foot_frames={len(self._recent_foot_frames)}",
            (18, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 80, 80),
            1,
        )
        cv2.putText(
            canvas,
            f"MirrorX: {'ON' if self._view_mirror_x else 'OFF'}",
            (330, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (90, 90, 90),
            1,
        )

    def _update_3d_plot(self, points_3d, valid_mask=None):
        if points_3d is not None:
            # EMA smoothing in 3D to suppress frame jitter.
            if self._pts3d_ema is None or self._pts3d_ema.shape != points_3d.shape:
                self._pts3d_ema = points_3d.copy()
            else:
                if valid_mask is None:
                    vm = np.ones((points_3d.shape[0],), dtype=bool)
                else:
                    vm = np.asarray(valid_mask, dtype=bool)
                self._pts3d_ema[vm] = self._ema_alpha * points_3d[vm] + (1.0 - self._ema_alpha) * self._pts3d_ema[vm]
            self._last_3d_points = self._pts3d_ema.copy()
            self._update_auto_scale(self._last_3d_points)
        pts = self._last_3d_points
        if valid_mask is None:
            valid_mask = np.ones((self.recon_joint_count,), dtype=bool)
        valid_mask = np.asarray(valid_mask, dtype=bool).copy()
        for jid in self._excluded_joints_3d:
            if 0 <= jid < len(valid_mask):
                valid_mask[jid] = False
        self._last_valid_mask_3d = valid_mask.copy()
        h, w = self._3d_canvas_size
        canvas = np.full((h, w, 3), 242, dtype=np.uint8)

        if pts is not None and pts.shape[0] >= self.recon_joint_count:
            pts_world = self._upright_and_ground(self._to_world_axes(pts[:self.recon_joint_count]))
            self._try_initialize_axis_bounds(pts_world, valid_mask)
            center, rot, scale, ox, oy = self._build_view_params()
            self._draw_metric_axes(canvas, center, rot, scale, ox, oy)
            uv = self._project_world_to_canvas(pts_world, center, rot, scale, ox, oy)
            for a, b in self.skeleton_edges:
                if not (valid_mask[a] and valid_mask[b]):
                    continue
                p1 = tuple(uv[a])
                p2 = tuple(uv[b])
                cv2.line(canvas, p1, p2, (40, 120, 220), 3, lineType=cv2.LINE_AA)
            for i in range(self.recon_joint_count):
                if not valid_mask[i]:
                    continue
                cv2.circle(canvas, tuple(uv[i]), 8, (20, 20, 20), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, tuple(uv[i]), 6, (0, 170, 155), -1, lineType=cv2.LINE_AA)
        else:
            cv2.putText(canvas, "No valid 3D points yet", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

        self._draw_3d_hud(canvas)

        if self.show_3d_window:
            cv2.namedWindow(self._3d_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._3d_window_name, self._3d_canvas_size[1], self._3d_canvas_size[0])
            cv2.imshow(self._3d_window_name, canvas)
        self._last_3d_canvas = canvas.copy()
        if callable(self.on_3d_canvas):
            try:
                self.on_3d_canvas(self._last_3d_canvas.copy())
            except Exception:
                pass

    def _pop_synchronized_pair(self, buffers):
        # buffers: {0: deque, 1: deque}
        left_q = buffers[0]
        right_q = buffers[1]
        if not left_q or not right_q:
            return None, None, None

        while left_q and right_q:
            l = left_q[0]
            r = right_q[0]
            dt = l["ts_ms"] - r["ts_ms"]
            adt = abs(dt)
            if adt <= self.sync_tol_ms:
                left_q.popleft()
                right_q.popleft()
                self.sync_pairs += 1
                self.sync_last_dt = dt
                return l, r, dt
            # Drop the older one, keep strict temporal consistency.
            if dt < 0:
                left_q.popleft()
                self.sync_drop_left += 1
            else:
                right_q.popleft()
                self.sync_drop_right += 1
        return None, None, None

    def infer_and_draw(self, frame_bgr, cam_id):
        self._frame_count[cam_id] += 1
        cam_label = self.display_label_map[cam_id]

        work_frame = frame_bgr
        h, w = frame_bgr.shape[:2]
        resize_scale = 1.0
        if w > self.max_infer_width:
            resize_scale = self.max_infer_width / float(w)
            new_size = (self.max_infer_width, int(h * resize_scale))
            work_frame = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)

        # Skip frames to keep up with realtime; reuse last rendered frame.
        if self.frame_skip > 1 and (self._frame_count[cam_id] % self.frame_skip) != 0:
            cached = self._last_vis.get(cam_id)
            if cached is not None:
                draw_cached = cached.copy()
                cv2.putText(draw_cached, f"{cam_label} | cached", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
                return draw_cached

        if self.device.type == "cuda" and self._cuda_cache_clear_interval > 0:
            if (self._frame_count[cam_id] % self._cuda_cache_clear_interval) == 0:
                self._clear_cuda_runtime_state()

        try:
            det_out = self._run_detection(work_frame)
            if det_out is None:
                draw = self._make_status_frame(work_frame, cam_label, "no person", color=(0, 200, 255))
                self._last_vis[cam_id] = draw
                self._latest_frames_vis[cam_id] = draw.copy()
                return draw

            orig_img_rgb, result, kpts_xy, kpts_score = self._run_pose(work_frame, det_out)
            vis_rgb = vis_frame_fast(orig_img_rgb, result, self.opt, self.vis_thres.copy())
            vis_bgr = np.ascontiguousarray(vis_rgb[:, :, ::-1])
        except RuntimeError as exc:
            if not self._handle_inference_runtime_error(exc, cam_id):
                raise
            self.latest_kpts[cam_id] = None
            self.latest_kpt_scores[cam_id] = None
            draw = self._make_status_frame(work_frame, cam_label, "inference recovered; retrying...", color=(0, 180, 255))
            self._last_vis[cam_id] = draw
            self._latest_frames_vis[cam_id] = draw.copy()
            return draw

        # Triangulation uses camera projection matrices built for original frame size.
        # Convert keypoints back to original-resolution pixel coordinates.
        if kpts_xy is not None:
            kpts_xy_full = np.asarray(kpts_xy, dtype=np.float32).copy()
            if resize_scale != 1.0:
                kpts_xy_full[:, 0] /= resize_scale
                kpts_xy_full[:, 1] /= resize_scale
            self.latest_kpts[cam_id] = kpts_xy_full
        else:
            self.latest_kpts[cam_id] = None
        self.latest_kpt_scores[cam_id] = kpts_score

        cv2.putText(
            vis_bgr,
            f"{cam_label} | persons: {len(result['result'])}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        self._last_vis[cam_id] = vis_bgr
        self._latest_frames_vis[cam_id] = vis_bgr.copy()
        return vis_bgr

    def run(self):
        context = Context()
        device_list = context.query_devices()

        if device_list.get_count() < 2:
            raise RuntimeError(f"Need 2 Orbbec devices, found {device_list.get_count()}")

        devices = [device_list.get_device_by_index(i) for i in range(2)]
        buffers = {
            0: deque(maxlen=self.sync_buffer_size),
            1: deque(maxlen=self.sync_buffer_size),
        }
        lock = threading.Lock()
        threads = [
            OrbbecTimedCameraThread(devices[i], i, buffers, lock, profile_candidates=self.capture_profiles)
            for i in range(2)
        ]

        for t in threads:
            t.start()

        panel_w, panel_h = self.preview_panel_size
        if self.show_preview_window:
            cv2.namedWindow("Orbbec Stereo AlphaPose", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Orbbec Stereo AlphaPose", panel_w * 2, panel_h)
            print(f"[System] Stereo AlphaPose running with strict sync (tol={self.sync_tol_ms:.1f}ms). Press Q / ESC to exit.")
        else:
            print(f"[System] Stereo AlphaPose running without popup preview (tol={self.sync_tol_ms:.1f}ms).")

        try:
            while True:
                if callable(self.on_should_stop):
                    try:
                        if bool(self.on_should_stop()):
                            break
                    except Exception:
                        pass
                with lock:
                    left_rec, right_rec, dt_ms = self._pop_synchronized_pair(buffers)
                pair_ts_ms = time.monotonic() * 1000.0

                if left_rec is None or right_rec is None:
                    vis_by_label = {
                        "LEFT": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
                        "RIGHT": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
                    }
                    for cam_id in ("CAM0", "CAM1"):
                        disp = self.display_label_map[cam_id]
                        cv2.putText(vis_by_label[disp], f"{disp}: waiting synced pair...", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                        if self._last_vis[cam_id] is not None:
                            vis_by_label[disp] = self._fit_to_canvas_16_9(self._last_vis[cam_id], panel_w, panel_h)
                    dt_text = "dt=N/A"
                else:
                    left = left_rec["image"]
                    right = right_rec["image"]
                    pair_ts_ms = 0.5 * (float(left_rec["ts_ms"]) + float(right_rec["ts_ms"]))
                    self._latest_frames_raw["CAM0"] = left.copy()
                    self._latest_frames_raw["CAM1"] = right.copy()
                    self._last_synced_pair["CAM0"] = left.copy()
                    self._last_synced_pair["CAM1"] = right.copy()
                    try:
                        vis_cam0 = self.infer_and_draw(left, "CAM0")
                    except Exception as exc:
                        print(f"[WARN][CAM0] infer_and_draw failed: {exc}")
                        self.latest_kpts["CAM0"] = None
                        self.latest_kpt_scores["CAM0"] = None
                        vis_cam0 = self._make_status_frame(left, self.display_label_map["CAM0"], "infer error; frame skipped")
                    try:
                        vis_cam1 = self.infer_and_draw(right, "CAM1")
                    except Exception as exc:
                        print(f"[WARN][CAM1] infer_and_draw failed: {exc}")
                        self.latest_kpts["CAM1"] = None
                        self.latest_kpt_scores["CAM1"] = None
                        vis_cam1 = self._make_status_frame(right, self.display_label_map["CAM1"], "infer error; frame skipped")
                    vis_by_label = {
                        self.display_label_map["CAM0"]: self._fit_to_canvas_16_9(vis_cam0, panel_w, panel_h),
                        self.display_label_map["CAM1"]: self._fit_to_canvas_16_9(vis_cam1, panel_w, panel_h),
                    }
                    dt_text = f"dt={dt_ms:+.2f}ms"
                    self._latest_dt_ms = float(dt_ms)

                # Realtime 3D reconstruction + visualization (when both side keypoints are available)
                pts3d, valid_mask = self._triangulate_keypoints(
                    self.latest_kpts["CAM0"],
                    self.latest_kpts["CAM1"],
                    self.latest_kpt_scores["CAM0"],
                    self.latest_kpt_scores["CAM1"],
                )
                self._update_3d_plot(pts3d, valid_mask=valid_mask)
                gait_ts_ms = time.monotonic() * 1000.0
                gait_enabled = True
                if callable(self.on_gait_enable_state):
                    try:
                        gait_enabled = bool(self.on_gait_enable_state())
                    except Exception:
                        gait_enabled = True
                self.gait_analytics.set_enabled(gait_enabled, reset_on_enable=False)
                if callable(self.on_gait_reset_requested):
                    try:
                        if bool(self.on_gait_reset_requested()):
                            self.gait_analytics.reset_session()
                    except Exception:
                        pass

                if pts3d is not None:
                    pts_world_raw = self._to_world_axes(pts3d)
                    self._collect_ground_points_from_frame(pts_world_raw, valid_mask)
                    pts_world = self._upright_and_ground(pts_world_raw)
                    self._latest_export_points = pts_world.copy()
                    self._latest_valid_mask = valid_mask.copy() if valid_mask is not None else None
                    self.gait_analytics.update(pts_world, valid_mask, gait_ts_ms)
                else:
                    self.gait_analytics.update(None, None, gait_ts_ms)

                combined = cv2.hconcat([vis_by_label["LEFT"], vis_by_label["RIGHT"]])
                overlay = (
                    f"{dt_text} | pairs={self.sync_pairs} "
                    f"| dropL={self.sync_drop_left} dropR={self.sync_drop_right} "
                    f"| tol={self.sync_tol_ms:.1f}ms"
                )
                cv2.putText(combined, overlay, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2)
                if callable(self.on_stereo_preview):
                    try:
                        self.on_stereo_preview(
                            vis_by_label["LEFT"].copy(),
                            vis_by_label["RIGHT"].copy(),
                        )
                    except Exception:
                        pass

                if callable(self.on_ground_board_calibrate_requested):
                    try:
                        if bool(self.on_ground_board_calibrate_requested()):
                            if left_rec is not None and right_rec is not None:
                                self._calibrate_ground_from_board_pair(left_rec["image"], right_rec["image"])
                            elif self._last_synced_pair["CAM0"] is not None and self._last_synced_pair["CAM1"] is not None:
                                self._calibrate_ground_from_board_pair(self._last_synced_pair["CAM0"], self._last_synced_pair["CAM1"])
                            else:
                                print("[GroundBoard] No cached synced pair yet. Wait 1-2s and try again.")
                    except Exception:
                        pass

                if self.show_preview_window:
                    cv2.imshow("Orbbec Stereo AlphaPose", combined)
                    key = cv2.waitKeyEx(1) & 0xFF
                    if key in (ord("g"), ord("G")):
                        self._calibrate_ground_from_recent_points()
                    if key in (ord("b"), ord("B")):
                        if left_rec is not None and right_rec is not None:
                            self._calibrate_ground_from_board_pair(left_rec["image"], right_rec["image"])
                        elif self._last_synced_pair["CAM0"] is not None and self._last_synced_pair["CAM1"] is not None:
                            self._calibrate_ground_from_board_pair(self._last_synced_pair["CAM0"], self._last_synced_pair["CAM1"])
                        else:
                            print("[GroundBoard] No cached synced pair yet. Wait 1-2s and press B again.")
                    if key in (ord("r"), ord("R")):
                        self._reset_ground_calibration()
                    if key in (ord("p"), ord("P")):
                        self._save_current_snapshot_bundle()
                    if key in (27, ord("q"), ord("Q")):
                        break

        finally:
            for t in threads:
                t.running = False
            for t in threads:
                t.join(timeout=2)
            self.gait_analytics.close()
            cv2.destroyAllWindows()
            os.chdir(_ORIGINAL_CWD)
            print("[System] Closed safely.")


if __name__ == "__main__":
    app = StereoAlphaPose()
    app.run()

