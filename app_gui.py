"""
GUI entry for binocular camera control, calibration, and embedded 3D/gait monitoring.
"""
import copy
import datetime
import logging
import math
import numbers
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from camera_gui_components import CalibrationPanel, LogDisplay, StereoPreviewPanel
from camera_threads import CalibrationThread
from gait_metric_safety import GaitMetricSafetyFilter
from pose_monitor_panel import PoseMonitorPanel
from rehab_reporter import RehabReportExporter
from trajectory_monitor_panel import TrajectoryMonitorPanel

APP_DIR = os.path.dirname(os.path.abspath(__file__))
GAIT_PROFILE_DEFAULT = "healthy_strict"

try:
    from camera_1 import camera_main
except ImportError:
    print("Warning: failed to import camera_1, camera view/record will be unavailable")
    camera_main = None

try:
    from realtime_pose import StereoAlphaPose
except ImportError:
    print("Warning: failed to import realtime_pose, realtime 3D pose will be unavailable")
    StereoAlphaPose = None


class CameraApp:
    def __init__(self):
        os.environ["GAIT_POPULATION_PROFILE"] = GAIT_PROFILE_DEFAULT
        self.root = tk.Tk()
        self.setup_logging()
        logging.info("Gait population profile forced to: %s", os.environ.get("GAIT_POPULATION_PROFILE"))

        self.calibration_thread = None
        self.calibration_results = {}

        self.camera_window_active = False
        self.pose_window_active = False
        self.gait_analysis_enabled = False
        self._gait_reset_counter = 0
        self._gait_reset_consumed = 0
        self._ground_board_calib_counter = 0
        self._ground_board_calib_consumed = 0
        self._pose_stop_requested = False
        self._gait_session_id = 1
        self._gait_session_start = None
        self._latest_live_metrics = {}
        self._latest_periodic_report = {}
        self._latest_quality_info = {}
        self._session_live_history = []
        self._session_report_history = []
        self._last_live_history_ts = None
        self._last_report_ts_ms = None
        self.metric_safety = GaitMetricSafetyFilter()
        self.report_exporter = RehabReportExporter(os.path.join(APP_DIR, "rehab_reports"))
        self._treadmill_speed_kmh = 4.0
        os.environ["GAIT_TREADMILL_TARGET_KMH"] = f"{self._treadmill_speed_kmh:.2f}"

        self.camera_control_state = {"recording_command": None, "is_recording": False}
        self._stereo_preview_update_pending = False
        self._latest_stereo_preview = (None, None)
        self._stereo_preview_paused = False
        self._pose_canvas_update_pending = False
        self._latest_pose_canvas = None

        self.setup_style()
        self.setup_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def setup_style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        elif "alt" in style.theme_names():
            style.theme_use("alt")

        primary = "#0B5FAE"
        secondary = "#2A3A4D"
        bg = "#EDF2F8"
        card_bg = "#F8FBFF"
        fg = "white"

        self.root.configure(bg=bg)
        style.configure(".", background=bg, foreground=secondary, font=("Microsoft YaHei", 10))
        style.configure("TFrame", background=bg)
        style.configure("TLabelframe", bordercolor="#CFD8E3", background=card_bg, relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=card_bg, foreground=primary, font=("Microsoft YaHei", 12, "bold"))
        style.configure("TButton", font=("Microsoft YaHei", 11), padding=7, background="#DFE7F1", borderwidth=0)
        style.map(
            "TButton",
            foreground=[("active", secondary), ("disabled", "#8492A6")],
            background=[("active", "#D1DDEB"), ("disabled", "#E5EAF0")],
        )
        style.configure("Accent.TButton", background=primary, foreground=fg, borderwidth=0)
        style.map(
            "Accent.TButton",
            background=[("active", "#084E90"), ("disabled", "#9DA8B5")],
            foreground=[("active", fg), ("disabled", "#F0F0F0")],
        )
        style.configure("TLabel", background=bg, foreground=secondary)
        style.configure("TEntry", fieldbackground="white", bordercolor="#CAD5E3")
        style.configure("StatusBar.TFrame", background="#2C3E50")
        style.configure("StatusBar.TLabel", background="#2C3E50", foreground="white", font=("Microsoft YaHei", 9))
        style.configure("StatusChip.TLabel", background="#39526D", foreground="#EAF2FA", font=("Microsoft YaHei", 9, "bold"), padding=(10, 4))
        style.configure("ControlBar.TLabelframe", background="#F6FAFF", bordercolor="#C9D9ED", relief="solid", borderwidth=1)
        style.configure(
            "ControlBar.TLabelframe.Label",
            background="#F6FAFF",
            foreground=primary,
            font=("Microsoft YaHei", 12, "bold"),
        )
        style.configure("PanelCard.TLabelframe", background="#F8FBFF", bordercolor="#C9D9ED", relief="solid", borderwidth=1)
        style.configure("PanelCard.TLabelframe.Label", background="#F8FBFF", foreground="#325D88", font=("Microsoft YaHei", 11, "bold"))
        style.configure("ControlGroup.TFrame", background="#F6FAFF")
        style.configure(
            "GroupTitle.TLabel",
            background="#F6FAFF",
            foreground="#325D88",
            font=("Microsoft YaHei", 10, "bold"),
        )
        style.configure("Control.TButton", font=("Microsoft YaHei", 10), padding=(8, 7))
        style.configure("Secondary.TButton", font=("Microsoft YaHei", 10), padding=(8, 7), background="#EAF1F8")
        style.map(
            "Secondary.TButton",
            foreground=[("active", "#1E3953"), ("disabled", "#8492A6")],
            background=[("active", "#DCE8F4"), ("disabled", "#EEF3F8")],
        )
        style.configure("Danger.TButton", font=("Microsoft YaHei", 10, "bold"), padding=(10, 7), background="#D9534F", foreground="#FFFFFF")
        style.map(
            "Danger.TButton",
            foreground=[("active", "#FFFFFF"), ("disabled", "#F2D6D5")],
            background=[("active", "#C64743"), ("disabled", "#D6A3A1")],
        )
        style.configure("Report.TButton", font=("Microsoft YaHei", 10, "bold"), padding=(8, 7), background="#DDF4DF", foreground="#245C2F")
        style.map(
            "Report.TButton",
            foreground=[("active", "#1E4D27"), ("disabled", "#8AA090")],
            background=[("active", "#CDEFD1"), ("disabled", "#EEF3F8")],
        )
        style.configure("Title.TLabel", background=bg, foreground="#0A4F8A", font=("Microsoft YaHei", 22, "bold"))
        style.configure("Subtitle.TLabel", background=bg, foreground="#617A95", font=("Microsoft YaHei", 11))

    def setup_logging(self):
        log_dir = os.path.join(APP_DIR, "app_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"camera_app_{datetime.datetime.now().strftime('%Y%m%d')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler()],
        )
        logging.info("App started")

    def setup_ui(self):
        self.root.title("Binocular Camera Control & Monitoring")
        self.root.geometry("2560x1600")
        self.root.minsize(1600, 900)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        title_wrap = ttk.Frame(top_frame)
        title_wrap.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            title_wrap,
            text="Binocular Camera Sync Control + Realtime Gait Monitor",
            style="Title.TLabel",
        ).pack(side=tk.TOP, anchor="w")
        ttk.Label(
            title_wrap,
            text="Clinical Motion Capture Workbench",
            style="Subtitle.TLabel",
        ).pack(side=tk.TOP, anchor="w", pady=(2, 0))
        self.quit_button = ttk.Button(top_frame, text="Exit", command=self.close_app, style="Danger.TButton")
        self.quit_button.pack(side=tk.RIGHT, anchor="n", pady=(4, 0))
        ttk.Separator(main_frame, orient="horizontal").pack(fill=tk.X, pady=(0, 10))

        control_frame = ttk.LabelFrame(main_frame, text="Quick Controls", padding=10, style="ControlBar.TLabelframe")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)

        camera_group = ttk.Frame(control_frame, style="ControlGroup.TFrame")
        camera_group.grid(row=0, column=0, sticky="nsew", padx=(2, 8), pady=2)
        camera_group.columnconfigure(0, weight=1)
        camera_group.columnconfigure(1, weight=1)
        ttk.Label(camera_group, text="Camera & Recording", style="GroupTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        self.camera_button = ttk.Button(
            camera_group, text="Open Binocular Camera", command=self.open_camera_window, style="Accent.TButton"
        )
        self.camera_button.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        self.start_record_button = ttk.Button(
            camera_group, text="Start Recording", command=self.start_recording, state="disabled", style="Control.TButton"
        )
        self.start_record_button.grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        self.stop_record_button = ttk.Button(
            camera_group, text="Stop Recording", command=self.stop_recording, state="disabled", style="Control.TButton"
        )
        self.stop_record_button.grid(row=2, column=1, sticky="ew", padx=2, pady=2)

        pose_group = ttk.Frame(control_frame, style="ControlGroup.TFrame")
        pose_group.grid(row=0, column=1, sticky="nsew", padx=8, pady=2)
        pose_group.columnconfigure(0, weight=1)
        pose_group.columnconfigure(1, weight=1)
        ttk.Label(pose_group, text="Pose & Analysis", style="GroupTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        self.pose_button = ttk.Button(
            pose_group, text="Open Realtime 3D Pose", command=self.open_pose_window, style="Accent.TButton"
        )
        self.pose_button.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        self.gait_button = ttk.Button(
            pose_group,
            text="Start Gait Analysis",
            command=self.toggle_gait_analysis,
            state="disabled",
            style="Control.TButton",
        )
        self.gait_button.grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        self.reset_gait_button = ttk.Button(
            pose_group,
            text="Clear Session Timer",
            command=self.reset_gait_session,
            state="disabled",
            style="Control.TButton",
        )
        self.reset_gait_button.grid(row=2, column=1, sticky="ew", padx=2, pady=2)
        self.calib_ground_board_button = ttk.Button(
            pose_group,
            text="Calibrate Ground (Board)",
            command=self.request_ground_board_calibration,
            state="disabled",
            style="Control.TButton",
        )
        self.calib_ground_board_button.grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        report_group = ttk.Frame(control_frame, style="ControlGroup.TFrame")
        report_group.grid(row=0, column=2, sticky="nsew", padx=(8, 2), pady=2)
        report_group.columnconfigure(0, weight=1)
        ttk.Label(report_group, text="Report & System", style="GroupTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self.print_report_button = ttk.Button(
            report_group,
            text="Export Rehab Report",
            command=self.export_or_print_report,
            state="disabled",
            style="Report.TButton",
        )
        self.print_report_button.grid(row=1, column=0, sticky="ew", padx=2, pady=2)

        center_frame = ttk.Frame(main_frame)
        center_frame.pack(fill=tk.BOTH, expand=True)
        center_frame.columnconfigure(0, weight=0, minsize=430)
        center_frame.columnconfigure(1, weight=1)
        center_frame.rowconfigure(0, weight=1)

        left_col = ttk.Frame(center_frame)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_col.rowconfigure(1, weight=1)

        self.calibration_panel = CalibrationPanel(left_col, self)
        self.calibration_panel.pack(fill=tk.X, pady=(0, 8))

        self.log_display = LogDisplay(left_col)
        self.log_display.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        self.log_display.log_text.configure(height=5)
        self.stereo_preview_panel = StereoPreviewPanel(left_col)
        self.stereo_preview_panel.pack(fill=tk.BOTH, expand=True)

        monitor_notebook = ttk.Notebook(center_frame)
        monitor_notebook.grid(row=0, column=1, sticky="nsew")
        self.monitor_notebook = monitor_notebook

        pose_tab = ttk.Frame(monitor_notebook)
        pose_tab.columnconfigure(0, weight=1)
        pose_tab.rowconfigure(0, weight=1)
        self.pose_monitor = PoseMonitorPanel(pose_tab)
        self.pose_monitor.grid(row=0, column=0, sticky="nsew")
        self.pose_tab = pose_tab

        trajectory_tab = ttk.Frame(monitor_notebook)
        trajectory_tab.columnconfigure(0, weight=1)
        trajectory_tab.rowconfigure(0, weight=1)
        self.trajectory_monitor = TrajectoryMonitorPanel(trajectory_tab)
        self.trajectory_monitor.grid(row=0, column=0, sticky="nsew")
        self.trajectory_tab = trajectory_tab

        monitor_notebook.add(pose_tab, text="Pose Monitor")
        monitor_notebook.add(trajectory_tab, text="Trajectory Stream")
        monitor_notebook.bind("<<NotebookTabChanged>>", self._on_monitor_tab_changed)
        self.root.after(0, self._on_monitor_tab_changed)

        self.setup_status_bar()

    def setup_status_bar(self):
        status_frame = ttk.Frame(self.root, style="StatusBar.TFrame")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_container = ttk.Frame(status_frame, style="StatusBar.TFrame")
        status_container.pack(fill=tk.X, padx=6, pady=4)
        row1 = ttk.Frame(status_container, style="StatusBar.TFrame")
        row2 = ttk.Frame(status_container, style="StatusBar.TFrame")
        row1.pack(fill=tk.X, pady=(0, 2))
        row2.pack(fill=tk.X)

        self.camera_status_label = ttk.Label(row1, text="Camera: idle", style="StatusChip.TLabel")
        self.camera_status_label.pack(side=tk.LEFT, padx=(0, 8))
        self.record_status_label = ttk.Label(row1, text="Recording: idle", style="StatusChip.TLabel")
        self.record_status_label.pack(side=tk.LEFT, padx=(0, 8))
        self.calibration_status_label = ttk.Label(row2, text="Calibration: idle", style="StatusChip.TLabel")
        self.calibration_status_label.pack(side=tk.LEFT, padx=(0, 8))
        self.pose_status_label = ttk.Label(row2, text="Pose: idle", style="StatusChip.TLabel")
        self.pose_status_label.pack(side=tk.LEFT, padx=(0, 8))

    def _on_pose_3d_canvas(self, canvas_bgr):
        if not self.root.winfo_exists():
            return
        self._latest_pose_canvas = canvas_bgr
        if not self._is_pose_tab_active():
            return
        if self._pose_canvas_update_pending:
            return
        self._pose_canvas_update_pending = True

        def _update():
            self._pose_canvas_update_pending = False
            if not self._is_pose_tab_active():
                return
            latest_canvas = self._latest_pose_canvas
            if latest_canvas is None:
                return
            self.pose_monitor.update_canvas(latest_canvas)

        self.root.after(0, _update)

    def _on_pose_stereo_preview(self, left_bgr, right_bgr):
        if not self.root.winfo_exists():
            return
        self._latest_stereo_preview = (left_bgr, right_bgr)
        if self._stereo_preview_paused:
            return
        if self._stereo_preview_update_pending:
            return
        self._stereo_preview_update_pending = True

        def _update():
            self._stereo_preview_update_pending = False
            if hasattr(self, "stereo_preview_panel") and self.stereo_preview_panel is not None:
                latest_left, latest_right = self._latest_stereo_preview
                self.stereo_preview_panel.update_frames(latest_left, latest_right)

        self.root.after(0, _update)

    def _on_monitor_tab_changed(self, _event=None):
        if not hasattr(self, "monitor_notebook") or not hasattr(self, "trajectory_monitor"):
            return
        try:
            selected = self.monitor_notebook.nametowidget(self.monitor_notebook.select())
        except Exception:
            return
        if selected == getattr(self, "trajectory_tab", None):
            self.trajectory_monitor.ensure_stream_running()
            self._set_stereo_preview_paused(True)
        else:
            self.trajectory_monitor.ensure_stream_stopped()
            self._set_stereo_preview_paused(False)
            latest_left, latest_right = self._latest_stereo_preview
            if latest_left is not None or latest_right is not None:
                self.stereo_preview_panel.update_frames(latest_left, latest_right)
            if self._latest_pose_canvas is not None:
                self.pose_monitor.update_canvas(self._latest_pose_canvas)

    def _is_pose_tab_active(self):
        if not hasattr(self, "monitor_notebook") or not hasattr(self, "pose_tab"):
            return False
        try:
            selected = self.monitor_notebook.nametowidget(self.monitor_notebook.select())
        except Exception:
            return False
        return selected == self.pose_tab

    def _set_stereo_preview_paused(self, paused):
        self._stereo_preview_paused = bool(paused)
        if hasattr(self, "stereo_preview_panel") and self.stereo_preview_panel is not None:
            self.stereo_preview_panel.set_paused(self._stereo_preview_paused)

    def _on_pose_metrics(self, live_metrics, periodic_report):
        if not self.root.winfo_exists():
            return

        def _update():
            safe_live, safe_report, quality = self.metric_safety.sanitize(live_metrics or {}, periodic_report or {})
            self._latest_live_metrics = dict(safe_live or {})
            self._latest_periodic_report = dict(safe_report or {})
            self._latest_quality_info = dict(quality or {})
            self._record_session_metrics(self._latest_live_metrics, self._latest_periodic_report)
            self.pose_monitor.update_metrics(self._latest_live_metrics, self._latest_periodic_report)
            mode = "ENABLED" if self.gait_analysis_enabled else "PAUSED"
            motion_state = str(self._latest_live_metrics.get("motion_state", "")).strip().upper()
            if motion_state not in ("WALKING", "STANDING"):
                motion_state = "--"
            self.print_report_button.config(state="normal" if self._latest_periodic_report else "disabled")
            self.pose_monitor.set_status(f"Pose: running\nGait: {mode} | Activity: {motion_state}")

        self.root.after(0, _update)

    @staticmethod
    def _is_finite_number(v):
        return isinstance(v, numbers.Real) and math.isfinite(float(v))

    @staticmethod
    def _percentile(sorted_vals, pct):
        if not sorted_vals:
            return None
        n = len(sorted_vals)
        if n == 1:
            return float(sorted_vals[0])
        p = min(max(float(pct), 0.0), 100.0) / 100.0
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return float(sorted_vals[lo])
        w = idx - lo
        return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)

    @classmethod
    def _robust_mean(cls, values, low_q=10.0, high_q=90.0, min_samples=4):
        vals = [float(v) for v in values if cls._is_finite_number(v)]
        if not vals:
            return None
        if len(vals) < int(min_samples):
            return float(sum(vals) / len(vals))
        vals_sorted = sorted(vals)
        lo = cls._percentile(vals_sorted, low_q)
        hi = cls._percentile(vals_sorted, high_q)
        if lo is None or hi is None:
            return float(sum(vals_sorted) / len(vals_sorted))
        if hi < lo:
            lo, hi = hi, lo
        clipped = [min(max(v, lo), hi) for v in vals_sorted]
        return float(sum(clipped) / len(clipped))

    @staticmethod
    def _si_percent(a, b):
        if not CameraApp._is_finite_number(a) or not CameraApp._is_finite_number(b):
            return None
        denom = 0.5 * (abs(float(a)) + abs(float(b))) + 1e-9
        return float(abs(float(a) - float(b)) / denom * 100.0)

    @staticmethod
    def _ratio_score(a, b):
        if not CameraApp._is_finite_number(a) or not CameraApp._is_finite_number(b):
            return None
        a = float(a)
        b = float(b)
        if a <= 0.0 or b <= 0.0:
            return None
        return float(min(a, b) / max(a, b) * 100.0)

    @staticmethod
    def _symmetry_quality(step_sym_si):
        if not CameraApp._is_finite_number(step_sym_si):
            return None
        si = float(step_sym_si)
        return float(max(0.0, min(100.0, 100.0 - 2.5 * si)))

    @staticmethod
    def _trunk_quality(trunk_std_deg):
        if not CameraApp._is_finite_number(trunk_std_deg):
            return None
        v = float(trunk_std_deg)
        return float(max(40.0, min(100.0, 100.0 - max(0.0, v - 2.5) * 9.0)))

    def _compute_session_functional_scores(self):
        if not self._session_live_history:
            return None, None

        stability_scores = []
        rhythm_scores = []
        for item in self._session_live_history:
            if not isinstance(item, dict):
                continue
            motion_state = str(item.get("motion_state", "")).strip().upper()
            if motion_state != "WALKING":
                continue

            step_l = item.get("step_l_s")
            step_r = item.get("step_r_s")
            stance_l = item.get("stance_l_pct")
            stance_r = item.get("stance_r_pct")
            sym_si = item.get("sym_step_pct")
            if sym_si is None:
                sym_si = self._si_percent(step_l, step_r)
            trunk_std = item.get("trunk_sway_std_deg")

            stance_balance = self._ratio_score(stance_l, stance_r)
            symmetry_quality = self._symmetry_quality(sym_si)
            trunk_quality = self._trunk_quality(trunk_std)

            rhythm_from_ratio = None
            if (
                self._is_finite_number(step_l) and self._is_finite_number(step_r) and
                0.25 <= float(step_l) <= 1.20 and 0.25 <= float(step_r) <= 1.20
            ):
                rhythm_from_ratio = self._ratio_score(step_l, step_r)
            rhythm_from_si = self._symmetry_quality(sym_si)

            if rhythm_from_si is not None and rhythm_from_ratio is not None:
                rhythm = 0.65 * rhythm_from_si + 0.35 * rhythm_from_ratio
            elif rhythm_from_si is not None:
                rhythm = rhythm_from_si
            else:
                rhythm = rhythm_from_ratio
            if self._is_finite_number(rhythm):
                rhythm_scores.append(float(rhythm))

            stability_terms = []
            if self._is_finite_number(stance_balance):
                stability_terms.append((0.35, float(stance_balance)))
            if self._is_finite_number(symmetry_quality):
                stability_terms.append((0.45, float(symmetry_quality)))
            if self._is_finite_number(trunk_quality):
                stability_terms.append((0.20, float(trunk_quality)))
            if stability_terms:
                wsum = sum(w for w, _ in stability_terms)
                stability = sum(w * v for w, v in stability_terms) / max(wsum, 1e-6)
                stability_scores.append(float(stability))

        stability_agg = self._robust_mean(stability_scores) if stability_scores else None
        rhythm_agg = self._robust_mean(rhythm_scores) if rhythm_scores else None
        return stability_agg, rhythm_agg

    @staticmethod
    def _get_nested(dct, path):
        cur = dct
        for k in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k, None)
        return cur

    @staticmethod
    def _set_nested(dct, path, value):
        cur = dct
        for k in path[:-1]:
            nxt = cur.get(k, None)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[k] = nxt
            cur = nxt
        cur[path[-1]] = value

    def _clear_session_aggregation(self):
        self._session_live_history = []
        self._session_report_history = []
        self._last_live_history_ts = None
        self._last_report_ts_ms = None

    def _record_session_metrics(self, live, report):
        now_ts = datetime.datetime.now().timestamp()
        if self._last_live_history_ts is None or (now_ts - self._last_live_history_ts) >= 1.0:
            self._session_live_history.append(copy.deepcopy(live or {}))
            self._last_live_history_ts = now_ts

        if isinstance(report, dict) and report:
            report_ts = report.get("ts_ms", None)
            if self._is_finite_number(report_ts):
                report_ts = float(report_ts)
                if self._last_report_ts_ms is None or abs(report_ts - self._last_report_ts_ms) > 1e-6:
                    self._session_report_history.append(copy.deepcopy(report))
                    self._last_report_ts_ms = report_ts

    def _build_aggregated_report_payload(self):
        live_out = copy.deepcopy(self._latest_live_metrics or {})
        report_out = copy.deepcopy(self._latest_periodic_report or {})

        live_paths = [
            ("cadence_spm",),
            ("step_l_s",),
            ("step_r_s",),
            ("stride_l_s",),
            ("stride_r_s",),
            ("gait_speed_mps",),
            ("stance_l_pct",),
            ("stance_r_pct",),
            ("double_support_pct",),
            ("step_width_m",),
            ("step_length_sur_m",),
            ("step_length_speed_m",),
            ("toe_clear_l_m",),
            ("toe_clear_r_m",),
            ("com_ml_std_m",),
            ("trunk_sway_std_deg",),
            ("com_speed_mps",),
        ]
        report_paths = [
            ("rom", "hip_l"),
            ("rom", "hip_r"),
            ("rom", "knee_l"),
            ("rom", "knee_r"),
            ("rom", "ankle_l"),
            ("rom", "ankle_r"),
            ("stability", "com_ml_std_m"),
            ("stability", "trunk_sway_std_deg"),
            ("symmetry", "step_time_si_pct"),
            ("symmetry", "stride_time_si_pct"),
            ("symmetry", "knee_rom_si_pct"),
            ("symmetry", "hip_rom_si_pct"),
            ("fatigue", "cadence_drift_pct"),
            ("fatigue", "toe_clear_l_drift_pct"),
            ("fatigue", "toe_clear_r_drift_pct"),
        ]

        if self._session_live_history:
            for path in live_paths:
                vals = [self._get_nested(item, path) for item in self._session_live_history]
                agg = self._robust_mean(vals)
                if agg is not None:
                    self._set_nested(live_out, path, agg)
            if self._latest_live_metrics:
                live_out["phase_l"] = self._latest_live_metrics.get("phase_l", live_out.get("phase_l"))
                live_out["phase_r"] = self._latest_live_metrics.get("phase_r", live_out.get("phase_r"))
                live_out["last_event"] = self._latest_live_metrics.get("last_event", live_out.get("last_event"))
            live_out["sym_step_pct"] = self._si_percent(live_out.get("step_l_s"), live_out.get("step_r_s"))
            live_out["sym_stride_pct"] = self._si_percent(live_out.get("stride_l_s"), live_out.get("stride_r_s"))

        if self._session_report_history:
            for path in report_paths:
                vals = [self._get_nested(item, path) for item in self._session_report_history]
                agg = self._robust_mean(vals)
                if agg is not None:
                    self._set_nested(report_out, path, agg)
            latest_report = self._session_report_history[-1]
            for key in ("ts_ms", "window_s", "report_interval_s"):
                if key in latest_report:
                    report_out[key] = latest_report.get(key)

        # Prefer session-level Step Symmetry SI from periodic report; fallback to aggregated live estimation.
        rep_step_si = self._get_nested(report_out, ("symmetry", "step_time_si_pct"))
        rep_stride_si = self._get_nested(report_out, ("symmetry", "stride_time_si_pct"))
        if self._is_finite_number(rep_step_si):
            live_out["sym_step_pct"] = float(rep_step_si)
        elif not self._is_finite_number(live_out.get("sym_step_pct")):
            live_out["sym_step_pct"] = self._si_percent(live_out.get("step_l_s"), live_out.get("step_r_s"))
        if self._is_finite_number(rep_stride_si):
            live_out["sym_stride_pct"] = float(rep_stride_si)
        elif not self._is_finite_number(live_out.get("sym_stride_pct")):
            live_out["sym_stride_pct"] = self._si_percent(live_out.get("stride_l_s"), live_out.get("stride_r_s"))

        # Process-level functional indicators for report export (not instantaneous frame values).
        gait_stability_score, gait_rhythm_score = self._compute_session_functional_scores()
        if self._is_finite_number(gait_stability_score):
            live_out["gait_stability_score_pct"] = float(gait_stability_score)
            self._set_nested(report_out, ("functional", "gait_stability_score_pct"), float(gait_stability_score))
        if self._is_finite_number(gait_rhythm_score):
            live_out["gait_rhythm_score_pct"] = float(gait_rhythm_score)
            self._set_nested(report_out, ("functional", "gait_rhythm_score_pct"), float(gait_rhythm_score))

        return live_out, report_out

    def _is_gait_analysis_enabled(self):
        return bool(self.gait_analysis_enabled)

    def _consume_gait_reset_request(self):
        if self._gait_reset_counter > self._gait_reset_consumed:
            self._gait_reset_consumed = self._gait_reset_counter
            return True
        return False

    def _consume_ground_board_calibration_request(self):
        if self._ground_board_calib_counter > self._ground_board_calib_consumed:
            self._ground_board_calib_consumed = self._ground_board_calib_counter
            return True
        return False

    def _is_pose_stop_requested(self):
        return bool(self._pose_stop_requested)

    def _bind_shortcuts(self):
        # Global treadmill speed presets:
        # key "1" -> 2 km/h, "2" -> 4 km/h, "3" -> 6 km/h.
        self.root.bind_all("<KeyPress-1>", lambda _event: self._set_treadmill_speed_kmh(2.0))
        self.root.bind_all("<KeyPress-2>", lambda _event: self._set_treadmill_speed_kmh(4.0))
        self.root.bind_all("<KeyPress-3>", lambda _event: self._set_treadmill_speed_kmh(6.0))

    def _set_treadmill_speed_kmh(self, speed_kmh, show_log=True):
        speed_kmh = min(max(float(speed_kmh), 0.0), 20.0)
        self._treadmill_speed_kmh = speed_kmh
        os.environ["GAIT_TREADMILL_TARGET_KMH"] = f"{speed_kmh:.3f}"
        if show_log and hasattr(self, "log_display"):
            self.log_display.add_log(
                f"Treadmill preset applied: {speed_kmh:.1f} km/h ({speed_kmh / 3.6:.3f} m/s) via keyboard."
            )

    def toggle_gait_analysis(self):
        if not self.pose_window_active:
            messagebox.showinfo("Info", "Start realtime 3D pose first, then enable gait analysis.")
            return
        self.gait_analysis_enabled = not self.gait_analysis_enabled
        if self.gait_analysis_enabled:
            if self._gait_session_start is None:
                self._gait_session_start = datetime.datetime.now()
            self.gait_button.config(text="Stop Gait Analysis")
            self.log_display.add_log("Gait analysis manually started.")
            self.pose_monitor.set_status("Pose: running\nGait: ENABLED")
        else:
            self.gait_button.config(text="Start Gait Analysis")
            self.log_display.add_log("Gait analysis manually stopped.")
            self.pose_monitor.set_status("Pose: running\nGait: PAUSED")

    def reset_gait_session(self):
        if not self.pose_window_active:
            messagebox.showinfo("Info", "Start realtime 3D pose first.")
            return
        self._gait_session_id += 1
        self._gait_reset_counter += 1
        self._gait_session_start = datetime.datetime.now()
        self._latest_live_metrics = {}
        self._latest_periodic_report = {}
        self._latest_quality_info = {}
        self._clear_session_aggregation()
        self.pose_monitor.clear_metrics_only()
        self.pose_monitor.set_status("Pose: running\nGait session reset | collecting data...")
        self.print_report_button.config(state="disabled")
        self.log_display.add_log(f"Gait session reset requested (Session #{self._gait_session_id}).")

    def request_ground_board_calibration(self):
        if not self.pose_window_active:
            messagebox.showinfo("Info", "Start realtime 3D pose first.")
            return
        self._ground_board_calib_counter += 1
        self.log_display.add_log("Ground calibration requested from board.")

    def request_stop_pose_window(self):
        if not self.pose_window_active:
            return
        self._pose_stop_requested = True
        self.pose_button.config(text="Stopping pose...", state="disabled")
        self.log_display.add_log("Stopping realtime 3D pose...")

    def export_or_print_report(self):
        if not self._latest_periodic_report:
            messagebox.showinfo(
                "Info",
                "No periodic rehab report yet.\n"
                "Please keep gait analysis running long enough to accumulate data.",
            )
            return
        periodic_n = len(self._session_report_history)
        if periodic_n < 4:
            self.log_display.add_log(
                f"Warning: low periodic samples ({periodic_n}); report reliability may be limited."
            )

        now = datetime.datetime.now()
        start = self._gait_session_start or now
        duration_s = max((now - start).total_seconds(), 0.0)
        agg_live, agg_report = self._build_aggregated_report_payload()
        session_info = {
            "session_id": self._gait_session_id,
            "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": duration_s,
        }
        exported = self.report_exporter.export_report(
            session_info=session_info,
            live_metrics=agg_live,
            periodic_report=agg_report,
        )
        html_zh_path = exported["html_zh_path"]
        html_en_path = exported["html_en_path"]
        json_path = exported["json_path"]
        self.log_display.add_log(f"Rehab report exported (ZH): {html_zh_path}")
        self.log_display.add_log(f"Rehab report exported (EN): {html_en_path}")
        self.log_display.add_log(f"Rehab report exported (JSON): {json_path}")
        opened = self.report_exporter.open_file(html_zh_path)
        if not opened:
            self.log_display.add_log(f"Warning: failed to auto-open report file: {html_zh_path}")

    def open_camera_window(self):
        if camera_main is None:
            messagebox.showerror("Error", "camera_1 module not available.")
            return
        if self.pose_window_active:
            messagebox.showwarning("Warning", "Realtime 3D pose is running. Close it before opening camera window.")
            return
        if self.camera_window_active:
            messagebox.showwarning("Warning", "Camera window is already open.")
            return

        self.log_display.add_log("Opening binocular camera window...")
        self.camera_button.config(text="Camera running...", state="disabled")
        self.camera_status_label.config(text="Camera: starting...")

        def start_camera():
            try:
                self.camera_window_active = True
                self.root.after(0, lambda: self.camera_status_label.config(text="Camera: running"))
                self.root.after(0, lambda: self.start_record_button.config(state="normal"))

                camera_main(
                    control_state=self.camera_control_state,
                    exit_callback=self.camera_window_closed,
                    start_recording_callback=self._on_recording_started,
                    stop_recording_callback=self._on_recording_stopped,
                )
            except Exception as e:
                error_msg = f"Failed to open camera window: {str(e)}"
                logging.exception("Failed to open camera window")
                self.root.after(0, lambda: self.log_display.add_log(f"Error: {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                self.camera_window_closed()

        threading.Thread(target=start_camera, daemon=True).start()

    def open_pose_window(self):
        if StereoAlphaPose is None:
            messagebox.showerror("Error", "realtime_pose module not available.")
            return
        if self.camera_window_active:
            messagebox.showwarning("Warning", "Camera/recording window is running. Close it before 3D pose.")
            return
        if self.pose_window_active:
            messagebox.showwarning("Warning", "Realtime 3D pose window is already open.")
            return

        self.log_display.add_log("Opening realtime 3D pose window...")
        self.pose_window_active = True
        self.pose_button.config(text="Stop Realtime 3D Pose", state="normal", command=self.request_stop_pose_window)
        self.camera_button.config(state="disabled")
        self.camera_status_label.config(text="Camera: realtime 3D pose running")
        self.pose_status_label.config(text="Pose: starting...")
        self.pose_monitor.set_status("Starting pose engine...")
        self.gait_analysis_enabled = False
        self.gait_button.config(state="normal", text="Start Gait Analysis")
        self.reset_gait_button.config(state="normal")
        self.calib_ground_board_button.config(state="normal")
        self.print_report_button.config(state="disabled")
        self._pose_stop_requested = False
        self._gait_session_start = None
        self._gait_session_id = 1
        self._gait_reset_counter = 0
        self._gait_reset_consumed = 0
        self._ground_board_calib_counter = 0
        self._ground_board_calib_consumed = 0
        self._latest_live_metrics = {}
        self._latest_periodic_report = {}
        self._latest_quality_info = {}
        self._clear_session_aggregation()
        self._stereo_preview_update_pending = False
        self._latest_stereo_preview = (None, None)
        self._stereo_preview_paused = False
        self._pose_canvas_update_pending = False
        self._latest_pose_canvas = None
        self.stereo_preview_panel.clear()

        def start_pose():
            try:
                app = StereoAlphaPose(
                    show_3d_window=False,
                    show_preview_window=False,
                    show_gait_dashboard=False,
                    on_3d_canvas=self._on_pose_3d_canvas,
                    on_stereo_preview=self._on_pose_stereo_preview,
                    on_gait_metrics=self._on_pose_metrics,
                    on_gait_enable_state=self._is_gait_analysis_enabled,
                    on_gait_reset_requested=self._consume_gait_reset_request,
                    on_ground_board_calibrate_requested=self._consume_ground_board_calibration_request,
                    on_should_stop=self._is_pose_stop_requested,
                )
                self.root.after(0, lambda: self.pose_status_label.config(text="Pose: running"))
                app.run()
            except Exception as e:
                error_msg = f"Failed to open realtime 3D pose: {str(e)}"
                logging.exception("Failed to open realtime 3D pose")
                self.root.after(0, lambda: self.log_display.add_log(f"Error: {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            finally:
                self.pose_window_closed()

        threading.Thread(target=start_pose, daemon=True).start()

    def start_recording(self):
        if not self.camera_window_active:
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = os.path.join(APP_DIR, "recording", timestamp)
        self.camera_control_state["recording_timestamp"] = timestamp
        self.camera_control_state["recording_dir"] = record_dir
        self.camera_control_state["recording_command"] = "START"
        self.start_record_button.config(state="disabled")
        self.stop_record_button.config(state="normal")
        self.log_display.add_log(f"Send recording START, dir={record_dir}")

    def stop_recording(self):
        if not self.camera_window_active:
            return
        self.camera_control_state["recording_command"] = "STOP"
        self.start_record_button.config(state="normal")
        self.stop_record_button.config(state="disabled")
        self.log_display.add_log("Send recording STOP")

    def _on_recording_started(self):
        self.root.after(0, lambda: self.record_status_label.config(text="Recording: running", foreground="#00FF00"))
        self.camera_control_state["is_recording"] = True

    def _on_recording_stopped(self):
        self.root.after(0, lambda: self.record_status_label.config(text="Recording: idle", foreground="white"))
        self.camera_control_state["is_recording"] = False

    def camera_window_closed(self):
        def update_ui():
            self.camera_window_active = False
            self.camera_control_state["is_recording"] = False
            self.camera_button.config(text="Open Binocular Camera", state="normal")
            if not self.pose_window_active:
                self.pose_button.config(state="normal")
            self.start_record_button.config(state="disabled")
            self.stop_record_button.config(state="disabled")
            self.camera_status_label.config(text="Camera: closed")
            self.record_status_label.config(text="Recording: idle", foreground="white")
            self.log_display.add_log("Camera window closed.")

        if self.root.winfo_exists():
            self.root.after(0, update_ui)

    def pose_window_closed(self):
        def update_ui():
            self.pose_window_active = False
            self.pose_button.config(text="Open Realtime 3D Pose", state="normal", command=self.open_pose_window)
            self.pose_status_label.config(text="Pose: idle")
            self.pose_monitor.set_status("Pose stopped")
            self.gait_analysis_enabled = False
            self.gait_button.config(state="disabled", text="Start Gait Analysis")
            self.reset_gait_button.config(state="disabled")
            self.calib_ground_board_button.config(state="disabled")
            self.print_report_button.config(state="disabled")
            self._pose_stop_requested = False
            self._latest_live_metrics = {}
            self._latest_periodic_report = {}
            self._latest_quality_info = {}
            self._clear_session_aggregation()
            self._stereo_preview_update_pending = False
            self._latest_stereo_preview = (None, None)
            self.stereo_preview_panel.clear()
            if not self.camera_window_active:
                self.camera_button.config(state="normal")
                self.camera_status_label.config(text="Camera: idle")
            self.log_display.add_log("Realtime 3D pose window closed.")

        if self.root.winfo_exists():
            self.root.after(0, update_ui)

    def start_calibration(self):
        base_sync_dir = os.path.join(APP_DIR, "calib_images", "sync")
        if not os.path.exists(base_sync_dir):
            messagebox.showwarning("Warning", "Calibration image folder not found. Capture synced images first (S key).")
            return
        try:
            square_size = float(self.calibration_panel.square_size_input.get())
            board_width = int(self.calibration_panel.board_width_input.get())
            board_height = int(self.calibration_panel.board_height_input.get())
            board_size = (board_width, board_height)
        except ValueError as e:
            messagebox.showerror("Parameter Error", f"Calibration parameter format error: {str(e)}")
            return

        self.calibration_panel.calibrate_button.config(state="disabled")
        self.calibration_panel.calibration_status.config(text="Calibrating...")
        self.calibration_thread = CalibrationThread(
            base_sync_dir,
            square_size,
            board_size,
            progress_callback=self.update_calibration_progress,
            finished_callback=self.handle_calibration_completed,
            failed_callback=self.handle_calibration_failed,
        )
        self.calibration_thread.start()
        self.log_display.add_log(f"Start camera calibration, square size={square_size}mm")

    def update_calibration_progress(self, message):
        def update_ui():
            self.calibration_panel.calibration_status.config(text=message)
            self.log_display.add_log(f"Calibration: {message}")

        self.root.after(0, update_ui)

    def handle_calibration_completed(self, results):
        def update_ui():
            self.calibration_results = results
            self.calibration_status_label.config(text="Calibration: Done")
            self.calibration_panel.calibration_status.config(text="Calibration completed.")
            messagebox.showinfo("Done", "Binocular calibration finished.")
            self.calibration_panel.calibrate_button.config(state="normal")

        self.root.after(0, update_ui)

    def handle_calibration_failed(self, error_message):
        def update_ui():
            self.calibration_panel.calibrate_button.config(state="normal")
            self.calibration_panel.calibration_status.config(text=f"Calibration Failed：{error_message}")
            self.calibration_status_label.config(text="Calibration: failed")
            messagebox.showerror("Calibration Failed", f"Calibration Failed：\n{error_message}")

        self.root.after(0, update_ui)

    def delete_calibration_images(self):
        base_sync_dir = os.path.join(APP_DIR, "calib_images", "sync")
        if not os.path.exists(base_sync_dir):
            messagebox.showwarning("Warning", "Calibration image folder does not exist.")
            return
        try:
            for root, _, files in os.walk(base_sync_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
            self.log_display.add_log("Deleted all calibration images.")
            messagebox.showinfo("Success", "Deleted all calibration images.")
        except Exception as e:
            error_msg = f"Failed to delete calibration images: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.log_display.add_log(f"Error: {error_msg}")

    def close_app(self):
        if messagebox.askyesno("Confirm Exit", "Exit application?"):
            logging.info("App closed")
            self._pose_stop_requested = True
            try:
                if hasattr(self, "trajectory_monitor") and self.trajectory_monitor is not None:
                    self.trajectory_monitor.shutdown()
            except Exception:
                logging.exception("Failed to shutdown trajectory monitor cleanly")
            self.root.quit()
            self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.close_app()
        except Exception as e:
            logging.error(f"Runtime error: {str(e)}")
            messagebox.showerror("Fatal Error", f"Application error:\n{str(e)}")
            self.close_app()


def main():
    try:
        app = CameraApp()
        app.run()
    except Exception as e:
        print(f"Application startup failed: {str(e)}")


if __name__ == "__main__":
    main()


