"""
Pose monitoring panel for embedded 3D skeleton and gait metrics.
"""
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk


class PoseMonitorPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Realtime Pose Monitor")
        self._photo = None
        self._score_smooth = {}
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=7)
        self.columnconfigure(1, weight=4)
        self.columnconfigure(1, minsize=520)
        self.columnconfigure(0, minsize=760)
        self.rowconfigure(0, weight=1)

        self.canvas_label = ttk.Label(self, anchor="center")
        self.canvas_label.grid(row=0, column=0, sticky="nsew", padx=(8, 6), pady=8)

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 8), pady=8)
        right.columnconfigure(0, weight=1)
        for ridx in range(7):
            right.rowconfigure(ridx, weight=0)
        self._right_panel = right
        self.bind("<Configure>", self._on_panel_resize)

        self.state_var = tk.StringVar(value="Pose monitor idle")
        self.live_var = tk.StringVar(value="--")
        self.functional_var = tk.StringVar(value="--")
        self.report_var = tk.StringVar(value="--")
        self.refresh_var = tk.StringVar(value="康复刷新：15s ｜窗口 30s")
        ttk.Label(right, text="步态时空参数", font=("Microsoft YaHei", 16, "bold")).grid(row=0, column=0, sticky="w")
        self.live_label = ttk.Label(
            right,
            textvariable=self.live_var,
            justify="left",
            wraplength=620,
            font=("Microsoft YaHei", 14),
            anchor="nw",
        )
        self.live_label.grid(row=1, column=0, sticky="ew", pady=(2, 12))

        ttk.Label(right, text="步态功能指标", font=("Microsoft YaHei", 16, "bold")).grid(row=2, column=0, sticky="w")
        self.functional_label = ttk.Label(
            right,
            textvariable=self.functional_var,
            justify="left",
            wraplength=620,
            font=("Microsoft YaHei", 14),
            anchor="nw",
        )
        self.functional_label.grid(row=3, column=0, sticky="ew", pady=(2, 12))

        ttk.Label(right, text="关节运动参数", font=("Microsoft YaHei", 16, "bold")).grid(row=4, column=0, sticky="w")
        self.report_label = ttk.Label(
            right,
            textvariable=self.report_var,
            justify="left",
            wraplength=620,
            font=("Microsoft YaHei", 14),
            anchor="nw",
        )
        self.report_label.grid(row=5, column=0, sticky="ew", pady=(2, 6))
        self.refresh_label = ttk.Label(
            right,
            textvariable=self.refresh_var,
            justify="left",
            wraplength=620,
            foreground="#1f5d99",
            font=("Microsoft YaHei", 12, "bold"),
            anchor="nw",
        )
        self.refresh_label.grid(row=6, column=0, sticky="ew", pady=(3, 0))
        self._right_panel.bind("<Configure>", self._on_right_panel_resize)

    def _on_right_panel_resize(self, event):
        usable = max(220, int(getattr(event, "width", 560)) - 18)
        for label in (
            getattr(self, "live_label", None),
            getattr(self, "functional_label", None),
            getattr(self, "report_label", None),
            getattr(self, "refresh_label", None),
        ):
            if label is not None:
                label.configure(wraplength=usable)

    def _on_panel_resize(self, _event):
        total_w = max(int(self.winfo_width()), 0)
        if total_w <= 0:
            return
        right_w = min(700, max(520, int(total_w * 0.33)))
        left_w = max(700, total_w - right_w - 24)
        self.columnconfigure(1, minsize=right_w)
        self.columnconfigure(0, minsize=left_w)

    @staticmethod
    def _fmt(v, nd=2, suffix=""):
        try:
            if v is None:
                return "--"
            return f"{float(v):.{nd}f}{suffix}"
        except Exception:
            return "--"

    def set_status(self, text):
        self.state_var.set(str(text))

    def clear(self):
        self._photo = None
        self._score_smooth = {}
        self.canvas_label.configure(image="")
        self.state_var.set("Pose monitor idle")
        self.live_var.set("--")
        self.functional_var.set("--")
        self.report_var.set("--")
        self.refresh_var.set("康复刷新：15s ｜窗口 30s")

    def clear_metrics_only(self):
        self.live_var.set("--")
        self.functional_var.set("--")
        self.report_var.set("--")

    def update_canvas(self, canvas_bgr):
        if canvas_bgr is None:
            return
        frame = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        panel_w = max(int(self.winfo_width()), 0)
        right_reserved = max(520, int(panel_w * 0.33)) if panel_w > 0 else 600
        panel_h = max(int(self.canvas_label.winfo_height()), 0)
        if panel_w > 0:
            max_w = max(700, min(1320, panel_w - right_reserved))
        else:
            max_w = 1100
        if panel_h > 0:
            max_h = max(520, min(1240, panel_h - 8))
        else:
            max_h = 980
        scale = min(max_w / max(float(w), 1.0), max_h / max(float(h), 1.0))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(frame)
        self._photo = ImageTk.PhotoImage(image)
        self.canvas_label.configure(image=self._photo)

    def update_metrics(self, live, report):
        live = live or {}
        report = report or {}
        interval_s = live.get("report_interval_s", report.get("report_interval_s", 15.0))
        window_s = live.get("report_window_s", report.get("window_s", 30.0))
        self.refresh_var.set(f"康复刷新：{self._fmt(interval_s, 0, 's')} ｜窗口 {self._fmt(window_s, 0, 's')}")

        def _ratio_score(a, b):
            try:
                a = float(a)
                b = float(b)
                if a <= 0.0 or b <= 0.0:
                    return None
                return min(a, b) / max(a, b) * 100.0
            except Exception:
                return None

        def _symmetry_quality(si_pct):
            try:
                si = float(si_pct)
                # SI 越低越好。对健康行走场景降低瞬时惩罚敏感度：
                # 0 -> 100分，10 -> 82分，15 -> 73分。
                return max(0.0, min(100.0, 100.0 - 1.8 * si))
            except Exception:
                return None

        def _trunk_quality(trunk_std_deg):
            try:
                v = float(trunk_std_deg)
                # 轻微摆动不扣分，超过约 2.5deg 开始线性扣分
                return max(40.0, min(100.0, 100.0 - max(0.0, v - 2.5) * 9.0))
            except Exception:
                return None

        def _smooth_score(name, value, alpha=0.35, max_step=8.0):
            if value is None:
                return None
            try:
                v = float(value)
            except Exception:
                return None
            prev = self._score_smooth.get(name, None)
            if prev is None:
                self._score_smooth[name] = v
                return v
            delta = v - float(prev)
            if delta > max_step:
                v = float(prev) + max_step
            elif delta < -max_step:
                v = float(prev) - max_step
            out = (1.0 - alpha) * float(prev) + alpha * v
            self._score_smooth[name] = out
            return out

        rom = report.get("rom", {}) if isinstance(report, dict) else {}
        stab = report.get("stability", {}) if isinstance(report, dict) else {}
        sym_report = report.get("symmetry", {}) if isinstance(report, dict) else {}

        step_len = live.get("step_length_sur_m")
        step_len_speed = live.get("step_length_speed_m")
        cadence = live.get("cadence_spm")
        gait_speed = live.get("gait_speed_mps")
        if gait_speed is None:
            gait_speed = live.get("com_speed_mps")
        phase_l = live.get("phase_l", "--")
        phase_r = live.get("phase_r", "--")
        double_support = live.get("double_support_pct")
        toe_l = live.get("toe_clear_l_m")
        toe_r = live.get("toe_clear_r_m")
        stance_l = live.get("stance_l_pct")
        stance_r = live.get("stance_r_pct")
        step_l = live.get("step_l_s")
        step_r = live.get("step_r_s")
        step_sym_live = live.get("sym_step_pct")
        step_sym_report = sym_report.get("step_time_si_pct")
        trunk_std_live = live.get("trunk_sway_std_deg")

        step_sym_si = None
        if isinstance(step_sym_report, (int, float)) and isinstance(step_sym_live, (int, float)):
            # Prefer periodic 30s symmetry for stability, blend a small share of live trend.
            base = 0.65 * float(step_sym_report) + 0.35 * float(step_sym_live)
            step_sym_si = float(base)
        elif isinstance(step_sym_report, (int, float)):
            step_sym_si = float(step_sym_report)
        elif isinstance(step_sym_live, (int, float)):
            step_sym_si = float(step_sym_live)

        stance_balance = _ratio_score(stance_l, stance_r)
        symmetry_quality = _symmetry_quality(step_sym_si)
        trunk_quality = _trunk_quality(trunk_std_live)
        rhythm_from_ratio = None
        if (
            step_l is not None and step_r is not None and
            isinstance(step_l, (int, float)) and isinstance(step_r, (int, float)) and
            0.25 <= float(step_l) <= 1.20 and 0.25 <= float(step_r) <= 1.20
        ):
            rhythm_from_ratio = _ratio_score(step_l, step_r)
        rhythm_from_si = _symmetry_quality(step_sym_si)

        if rhythm_from_si is not None and rhythm_from_ratio is not None:
            gait_rhythm_score_raw = 0.65 * rhythm_from_si + 0.35 * rhythm_from_ratio
        elif rhythm_from_si is not None:
            gait_rhythm_score_raw = rhythm_from_si
        else:
            gait_rhythm_score_raw = rhythm_from_ratio

        stability_terms = []
        if stance_balance is not None:
            stability_terms.append((0.35, stance_balance))
        if symmetry_quality is not None:
            stability_terms.append((0.45, symmetry_quality))
        if trunk_quality is not None:
            stability_terms.append((0.20, trunk_quality))
        if stability_terms:
            wsum = sum(w for w, _ in stability_terms)
            gait_stability_score_raw = sum(w * v for w, v in stability_terms) / max(wsum, 1e-6)
        else:
            gait_stability_score_raw = None

        gait_stability_score = _smooth_score("stability", gait_stability_score_raw)
        gait_rhythm_score = _smooth_score("rhythm", gait_rhythm_score_raw)
        step_sym_display = _smooth_score("symmetry", step_sym_si, alpha=0.25, max_step=3.5)

        spatiotemporal_text = (
            f"步长/步频/步速：{self._fmt(step_len_speed if step_len_speed is not None else step_len, 3, ' m')} / {self._fmt(cadence, 1, ' spm')} / {self._fmt(gait_speed, 3, ' m/s')}\n"
            f"支撑相/摆动相：{phase_l} / {phase_r}\n"
            f"双支撑相时间：{self._fmt(double_support, 1, '%')}\n"
            f"足趾垂直位移：{self._fmt(toe_l, 3, ' m')} / {self._fmt(toe_r, 3, ' m')}"
        )
        functional_text = (
            f"步态稳定性：{self._fmt(gait_stability_score, 1, '%')}\n"
            f"步态对称性：{self._fmt(symmetry_quality, 1, '%')}\n"
            f"步时不对称SI：{self._fmt(step_sym_display, 1, '%')}\n"
            f"步态节律性：{self._fmt(gait_rhythm_score, 1, '%')}"
        )
        self.live_var.set(spatiotemporal_text)
        self.functional_var.set(functional_text)

        report_text = (
            f"髋关节L/R：{self._fmt(rom.get('hip_l'), 1, ' deg')} / {self._fmt(rom.get('hip_r'), 1, ' deg')}\n"
            f"膝关节L/R：{self._fmt(rom.get('knee_l'), 1, ' deg')} / {self._fmt(rom.get('knee_r'), 1, ' deg')}\n"
            f"踝关节L/R：{self._fmt(rom.get('ankle_l'), 1, ' deg')} / {self._fmt(rom.get('ankle_r'), 1, ' deg')}\n"
            f"躯干摆动标准差：{self._fmt(stab.get('trunk_sway_std_deg'), 2, ' deg')}"
        )
        self.report_var.set(report_text)

