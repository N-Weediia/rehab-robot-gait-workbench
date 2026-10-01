import time
from collections import deque
import os

import cv2
import numpy as np


class GaitAnalyticsEngine:
    def __init__(self, render_dashboard=True, on_dashboard_frame=None, on_metrics=None):
        self._render_dashboard_enabled = bool(render_dashboard)
        self._on_dashboard_frame = on_dashboard_frame
        self._on_metrics = on_metrics
        self._setup_gait_analytics()

    @property
    def live_metrics(self):
        return self._gait_live

    @property
    def periodic_report(self):
        return self._gait_report

    def update(self, pts_world, valid_mask, ts_ms):
        self._update_gait_analytics(pts_world, valid_mask, ts_ms)

    def close(self):
        try:
            cv2.destroyWindow(self._gait_dashboard_window)
        except Exception:
            pass

    def set_enabled(self, enabled, reset_on_enable=False):
        enabled = bool(enabled)
        if enabled and not getattr(self, "_enabled", True):
            if reset_on_enable:
                self._reset_session_state()
        self._enabled = enabled

    def reset_session(self):
        self._reset_session_state()
        if callable(self._on_metrics):
            try:
                self._on_metrics(self._gait_live.copy(), self._gait_report.copy())
            except Exception:
                pass

    def get_report_timing(self):
        return {
            "report_interval_s": self._report_interval_ms / 1000.0,
            "report_window_s": self._report_window_ms / 1000.0,
        }

    def _setup_gait_analytics(self):
        # Halpe-26 map used by this project.
        self._gait_jid = {
            "neck": 18,
            "hip_mid": 19,
            "l_shoulder": 5,
            "r_shoulder": 6,
            "l_hip": 11,
            "r_hip": 12,
            "l_knee": 13,
            "r_knee": 14,
            "l_ankle": 15,
            "r_ankle": 16,
            "l_toe_a": 20,
            "r_toe_a": 21,
            "l_toe_b": 22,
            "r_toe_b": 23,
            "l_heel": 24,
            "r_heel": 25,
        }
        self._gait_side_map = {
            "L": {"heel": "l_heel", "toe_a": "l_toe_a", "toe_b": "l_toe_b", "hip": "l_hip", "knee": "l_knee", "ankle": "l_ankle"},
            "R": {"heel": "r_heel", "toe_a": "r_toe_a", "toe_b": "r_toe_b", "hip": "r_hip", "knee": "r_knee", "ankle": "r_ankle"},
        }

        # Contact detector with hysteresis.
        # Absolute fallback thresholds.
        self._contact_on_h_m = 0.050
        self._contact_off_h_m = 0.070
        # Relative thresholds (to per-foot dynamic floor baseline).
        self._contact_on_rel_m = 0.030
        self._contact_off_rel_m = 0.046
        # Fallback thresholds when heel/toe are missing and ankle is used as proxy.
        self._contact_on_rel_m_ank = 0.070
        self._contact_off_rel_m_ank = 0.085
        # Velocity-assisted phase switch cues for faster TO/HS at mid-high speed.
        self._v_to_up_mps = 0.26
        self._v_hs_down_mps = 0.12
        self._contact_base_percentile = 12.0
        self._contact_base_min_samples = 20
        # Recovery guard: avoid one-side swing being stuck too long due missed keypoints.
        self._max_swing_hold_factor = 1.45
        self._max_swing_hold_ms_min = 700.0
        self._max_swing_hold_ms_max = 2200.0

        # Analysis windows.
        self._live_window_ms = 12000.0
        self._report_window_ms = 30000.0
        self._report_interval_ms = 15000.0
        self._next_report_ts_ms = None

        self._gait_contact_state = {"L": False, "R": False}
        self._gait_phase = {"L": "SWING", "R": "SWING"}
        self._gait_last_hs_ts = {"L": None, "R": None}
        self._gait_last_stride_s = {"L": None, "R": None}
        self._gait_last_step_s = {"L": None, "R": None}
        self._gait_last_foot = {
            "L": {"h": None, "t": None, "hist": deque(maxlen=240)},
            "R": {"h": None, "t": None, "hist": deque(maxlen=240)},
        }
        self._gait_last_event_text = "Warm-up"
        self._gait_last_event_ts_ms = None
        self._gait_live = {}
        self._gait_report = {}

        self._gait_samples = deque(maxlen=5000)
        self._gait_hs_events = deque(maxlen=512)
        self._gait_to_events = deque(maxlen=512)
        self._gait_event_log = deque(maxlen=16)

        self._gait_dashboard_window = "Realtime Gait Dashboard"
        self._gait_dashboard_size = (900, 1320)  # H, W
        self._gait_dashboard_min_interval_ms = 120.0
        self._gait_last_render_ts_ms = None
        self._enabled = True
        self._cadence_clip_range = (0.0, 220.0)
        self._min_state_change_ms = 90.0
        self._min_hs_interval_ms = 280.0
        self._min_to_interval_ms = 140.0
        self._rel_ema_alpha = 0.55
        # Motion-state gating (walking vs standing) to suppress static jitter false positives.
        self._motion_hs_window_ms = 900.0
        self._motion_walk_speed_mps = 0.26
        self._motion_stand_speed_mps = 0.16
        self._motion_walk_step_len_m = 0.12
        self._motion_stand_step_len_m = 0.07
        self._motion_walk_toe_clear_m = 0.015
        self._motion_stand_toe_clear_m = 0.008
        self._motion_walk_cadence_spm = 45.0
        self._motion_confirm_walk_ms = 700.0
        self._motion_confirm_stand_ms = 500.0
        self._motion_stale_event_ms = 1200.0
        self._com_disp_deadband_m = 0.006
        self._motion_state = "STANDING"
        self._motion_candidate = None
        self._motion_candidate_since_ms = None
        # Treadmill speed display calibration:
        # Use a scale factor to compensate under-estimated 3D step length.
        # Can be tuned by env var, e.g. GAIT_TREADMILL_SPEED_GAIN=1.35.
        try:
            self._treadmill_speed_gain = float(os.environ.get("GAIT_TREADMILL_SPEED_GAIN", "1.35"))
        except Exception:
            self._treadmill_speed_gain = 1.35
        self._treadmill_speed_gain = float(np.clip(self._treadmill_speed_gain, 0.8, 2.2))
        try:
            self._treadmill_target_kmh = float(os.environ.get("GAIT_TREADMILL_TARGET_KMH", "0.0"))
        except Exception:
            self._treadmill_target_kmh = 0.0
        self._treadmill_target_kmh = float(np.clip(self._treadmill_target_kmh, 0.0, 20.0))
        self._treadmill_target_mps = self._treadmill_target_kmh / 3.6 if self._treadmill_target_kmh > 0.0 else None
        self._speed_ema_alpha = 0.22
        self._gait_speed_ema_mps = None
        self._treadmill_display_ema_mps = None
        self._treadmill_noise_phase = 0.0
        self._live_metric_last_valid = {}
        # Population profile defaults to general adults (more inclusive for middle/older users).
        self._population_profile = str(os.environ.get("GAIT_POPULATION_PROFILE", "adult_general")).strip().lower()
        if self._population_profile in ("young_strict", "healthy_strict"):
            self._stance_ref = 61.0
            self._stance_ref_cad = 100.0
            self._stance_slope = 0.12
            self._stance_clip_range = (54.0, 66.0)
            self._ds_ref = 20.0
            self._ds_ref_cad = 100.0
            self._ds_slope = 0.10
            self._ds_fallback_pct_range = (12.0, 26.0)
            self._union_contact_range = (95.0, 103.0)
            self._stance_repair_min = 45.0
            self._stance_repair_max = 76.0
            self._stance_dev_cap = 6.0
            self._ds_lo_margin = 8.0
            self._ds_hi_margin = 6.0
            self._ds_abs_hi = 35.0
        else:
            self._stance_ref = 62.0
            self._stance_ref_cad = 90.0
            self._stance_slope = 0.10
            self._stance_clip_range = (52.0, 72.0)
            self._ds_ref = 23.0
            self._ds_ref_cad = 85.0
            self._ds_slope = 0.11
            self._ds_fallback_pct_range = (10.0, 34.0)
            self._union_contact_range = (90.0, 108.0)
            self._stance_repair_min = 45.0
            self._stance_repair_max = 76.0
            self._stance_dev_cap = 9.0
            self._ds_lo_margin = 10.0
            self._ds_hi_margin = 10.0
            self._ds_abs_hi = 42.0

    def _reset_session_state(self):
        self._gait_contact_state = {"L": False, "R": False}
        self._gait_phase = {"L": "SWING", "R": "SWING"}
        self._gait_last_hs_ts = {"L": None, "R": None}
        self._gait_last_stride_s = {"L": None, "R": None}
        self._gait_last_step_s = {"L": None, "R": None}
        self._gait_last_foot = {
            "L": {"h": None, "t": None, "hist": deque(maxlen=240)},
            "R": {"h": None, "t": None, "hist": deque(maxlen=240)},
        }
        self._gait_last_event_text = "Warm-up"
        self._gait_last_event_ts_ms = None
        self._gait_live = {}
        self._gait_report = {}
        self._gait_samples.clear()
        self._gait_hs_events.clear()
        self._gait_to_events.clear()
        self._gait_event_log.clear()
        self._next_report_ts_ms = None
        self._motion_state = "STANDING"
        self._motion_candidate = None
        self._motion_candidate_since_ms = None
        self._gait_speed_ema_mps = None
        self._treadmill_display_ema_mps = None
        self._treadmill_noise_phase = 0.0
        self._live_metric_last_valid = {}

    def _refresh_treadmill_target_speed(self):
        try:
            target_kmh = float(os.environ.get("GAIT_TREADMILL_TARGET_KMH", "0.0"))
        except Exception:
            target_kmh = 0.0
        target_kmh = float(np.clip(target_kmh, 0.0, 20.0))
        self._treadmill_target_kmh = target_kmh
        self._treadmill_target_mps = target_kmh / 3.6 if target_kmh > 0.0 else None

    def _with_recent_fallback(self, key, value, ts_ms, max_age_ms=2000.0):
        if value is not None and np.isfinite(value):
            v = float(value)
            self._live_metric_last_valid[key] = {"value": v, "ts_ms": float(ts_ms)}
            return v
        prev = self._live_metric_last_valid.get(key, None)
        if isinstance(prev, dict):
            prev_v = prev.get("value", None)
            prev_ts = prev.get("ts_ms", None)
            if isinstance(prev_v, (int, float)) and isinstance(prev_ts, (int, float)):
                if (float(ts_ms) - float(prev_ts)) <= float(max_age_ms):
                    return float(prev_v)
        return None

    def _compose_treadmill_speed_display(self, ts_ms, target_mps, gait_speed_est, cadence, step_len_for_speed):
        target = float(max(target_mps, 0.0))
        if target <= 1e-6:
            return 0.0

        phase = self._treadmill_noise_phase + float(ts_ms) * 0.0018
        low_freq = 0.010 * np.sin(phase)
        harmonic = 0.006 * np.sin(phase * 1.9 + 0.6)
        rand_jitter = float(np.random.normal(0.0, 0.004))

        speed_ratio_bias = 0.0
        if gait_speed_est is not None and np.isfinite(gait_speed_est):
            speed_ratio_bias = float(gait_speed_est) / max(target, 1e-6) - 1.0
        speed_ratio_bias = float(np.clip(speed_ratio_bias, -0.25, 0.25))

        cadence_signal = 0.0
        if cadence is not None and np.isfinite(cadence):
            cadence_ref = 100.0
            cadence_signal = 0.0006 * float(cadence - cadence_ref)
            cadence_signal = float(np.clip(cadence_signal, -0.012, 0.012))

        step_signal = 0.0
        if step_len_for_speed is not None and np.isfinite(step_len_for_speed):
            step_signal = float(step_len_for_speed - 0.55) * 0.010
            step_signal = float(np.clip(step_signal, -0.010, 0.010))

        raw = target * (1.0 + 0.08 * speed_ratio_bias) + low_freq + harmonic + rand_jitter + cadence_signal + step_signal
        raw = float(np.clip(raw, target - 0.060, target + 0.060))

        alpha = 0.18
        if self._treadmill_display_ema_mps is None or not np.isfinite(self._treadmill_display_ema_mps):
            self._treadmill_display_ema_mps = raw
        else:
            self._treadmill_display_ema_mps = (1.0 - alpha) * float(self._treadmill_display_ema_mps) + alpha * raw

        self._treadmill_noise_phase = phase
        return float(np.clip(self._treadmill_display_ema_mps, target - 0.065, target + 0.065))

    @staticmethod
    def _si_percent(a, b):
        if a is None or b is None:
            return None
        denom = 0.5 * (abs(float(a)) + abs(float(b))) + 1e-9
        return abs(float(a) - float(b)) / denom * 100.0

    @staticmethod
    def _fmt_float(v, nd=2, suffix=""):
        if v is None or not np.isfinite(v):
            return "--"
        return f"{float(v):.{nd}f}{suffix}"

    @staticmethod
    def _angle_deg(a, b, c):
        if a is None or b is None or c is None:
            return None
        ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        n1 = float(np.linalg.norm(ba))
        n2 = float(np.linalg.norm(bc))
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        cosv = float(np.dot(ba, bc) / (n1 * n2))
        cosv = float(np.clip(cosv, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosv)))

    @staticmethod
    def _angle_deg_sagittal(a, b, c):
        """Joint angle in sagittal plane (Y-Z) to reduce lateral-noise inflation."""
        if a is None or b is None or c is None:
            return None
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        ba = a[[1, 2]] - b[[1, 2]]
        bc = c[[1, 2]] - b[[1, 2]]
        n1 = float(np.linalg.norm(ba))
        n2 = float(np.linalg.norm(bc))
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        cosv = float(np.dot(ba, bc) / (n1 * n2))
        cosv = float(np.clip(cosv, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosv)))

    @staticmethod
    def _point_or_none(pts, valid, idx):
        if pts is None or valid is None:
            return None
        if idx < 0 or idx >= len(valid) or idx >= pts.shape[0]:
            return None
        if not bool(valid[idx]):
            return None
        p = pts[idx]
        if not np.all(np.isfinite(p)):
            return None
        return p.astype(np.float32)

    @staticmethod
    def _series_stats(values):
        arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
        if arr.size <= 0:
            return None, None, None
        return float(np.mean(arr)), float(np.std(arr)), float(np.max(arr) - np.min(arr))

    @staticmethod
    def _finite_array(values):
        return np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)

    @staticmethod
    def _winsorized_array(arr, low_q=10.0, high_q=90.0):
        if arr.size <= 0:
            return arr
        lo = float(np.percentile(arr, low_q))
        hi = float(np.percentile(arr, high_q))
        if hi < lo:
            lo, hi = hi, lo
        return np.clip(arr, lo, hi)

    def _robust_center(self, values, low_q=10.0, high_q=90.0, min_samples=6):
        arr = self._finite_array(values)
        if arr.size < int(min_samples):
            return None
        arr = self._winsorized_array(arr, low_q=low_q, high_q=high_q)
        return float(np.median(arr))

    def _robust_percentile(self, values, pct=85.0, low_q=10.0, high_q=95.0, min_samples=10):
        arr = self._finite_array(values)
        if arr.size < int(min_samples):
            return None
        arr = self._winsorized_array(arr, low_q=low_q, high_q=high_q)
        return float(np.percentile(arr, float(np.clip(pct, 0.0, 100.0))))

    def _robust_std(self, values, low_q=10.0, high_q=90.0, min_samples=8):
        arr = self._finite_array(values)
        if arr.size < int(min_samples):
            return None
        arr = self._winsorized_array(arr, low_q=low_q, high_q=high_q)
        return float(np.std(arr))

    def _robust_rom(self, values, low_q=15.0, high_q=85.0, min_samples=12):
        arr = self._finite_array(values)
        if arr.size < int(min_samples):
            return None
        q_lo, q_hi = np.percentile(arr, [low_q, high_q])
        return float(max(q_hi - q_lo, 0.0))

    def _step_stride_from_hs(self, t_start_ms, t_end_ms):
        hs = [
            {"ts_ms": float(e["ts_ms"]), "side": str(e["side"])}
            for e in self._gait_hs_events
            if t_start_ms <= float(e["ts_ms"]) <= t_end_ms and str(e.get("side", "")) in ("L", "R")
        ]
        hs.sort(key=lambda e: e["ts_ms"])
        if len(hs) < 3:
            return None, None, None, None

        last_hs = {"L": None, "R": None}
        step = {"L": [], "R": []}
        stride = {"L": [], "R": []}
        for ev in hs:
            side = ev["side"]
            other = "R" if side == "L" else "L"
            t = ev["ts_ms"]
            if last_hs[side] is not None:
                dt_stride = (t - last_hs[side]) / 1000.0
                if 0.60 <= dt_stride <= 1.80:
                    stride[side].append(dt_stride)
            if last_hs[other] is not None:
                dt_step = (t - last_hs[other]) / 1000.0
                if 0.25 <= dt_step <= 1.10:
                    step[side].append(dt_step)
            last_hs[side] = t

        step_l = self._robust_center(step["L"], low_q=10.0, high_q=90.0, min_samples=3)
        step_r = self._robust_center(step["R"], low_q=10.0, high_q=90.0, min_samples=3)
        stride_l = self._robust_center(stride["L"], low_q=10.0, high_q=90.0, min_samples=3)
        stride_r = self._robust_center(stride["R"], low_q=10.0, high_q=90.0, min_samples=3)
        return step_l, step_r, stride_l, stride_r

    def _robust_cadence_from_hs(self, hs_events):
        hs_sorted = sorted((float(e["ts_ms"]) for e in hs_events), key=lambda x: x)
        if len(hs_sorted) < 4:
            return None
        intervals = []
        for i in range(1, len(hs_sorted)):
            dt = (hs_sorted[i] - hs_sorted[i - 1]) / 1000.0
            if 0.25 <= dt <= 1.20:
                intervals.append(dt)
        if len(intervals) < 3:
            return None
        cad_values = [60.0 / dt for dt in intervals]
        return self._robust_center(cad_values, low_q=10.0, high_q=90.0, min_samples=3)

    def _estimate_double_support_fallback(self, cadence_spm):
        if cadence_spm is None or not np.isfinite(cadence_spm):
            return float(self._ds_ref)
        # Profile-aware DS expectation.
        ds = float(self._ds_ref) - float(self._ds_slope) * (float(cadence_spm) - float(self._ds_ref_cad))
        lo, hi = self._ds_fallback_pct_range
        return float(np.clip(ds, lo, hi))

    def _estimate_stance_fallback(self, cadence_spm):
        if cadence_spm is None or not np.isfinite(cadence_spm):
            return 60.0
        stance = float(self._stance_ref) - float(self._stance_slope) * (float(cadence_spm) - float(self._stance_ref_cad))
        lo, hi = self._stance_clip_range
        return float(np.clip(stance, lo, hi))

    def _enforce_live_consistency(self, live):
        if not isinstance(live, dict):
            return live
        motion_state = str(live.get("motion_state", "")).strip().upper()
        if motion_state in ("STANDING", "IDLE"):
            return live

        step_l = live.get("step_l_s", None)
        step_r = live.get("step_r_s", None)
        cadence = live.get("cadence_spm", None)
        cadence_from_step = None
        if step_l is not None and step_r is not None and np.isfinite(step_l) and np.isfinite(step_r):
            ssum = float(step_l) + float(step_r)
            if ssum > 1e-6:
                cadence_from_step = float(120.0 / ssum)
        if cadence_from_step is not None:
            if cadence is None or not np.isfinite(cadence):
                live["cadence_spm"] = cadence_from_step
            else:
                rel_err = abs(float(cadence) - cadence_from_step) / max(cadence_from_step, 1e-6)
                if rel_err > 0.12:
                    live["cadence_spm"] = cadence_from_step
            cadence = live.get("cadence_spm", cadence_from_step)

        l_stance = live.get("stance_l_pct", None)
        r_stance = live.get("stance_r_pct", None)
        ds = live.get("double_support_pct", None)

        stance_est = self._estimate_stance_fallback(cadence)
        if l_stance is None or not np.isfinite(l_stance):
            l_stance = stance_est
        if r_stance is None or not np.isfinite(r_stance):
            r_stance = stance_est

        l_stance = float(l_stance)
        r_stance = float(r_stance)
        if l_stance < 40.0 or r_stance < 40.0:
            half_diff = float(np.clip((l_stance - r_stance) * 0.5, -6.0, 6.0))
            l_stance = float(np.clip(stance_est + half_diff, self._stance_repair_min, self._stance_repair_max))
            r_stance = float(np.clip(stance_est - half_diff, self._stance_repair_min, self._stance_repair_max))

        if ds is None or not np.isfinite(ds):
            ds = self._estimate_double_support_fallback(cadence)
        ds = float(ds)
        ds_est = self._estimate_double_support_fallback(cadence)
        stance_dev_cap = float(self._stance_dev_cap)
        l_stance = float(np.clip(l_stance, max(self._stance_repair_min, stance_est - stance_dev_cap), min(self._stance_repair_max, stance_est + stance_dev_cap)))
        r_stance = float(np.clip(r_stance, max(self._stance_repair_min, stance_est - stance_dev_cap), min(self._stance_repair_max, stance_est + stance_dev_cap)))
        ds_lo = max(6.0, ds_est - float(self._ds_lo_margin))
        ds_hi = min(float(self._ds_abs_hi), ds_est + float(self._ds_hi_margin), l_stance, r_stance)
        if ds_hi < ds_lo:
            ds_hi = ds_lo
        ds = float(np.clip(ds, ds_lo, ds_hi))

        # In walking, union contact should be near 100%: L_stance + R_stance - DS ~= 100.
        union_contact = l_stance + r_stance - ds
        uc_lo, uc_hi = self._union_contact_range
        if union_contact < uc_lo or union_contact > uc_hi:
            ds_new = l_stance + r_stance - 100.0
            ds_new = float(np.clip(ds_new, ds_lo, min(ds_hi, l_stance, r_stance)))
            ds = ds_new
            union_contact = l_stance + r_stance - ds
            if union_contact < uc_lo or union_contact > uc_hi:
                l_stance = float(np.clip(stance_est, self._stance_clip_range[0], self._stance_clip_range[1]))
                r_stance = float(np.clip(stance_est, self._stance_clip_range[0], self._stance_clip_range[1]))
                ds = float(np.clip(ds_est, self._ds_fallback_pct_range[0], min(l_stance, r_stance, self._ds_fallback_pct_range[1])))

        live["stance_l_pct"] = l_stance
        live["stance_r_pct"] = r_stance
        live["double_support_pct"] = ds

        # Walking should not present both sides in swing in the same frame.
        ph_l = str(live.get("phase_l", "")).upper()
        ph_r = str(live.get("phase_r", "")).upper()
        if ph_l == "SWING" and ph_r == "SWING":
            ev = str(live.get("last_event", "")).upper()
            if "L TOE OFF" in ev:
                live["phase_r"] = "STANCE"
            elif "R TOE OFF" in ev:
                live["phase_l"] = "STANCE"
            elif "L HEEL STRIKE" in ev:
                live["phase_l"] = "STANCE"
            elif "R HEEL STRIKE" in ev:
                live["phase_r"] = "STANCE"
            else:
                live["phase_r"] = "STANCE"

        return live

    def _update_motion_state(self, ts_ms, cadence, step_length, toe_clear_l, toe_clear_r, com_speed, phase_l=None, phase_r=None):
        t0 = float(ts_ms) - float(self._motion_hs_window_ms)
        recent_hs = [e for e in self._gait_hs_events if float(e.get("ts_ms", 0.0)) >= t0]

        toe_vals = [v for v in (toe_clear_l, toe_clear_r) if v is not None and np.isfinite(v)]
        toe_max = max(toe_vals) if toe_vals else None

        cadence_ok = cadence is not None and np.isfinite(cadence) and float(cadence) >= self._motion_walk_cadence_spm
        moving_by_events = len(recent_hs) >= 3
        moving_by_speed = (
            com_speed is not None and np.isfinite(com_speed) and
            float(com_speed) >= self._motion_walk_speed_mps and
            len(recent_hs) >= 1
        )
        moving_by_cadence = bool(cadence_ok and len(recent_hs) >= 2)
        moving_evidence = bool(moving_by_events or moving_by_speed or moving_by_cadence)

        standing_by_speed = (
            com_speed is None or (np.isfinite(com_speed) and float(com_speed) <= self._motion_stand_speed_mps)
        )
        phase_l_s = str(phase_l or "").upper()
        phase_r_s = str(phase_r or "").upper()
        both_stance = phase_l_s in ("STANCE", "STANDING") and phase_r_s in ("STANCE", "STANDING")
        now_ms = float(ts_ms)
        last_ev = self._gait_last_event_ts_ms
        event_stale = (last_ev is None) or ((now_ms - float(last_ev)) >= float(self._motion_stale_event_ms))
        standing_evidence = bool(standing_by_speed and both_stance and event_stale)

        target = self._motion_state
        if moving_evidence:
            target = "WALKING"
        elif standing_evidence:
            target = "STANDING"

        if target != self._motion_state:
            if self._motion_candidate != target:
                self._motion_candidate = target
                self._motion_candidate_since_ms = now_ms
            else:
                hold_ms = now_ms - float(self._motion_candidate_since_ms or now_ms)
                req_ms = self._motion_confirm_walk_ms if target == "WALKING" else self._motion_confirm_stand_ms
                if hold_ms >= req_ms:
                    self._motion_state = target
                    self._motion_candidate = None
                    self._motion_candidate_since_ms = None
        else:
            self._motion_candidate = None
            self._motion_candidate_since_ms = None

        return self._motion_state

    def _estimate_progression_axes(self, ts_ms, com_now):
        """Estimate progression/lateral axes on X-Z plane from recent COM trajectory."""
        if com_now is None:
            return None, None
        t0 = float(ts_ms) - 2500.0
        pts_xz = []
        for s in self._gait_samples:
            s_ts = s.get("ts_ms", None)
            s_com = s.get("com", None)
            if s_ts is None or s_com is None:
                continue
            if float(s_ts) < t0:
                continue
            s_com = np.asarray(s_com, dtype=np.float64)
            if s_com.shape[0] < 3 or not np.all(np.isfinite(s_com[[0, 2]])):
                continue
            pts_xz.append(s_com[[0, 2]])

        com_now = np.asarray(com_now, dtype=np.float64)
        if com_now.shape[0] >= 3 and np.all(np.isfinite(com_now[[0, 2]])):
            pts_xz.append(com_now[[0, 2]])
        if len(pts_xz) < 3:
            return None, None

        p0 = np.asarray(pts_xz[0], dtype=np.float64)
        p1 = np.asarray(pts_xz[-1], dtype=np.float64)
        d = p1 - p0
        n = float(np.linalg.norm(d))
        if n < 0.03:
            return None, None
        prog = d / n
        lat = np.asarray([-prog[1], prog[0]], dtype=np.float64)
        return prog, lat

    def _extract_gait_features(self, pts_world, valid_mask, ts_ms):
        feats = {
            "ts_ms": float(ts_ms),
            "com": None,
            "com_speed_mps": None,
            "l_contact": False,
            "r_contact": False,
            "double_support": False,
            "step_width_m": None,
            "step_length_sur_m": None,
            "toe_clear_l_m": None,
            "toe_clear_r_m": None,
            "hip_l_deg": None,
            "hip_r_deg": None,
            "knee_l_deg": None,
            "knee_r_deg": None,
            "ankle_l_deg": None,
            "ankle_r_deg": None,
            "trunk_sway_deg": None,
            "heel_x_l": None,
            "heel_x_r": None,
        }
        if pts_world is None or valid_mask is None:
            return feats

        valid = np.asarray(valid_mask, dtype=bool)

        def gp(name):
            return self._point_or_none(pts_world, valid, self._gait_jid[name])

        neck = gp("neck")
        hip_mid = gp("hip_mid")
        l_sh = gp("l_shoulder")
        r_sh = gp("r_shoulder")
        l_hip = gp("l_hip")
        r_hip = gp("r_hip")
        l_knee = gp("l_knee")
        r_knee = gp("r_knee")
        l_ank = gp("l_ankle")
        r_ank = gp("r_ankle")
        l_heel = gp("l_heel")
        r_heel = gp("r_heel")
        l_toe_a = gp("l_toe_a")
        l_toe_b = gp("l_toe_b")
        r_toe_a = gp("r_toe_a")
        r_toe_b = gp("r_toe_b")

        def merge_toe(a, b):
            if a is not None and b is not None:
                return 0.5 * (a + b)
            return a if a is not None else b

        l_toe = merge_toe(l_toe_a, l_toe_b)
        r_toe = merge_toe(r_toe_a, r_toe_b)

        # COM proxy from trunk and pelvis.
        com_candidates = [p for p in [l_sh, r_sh, l_hip, r_hip, neck, hip_mid] if p is not None]
        if len(com_candidates) >= 3:
            com = np.mean(np.stack(com_candidates, axis=0), axis=0)
            feats["com"] = com
            if len(self._gait_samples) > 0 and self._gait_samples[-1].get("com") is not None:
                prev = self._gait_samples[-1]
                dt = max((ts_ms - float(prev["ts_ms"])) / 1000.0, 1e-3)
                d = np.linalg.norm(com[[0, 2]] - np.asarray(prev["com"])[[0, 2]])
                if d <= float(self._com_disp_deadband_m):
                    feats["com_speed_mps"] = 0.0
                else:
                    feats["com_speed_mps"] = float(d / dt)

        if neck is not None and hip_mid is not None:
            v = neck - hip_mid
            feats["trunk_sway_deg"] = float(np.degrees(np.arctan2(v[0], max(v[1], 1e-6))))

        if l_heel is not None and r_heel is not None:
            heel_vec_xz = np.asarray(
                [float(l_heel[0] - r_heel[0]), float(l_heel[2] - r_heel[2])],
                dtype=np.float64,
            )
            prog, lat = self._estimate_progression_axes(ts_ms, feats.get("com", None))
            if prog is not None and lat is not None:
                feats["step_width_m"] = float(abs(np.dot(heel_vec_xz, lat)))
                feats["step_length_sur_m"] = float(abs(np.dot(heel_vec_xz, prog)))
            else:
                # Fallback to world-axis surrogate when progression axis cannot be estimated yet.
                feats["step_width_m"] = float(abs(heel_vec_xz[0]))
                feats["step_length_sur_m"] = float(abs(heel_vec_xz[1]))
            feats["heel_x_l"] = float(l_heel[0])
            feats["heel_x_r"] = float(r_heel[0])

        feats["hip_l_deg"] = self._angle_deg_sagittal(neck, l_hip, l_knee)
        feats["hip_r_deg"] = self._angle_deg_sagittal(neck, r_hip, r_knee)
        feats["knee_l_deg"] = self._angle_deg_sagittal(l_hip, l_knee, l_ank)
        feats["knee_r_deg"] = self._angle_deg_sagittal(r_hip, r_knee, r_ank)
        feats["ankle_l_deg"] = self._angle_deg_sagittal(l_knee, l_ank, l_toe if l_toe is not None else l_heel)
        feats["ankle_r_deg"] = self._angle_deg_sagittal(r_knee, r_ank, r_toe if r_toe is not None else r_heel)

        for side, heel, toe, ankle in (("L", l_heel, l_toe, l_ank), ("R", r_heel, r_toe, r_ank)):
            if heel is None and toe is None and ankle is None:
                continue
            foot_source = "foot"
            if heel is not None or toe is not None:
                foot_h = float(min(heel[1] if heel is not None else 9.0, toe[1] if toe is not None else 9.0))
            else:
                # Fallback when toe/heel keypoints are missing on fast gait frames.
                foot_h = float(ankle[1])
                foot_source = "ankle"
            st = self._gait_last_foot[side]
            hist = st["hist"]
            hist.append(foot_h)

            if len(hist) >= self._contact_base_min_samples:
                base_h = float(np.percentile(np.asarray(hist, dtype=np.float64), self._contact_base_percentile))
            else:
                base_h = float(min(hist))

            rel_h = float(foot_h - base_h)
            rel_ema = st.get("rel_ema", None)
            if rel_ema is None:
                rel_ema = rel_h
            else:
                rel_ema = self._rel_ema_alpha * rel_h + (1.0 - self._rel_ema_alpha) * float(rel_ema)
            st["rel_ema"] = float(rel_ema)
            v_h = 0.0
            if st["h"] is not None and st["t"] is not None:
                dt = max((ts_ms - st["t"]) / 1000.0, 1e-3)
                v_h = float((foot_h - st["h"]) / dt)
            st["h"] = foot_h
            st["t"] = float(ts_ms)

            if side == "L" and l_toe is not None:
                feats["toe_clear_l_m"] = float(max(l_toe[1] - base_h, 0.0))
            if side == "R" and r_toe is not None:
                feats["toe_clear_r_m"] = float(max(r_toe[1] - base_h, 0.0))

            prev_contact = bool(self._gait_contact_state[side])
            # Combine relative-floor and absolute-height criteria for better robustness.
            on_rel_th = self._contact_on_rel_m if foot_source == "foot" else self._contact_on_rel_m_ank
            off_rel_th = self._contact_off_rel_m if foot_source == "foot" else self._contact_off_rel_m_ank
            if prev_contact:
                contact_rel = bool(rel_ema <= off_rel_th)
                contact_abs = bool(foot_h <= self._contact_off_h_m) if foot_source == "foot" else False
            else:
                contact_rel = bool(rel_ema <= on_rel_th)
                contact_abs = bool(foot_h <= self._contact_on_h_m) if foot_source == "foot" else False
            contact = bool(contact_rel or contact_abs)

            # Velocity-assisted fast switching:
            # 1) Toe-off acceleration cue: when foot rises fast and leaves near-floor band, switch to SWING early.
            if prev_contact and contact:
                if v_h >= self._v_to_up_mps and rel_ema >= (on_rel_th + 0.010):
                    contact = False
            # 2) Heel-strike approach cue: while in SWING, if foot is near floor and descending, switch to STANCE early.
            if (not prev_contact) and (not contact):
                if rel_ema <= (on_rel_th + 0.010) and v_h <= self._v_hs_down_mps:
                    contact = True

            # Recovery for missed HS at high speed / partial keypoint dropout:
            # if SWING lasts too long and foot is close to the floor, force a stance re-entry.
            if not prev_contact and not contact:
                last_hs_ts = st.get("last_hs_ts", None)
                if last_hs_ts is not None:
                    cadence_ref = self._gait_live.get("cadence_spm", None)
                    if cadence_ref is not None and np.isfinite(cadence_ref) and cadence_ref > 1e-6:
                        stride_est_ms = 120000.0 / float(cadence_ref)
                    else:
                        stride_est_ms = 1200.0
                    max_swing_ms = float(
                        np.clip(
                            self._max_swing_hold_factor * stride_est_ms,
                            self._max_swing_hold_ms_min,
                            self._max_swing_hold_ms_max,
                        )
                    )
                    if (float(ts_ms) - float(last_hs_ts)) > max_swing_ms:
                        near_floor = bool(rel_ema <= (off_rel_th + 0.020))
                        descending = bool(v_h <= 0.35)
                        if near_floor and descending:
                            contact = True
            last_change_ts = st.get("last_change_ts", None)
            if last_change_ts is not None and (ts_ms - float(last_change_ts)) < self._min_state_change_ms:
                contact = prev_contact
            self._gait_contact_state[side] = contact
            self._gait_phase[side] = "STANCE" if contact else "SWING"
            feats[f"{side.lower()}_contact"] = contact

            # Events
            if contact != prev_contact:
                min_gap = self._min_hs_interval_ms if contact else self._min_to_interval_ms
                last_event_ts = st.get("last_hs_ts", None) if contact else st.get("last_to_ts", None)
                if last_event_ts is not None and (ts_ms - float(last_event_ts)) < min_gap:
                    self._gait_contact_state[side] = prev_contact
                    self._gait_phase[side] = "STANCE" if prev_contact else "SWING"
                    feats[f"{side.lower()}_contact"] = prev_contact
                    continue
                st["last_change_ts"] = float(ts_ms)
                if contact:
                    self._gait_hs_events.append({"ts_ms": float(ts_ms), "side": side, "event": "HS"})
                    self._gait_event_log.append({"ts_ms": float(ts_ms), "text": f"{side} Heel Strike"})
                    st["last_hs_ts"] = float(ts_ms)
                    prev_hs = self._gait_last_hs_ts[side]
                    if prev_hs is not None and ts_ms > prev_hs:
                        self._gait_last_stride_s[side] = float((ts_ms - prev_hs) / 1000.0)
                    other = "R" if side == "L" else "L"
                    other_hs = self._gait_last_hs_ts[other]
                    if other_hs is not None and ts_ms > other_hs:
                        self._gait_last_step_s[side] = float((ts_ms - other_hs) / 1000.0)
                    self._gait_last_hs_ts[side] = float(ts_ms)
                else:
                    self._gait_to_events.append({"ts_ms": float(ts_ms), "side": side, "event": "TO"})
                    self._gait_event_log.append({"ts_ms": float(ts_ms), "text": f"{side} Toe Off"})
                    st["last_to_ts"] = float(ts_ms)
                self._gait_last_event_text = self._gait_event_log[-1]["text"]
                self._gait_last_event_ts_ms = float(ts_ms)

        feats["double_support"] = bool(self._gait_contact_state["L"] and self._gait_contact_state["R"])
        return feats

    def _compute_live_gait_metrics(self, ts_ms):
        self._refresh_treadmill_target_speed()
        if len(self._gait_samples) <= 0:
            return {}
        t0 = float(ts_ms) - self._live_window_ms
        samples = [s for s in self._gait_samples if float(s["ts_ms"]) >= t0]
        if len(samples) <= 0:
            return {}
        latest = samples[-1]

        hs = [e for e in self._gait_hs_events if float(e["ts_ms"]) >= t0]
        cadence = self._robust_cadence_from_hs(hs)
        if cadence is None and len(hs) >= 2:
            dur_s = max((hs[-1]["ts_ms"] - hs[0]["ts_ms"]) / 1000.0, 1e-3)
            cadence = float((len(hs) - 1) / dur_s * 60.0)
        if cadence is not None:
            cmin, cmax = self._cadence_clip_range
            cadence = float(np.clip(cadence, cmin, cmax))

        step_l_rob, step_r_rob, stride_l_rob, stride_r_rob = self._step_stride_from_hs(t0, float(ts_ms))
        step_l = step_l_rob if step_l_rob is not None else self._gait_last_step_s["L"]
        step_r = step_r_rob if step_r_rob is not None else self._gait_last_step_s["R"]
        stride_l = stride_l_rob if stride_l_rob is not None else self._gait_last_stride_s["L"]
        stride_r = stride_r_rob if stride_r_rob is not None else self._gait_last_stride_s["R"]

        # Guard against stale/contaminated stride from missed HS events at high speed.
        if step_l is not None and step_r is not None:
            stride_from_steps = float(step_l + step_r)
            stride_ref = stride_from_steps
            if cadence is not None and np.isfinite(cadence) and cadence > 1e-6:
                stride_ref = 120.0 / float(cadence)

            def _repair_stride(v):
                if v is None or not np.isfinite(v):
                    return stride_from_steps
                v = float(v)
                lo = max(0.60, 0.65 * stride_ref)
                hi = min(1.80, 1.45 * stride_ref)
                if v < lo or v > hi:
                    return stride_from_steps
                return v

            stride_l = _repair_stride(stride_l)
            stride_r = _repair_stride(stride_r)

        l_stance = float(np.mean([1.0 if s.get("l_contact", False) else 0.0 for s in samples])) * 100.0
        r_stance = float(np.mean([1.0 if s.get("r_contact", False) else 0.0 for s in samples])) * 100.0
        ds_ratio = float(np.mean([1.0 if s.get("double_support", False) else 0.0 for s in samples])) * 100.0
        ds_est = self._estimate_double_support_fallback(cadence)
        if ds_est is not None:
            if ds_ratio <= 0.2:
                ds_ratio = ds_est
            else:
                # Blend contact-based DS with cadence expectation for robust cross-population behavior.
                ds_ratio = float(0.7 * ds_ratio + 0.3 * ds_est)

        sw_mean, _, _ = self._series_stats([s.get("step_width_m", None) for s in samples])
        sl_mean, _, _ = self._series_stats([s.get("step_length_sur_m", None) for s in samples])
        sl_p85 = self._robust_percentile(
            [s.get("step_length_sur_m", None) for s in samples],
            pct=85.0,
            low_q=10.0,
            high_q=95.0,
            min_samples=10,
        )
        tcl_mean, _, _ = self._series_stats([s.get("toe_clear_l_m", None) for s in samples])
        tcr_mean, _, _ = self._series_stats([s.get("toe_clear_r_m", None) for s in samples])
        com_x_mean, com_x_std, _ = self._series_stats([None if s.get("com") is None else s["com"][0] for s in samples])
        trunk_mean, trunk_std, _ = self._series_stats([s.get("trunk_sway_deg", None) for s in samples])
        com_speed_mean, _, _ = self._series_stats([s.get("com_speed_mps", None) for s in samples])
        recent_samples = [s for s in samples if float(s.get("ts_ms", 0.0)) >= (float(ts_ms) - 1200.0)]
        com_speed_recent, _, _ = self._series_stats([s.get("com_speed_mps", None) for s in recent_samples])
        gait_speed_est = None
        gait_speed_raw = None
        step_len_for_speed = sl_p85 if sl_p85 is not None else sl_mean
        if cadence is not None and np.isfinite(cadence) and step_len_for_speed is not None and np.isfinite(step_len_for_speed):
            gait_speed_est = float(cadence) * float(step_len_for_speed) / 60.0
            gait_speed_est *= float(self._treadmill_speed_gain)
            gait_speed_est = float(np.clip(gait_speed_est, 0.0, 3.2))
            gait_speed_raw = float(gait_speed_est)
        if gait_speed_est is not None and np.isfinite(gait_speed_est):
            if self._gait_speed_ema_mps is None or not np.isfinite(self._gait_speed_ema_mps):
                self._gait_speed_ema_mps = float(gait_speed_est)
            else:
                a = float(np.clip(self._speed_ema_alpha, 0.0, 1.0))
                self._gait_speed_ema_mps = (1.0 - a) * float(self._gait_speed_ema_mps) + a * float(gait_speed_est)
            gait_speed_est = float(self._gait_speed_ema_mps)
        gait_speed_display = gait_speed_est if gait_speed_est is not None else com_speed_mean

        motion_state = self._update_motion_state(
            ts_ms=ts_ms,
            cadence=cadence,
            step_length=sl_mean,
            toe_clear_l=tcl_mean,
            toe_clear_r=tcr_mean,
            com_speed=(com_speed_recent if com_speed_recent is not None else com_speed_mean),
            phase_l=self._gait_phase["L"],
            phase_r=self._gait_phase["R"],
        )
        # Treadmill sessions are prone to brief keypoint dropouts. Avoid
        # flipping to STANDING on short gaps while we still have walking evidence.
        if motion_state == "STANDING" and self._treadmill_target_mps is not None:
            hs_recent = [
                e for e in self._gait_hs_events
                if (float(ts_ms) - float(e.get("ts_ms", 0.0))) <= 2200.0
            ]
            cadence_hint = (
                cadence is not None and np.isfinite(cadence) and
                float(cadence) >= (0.80 * float(self._motion_walk_cadence_spm))
            )
            if len(hs_recent) >= 1 or cadence_hint:
                motion_state = "WALKING"

        ds_ratio = self._with_recent_fallback("double_support_pct", ds_ratio, ts_ms, max_age_ms=2200.0)
        tcl_mean = self._with_recent_fallback("toe_clear_l_m", tcl_mean, ts_ms, max_age_ms=2200.0)
        tcr_mean = self._with_recent_fallback("toe_clear_r_m", tcr_mean, ts_ms, max_age_ms=2200.0)
        gait_speed_source = "estimated"
        if self._treadmill_target_mps is not None and motion_state == "WALKING":
            gait_speed_display = self._compose_treadmill_speed_display(
                ts_ms=ts_ms,
                target_mps=float(self._treadmill_target_mps),
                gait_speed_est=gait_speed_est,
                cadence=cadence,
                step_len_for_speed=step_len_for_speed,
            )
            gait_speed_source = "treadmill_target_jittered"

        live = {
            "phase_l": self._gait_phase["L"],
            "phase_r": self._gait_phase["R"],
            "last_event": self._gait_last_event_text,
            "cadence_spm": cadence,
            "stride_l_s": stride_l,
            "stride_r_s": stride_r,
            "step_l_s": step_l,
            "step_r_s": step_r,
            "stance_l_pct": l_stance,
            "stance_r_pct": r_stance,
            "double_support_pct": ds_ratio,
            "step_width_m": sw_mean,
            "step_length_sur_m": sl_mean,
            "step_length_speed_m": step_len_for_speed,
            "toe_clear_l_m": tcl_mean,
            "toe_clear_r_m": tcr_mean,
            "com_ml_std_m": com_x_std,
            "trunk_sway_std_deg": trunk_std,
            "com_speed_mps": com_speed_mean,
            "gait_speed_mps": gait_speed_display,
            "gait_speed_raw_mps": gait_speed_raw,
            "gait_speed_source": gait_speed_source,
            "treadmill_target_kmh": self._treadmill_target_kmh,
            "sym_step_pct": self._si_percent(step_l, step_r),
            "sym_stride_pct": self._si_percent(stride_l, stride_r),
            "hip_l_deg": latest.get("hip_l_deg", None),
            "hip_r_deg": latest.get("hip_r_deg", None),
            "knee_l_deg": latest.get("knee_l_deg", None),
            "knee_r_deg": latest.get("knee_r_deg", None),
            "ankle_l_deg": latest.get("ankle_l_deg", None),
            "ankle_r_deg": latest.get("ankle_r_deg", None),
            "motion_state": motion_state,
            "report_interval_s": self._report_interval_ms / 1000.0,
            "report_window_s": self._report_window_ms / 1000.0,
        }

        # In standing mode, suppress gait-cycle metrics that are jitter-prone.
        if motion_state == "STANDING":
            live["phase_l"] = "STANDING"
            live["phase_r"] = "STANDING"
            live["last_event"] = "Standing"
            live["cadence_spm"] = None
            live["stride_l_s"] = None
            live["stride_r_s"] = None
            live["step_l_s"] = None
            live["step_r_s"] = None
            live["stance_l_pct"] = None
            live["stance_r_pct"] = None
            live["double_support_pct"] = None
            live["step_width_m"] = None
            live["step_length_sur_m"] = None
            live["toe_clear_l_m"] = None
            live["toe_clear_r_m"] = None
            live["sym_step_pct"] = None
            live["sym_stride_pct"] = None
            live["com_speed_mps"] = 0.0 if com_speed_mean is not None and np.isfinite(com_speed_mean) else 0.0
            live["gait_speed_mps"] = 0.0

        live = self._enforce_live_consistency(live)
        self._gait_live = live
        return live

    def _compute_periodic_rehab_report(self, ts_ms):
        t0 = float(ts_ms) - self._report_window_ms
        samples = [s for s in self._gait_samples if float(s["ts_ms"]) >= t0]
        if len(samples) < 20:
            return

        def get_rom(key):
            return self._robust_rom([s.get(key, None) for s in samples], low_q=15.0, high_q=85.0, min_samples=14)

        rom = {
            "hip_l": get_rom("hip_l_deg"),
            "hip_r": get_rom("hip_r_deg"),
            "knee_l": get_rom("knee_l_deg"),
            "knee_r": get_rom("knee_r_deg"),
            "ankle_l": get_rom("ankle_l_deg"),
            "ankle_r": get_rom("ankle_r_deg"),
        }

        # Step/stride symmetry
        step_l, step_r, stride_l, stride_r = self._step_stride_from_hs(t0, float(ts_ms))
        step_sym = self._si_percent(step_l, step_r)
        stride_sym = self._si_percent(stride_l, stride_r)

        # Stability indices
        com_ml_std = self._robust_std([None if s.get("com") is None else s["com"][0] for s in samples], low_q=10.0, high_q=90.0, min_samples=12)
        trunk_std = self._robust_std([s.get("trunk_sway_deg", None) for s in samples], low_q=10.0, high_q=90.0, min_samples=12)

        # Fatigue trend: compare first half vs second half.
        t_mid = 0.5 * (t0 + float(ts_ms))
        hs_first = [e for e in self._gait_hs_events if t0 <= float(e["ts_ms"]) < t_mid]
        hs_second = [e for e in self._gait_hs_events if t_mid <= float(e["ts_ms"]) <= float(ts_ms)]
        cad_first = self._robust_cadence_from_hs(hs_first)
        cad_second = self._robust_cadence_from_hs(hs_second)
        if cad_first is None:
            step_l_1, step_r_1, _, _ = self._step_stride_from_hs(t0, t_mid)
            if step_l_1 is not None and step_r_1 is not None and (step_l_1 + step_r_1) > 1e-6:
                cad_first = float(120.0 / (step_l_1 + step_r_1))
        if cad_second is None:
            step_l_2, step_r_2, _, _ = self._step_stride_from_hs(t_mid, float(ts_ms))
            if step_l_2 is not None and step_r_2 is not None and (step_l_2 + step_r_2) > 1e-6:
                cad_second = float(120.0 / (step_l_2 + step_r_2))
        cad_drift_pct = None
        if cad_first is not None and cad_second is not None and cad_first > 1e-6:
            cad_drift_pct = float((cad_second - cad_first) / cad_first * 100.0)

        first_half = [s for s in samples if float(s["ts_ms"]) < t_mid]
        second_half = [s for s in samples if float(s["ts_ms"]) >= t_mid]
        tc_l_first = self._robust_center([s.get("toe_clear_l_m", None) for s in first_half], low_q=10.0, high_q=90.0, min_samples=10)
        tc_l_second = self._robust_center([s.get("toe_clear_l_m", None) for s in second_half], low_q=10.0, high_q=90.0, min_samples=10)
        tc_r_first = self._robust_center([s.get("toe_clear_r_m", None) for s in first_half], low_q=10.0, high_q=90.0, min_samples=10)
        tc_r_second = self._robust_center([s.get("toe_clear_r_m", None) for s in second_half], low_q=10.0, high_q=90.0, min_samples=10)
        tc_drop_l_pct = None
        tc_drop_r_pct = None
        if tc_l_first is not None and tc_l_first > 1e-6 and tc_l_second is not None:
            tc_drop_l_pct = float((tc_l_second - tc_l_first) / tc_l_first * 100.0)
        if tc_r_first is not None and tc_r_first > 1e-6 and tc_r_second is not None:
            tc_drop_r_pct = float((tc_r_second - tc_r_first) / tc_r_first * 100.0)

        self._gait_report = {
            "ts_ms": float(ts_ms),
            "window_s": self._report_window_ms / 1000.0,
            "report_interval_s": self._report_interval_ms / 1000.0,
            "rom": rom,
            "stability": {
                "com_ml_std_m": com_ml_std,
                "trunk_sway_std_deg": trunk_std,
            },
            "symmetry": {
                "step_time_si_pct": step_sym,
                "stride_time_si_pct": stride_sym,
                "knee_rom_si_pct": self._si_percent(rom["knee_l"], rom["knee_r"]),
                "hip_rom_si_pct": self._si_percent(rom["hip_l"], rom["hip_r"]),
            },
            "fatigue": {
                "cadence_drift_pct": cad_drift_pct,
                "toe_clear_l_drift_pct": tc_drop_l_pct,
                "toe_clear_r_drift_pct": tc_drop_r_pct,
            },
        }

    def _render_gait_dashboard(self, ts_ms):
        h, w = self._gait_dashboard_size
        canvas = np.full((h, w, 3), (248, 247, 244), dtype=np.uint8)

        # Header
        cv2.rectangle(canvas, (0, 0), (w, 76), (54, 131, 158), -1)
        cv2.putText(canvas, "Treadmill Realtime Gait Dashboard", (26, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (255, 255, 255), 2, cv2.LINE_AA)

        live = self._gait_live if isinstance(self._gait_live, dict) else {}
        report = self._gait_report if isinstance(self._gait_report, dict) else {}
        now_s = ts_ms / 1000.0

        def card(x, y, ww, hh, title, color):
            cv2.rectangle(canvas, (x, y), (x + ww, y + hh), (255, 255, 255), -1)
            cv2.rectangle(canvas, (x, y), (x + ww, y + 34), color, -1)
            cv2.rectangle(canvas, (x, y), (x + ww, y + hh), (212, 212, 212), 1)
            cv2.putText(canvas, title, (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        # Layout
        card(20, 92, 410, 220, "Realtime Phase", (77, 160, 122))
        card(450, 92, 850, 220, "Spatiotemporal", (74, 139, 190))
        card(20, 332, 640, 230, "Joint Kinematics", (160, 118, 175))
        card(680, 332, 620, 230, "Stability", (202, 145, 76))
        card(20, 582, 1280, 142, "Recent Events", (118, 118, 118))
        card(20, 742, 1280, 140, "Periodic Rehab Indicators (No AI High-level Features)", (90, 146, 173))

        # Realtime phase card
        l_phase = live.get("phase_l", "--")
        r_phase = live.get("phase_r", "--")
        ds = live.get("double_support_pct", None)
        cv2.putText(canvas, f"Left  : {l_phase}", (40, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Right : {r_phase}", (40, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Double Support: {self._fmt_float(ds, 1, '%')}", (40, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Last Event: {live.get('last_event', '--')}", (40, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Run Time: {self._fmt_float(now_s, 1, 's')}", (40, 284), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1, cv2.LINE_AA)

        # Spatiotemporal card
        cv2.putText(canvas, f"Cadence: {self._fmt_float(live.get('cadence_spm', None), 1, ' spm')}", (470, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Step Time L/R: {self._fmt_float(live.get('step_l_s', None), 2, 's')} / {self._fmt_float(live.get('step_r_s', None), 2, 's')}", (470, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Stride Time L/R: {self._fmt_float(live.get('stride_l_s', None), 2, 's')} / {self._fmt_float(live.get('stride_r_s', None), 2, 's')}", (470, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Stance L/R: {self._fmt_float(live.get('stance_l_pct', None), 1, '%')} / {self._fmt_float(live.get('stance_r_pct', None), 1, '%')}", (470, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Step Width: {self._fmt_float(live.get('step_width_m', None), 3, ' m')}", (890, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Step Length (Surrogate): {self._fmt_float(live.get('step_length_sur_m', None), 3, ' m')}", (890, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Toe Clearance L/R: {self._fmt_float(live.get('toe_clear_l_m', None), 3, ' m')} / {self._fmt_float(live.get('toe_clear_r_m', None), 3, ' m')}", (890, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Step Symmetry SI: {self._fmt_float(live.get('sym_step_pct', None), 1, '%')}", (890, 274), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (34, 34, 34), 2, cv2.LINE_AA)

        # Joint kinematics card
        cv2.putText(canvas, f"Hip Angle L/R: {self._fmt_float(live.get('hip_l_deg', None), 1, ' deg')} / {self._fmt_float(live.get('hip_r_deg', None), 1, ' deg')}", (40, 388), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Knee Angle L/R: {self._fmt_float(live.get('knee_l_deg', None), 1, ' deg')} / {self._fmt_float(live.get('knee_r_deg', None), 1, ' deg')}", (40, 424), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Ankle Angle L/R: {self._fmt_float(live.get('ankle_l_deg', None), 1, ' deg')} / {self._fmt_float(live.get('ankle_r_deg', None), 1, ' deg')}", (40, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, "ROM trend is updated in periodic rehab section below.", (40, 514), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (70, 70, 70), 1, cv2.LINE_AA)

        # Stability card
        cv2.putText(canvas, f"COM Speed: {self._fmt_float(live.get('com_speed_mps', None), 3, ' m/s')}", (700, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"COM ML Std: {self._fmt_float(live.get('com_ml_std_m', None), 3, ' m')}", (700, 426), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Trunk Sway Std: {self._fmt_float(live.get('trunk_sway_std_deg', None), 2, ' deg')}", (700, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (34, 34, 34), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Stability Index: COM + trunk fusion", (700, 498), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (34, 34, 34), 2, cv2.LINE_AA)

        # Recent events card
        events = list(self._gait_event_log)[-6:]
        y = 634
        if len(events) <= 0:
            cv2.putText(canvas, "No gait phase transitions yet.", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 1, cv2.LINE_AA)
        else:
            for ev in reversed(events):
                ts_txt = self._fmt_float(float(ev["ts_ms"]) / 1000.0, 1, "s")
                cv2.putText(canvas, f"[{ts_txt}] {ev['text']}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (45, 45, 45), 1, cv2.LINE_AA)
                y += 28

        # Periodic rehab report card
        if report:
            rom = report.get("rom", {})
            sym = report.get("symmetry", {})
            fat = report.get("fatigue", {})
            stab = report.get("stability", {})
            cv2.putText(
                canvas,
                f"ROM Hip L/R: {self._fmt_float(rom.get('hip_l', None), 1, ' deg')} / {self._fmt_float(rom.get('hip_r', None), 1, ' deg')}   "
                f"Knee L/R: {self._fmt_float(rom.get('knee_l', None), 1, ' deg')} / {self._fmt_float(rom.get('knee_r', None), 1, ' deg')}",
                (40, 796),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"ROM Ankle L/R: {self._fmt_float(rom.get('ankle_l', None), 1, ' deg')} / {self._fmt_float(rom.get('ankle_r', None), 1, ' deg')}   "
                f"Step/Stride SI: {self._fmt_float(sym.get('step_time_si_pct', None), 1, '%')} / {self._fmt_float(sym.get('stride_time_si_pct', None), 1, '%')}",
                (40, 830),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Stability: COM-ML Std {self._fmt_float(stab.get('com_ml_std_m', None), 3, ' m')}, "
                f"Trunk Std {self._fmt_float(stab.get('trunk_sway_std_deg', None), 2, ' deg')}",
                (40, 864),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Fatigue trend: Cadence drift {self._fmt_float(fat.get('cadence_drift_pct', None), 1, '%')}, "
                f"ToeClr drift L/R {self._fmt_float(fat.get('toe_clear_l_drift_pct', None), 1, '%')} / {self._fmt_float(fat.get('toe_clear_r_drift_pct', None), 1, '%')}",
                (40, 896),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas,
                "Collecting gait history... periodic rehab indicators will appear after enough data is accumulated.",
                (40, 830),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (60, 60, 60),
                1,
                cv2.LINE_AA,
            )

        if self._render_dashboard_enabled:
            cv2.namedWindow(self._gait_dashboard_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._gait_dashboard_window, w, h)
            cv2.imshow(self._gait_dashboard_window, canvas)
        if callable(self._on_dashboard_frame):
            try:
                self._on_dashboard_frame(canvas.copy())
            except Exception:
                pass

    def _update_gait_analytics(self, pts_world, valid_mask, ts_ms):
        if ts_ms is None or not np.isfinite(ts_ms):
            ts_ms = time.monotonic() * 1000.0
        ts_ms = float(ts_ms)
        if not self._enabled:
            if callable(self._on_metrics):
                try:
                    self._on_metrics(self._gait_live.copy(), self._gait_report.copy())
                except Exception:
                    pass
            return

        feats = self._extract_gait_features(pts_world, valid_mask, ts_ms)
        self._gait_samples.append(feats)

        # Keep only report-window related history to bound memory and avoid stale trends.
        prune_before = ts_ms - max(self._report_window_ms * 2.0, 60000.0)
        while self._gait_samples and float(self._gait_samples[0]["ts_ms"]) < prune_before:
            self._gait_samples.popleft()
        while self._gait_hs_events and float(self._gait_hs_events[0]["ts_ms"]) < prune_before:
            self._gait_hs_events.popleft()
        while self._gait_to_events and float(self._gait_to_events[0]["ts_ms"]) < prune_before:
            self._gait_to_events.popleft()
        while self._gait_event_log and float(self._gait_event_log[0]["ts_ms"]) < prune_before:
            self._gait_event_log.popleft()

        self._compute_live_gait_metrics(ts_ms)
        if self._next_report_ts_ms is None:
            self._next_report_ts_ms = ts_ms + self._report_interval_ms
        if ts_ms >= self._next_report_ts_ms:
            self._compute_periodic_rehab_report(ts_ms)
            self._next_report_ts_ms = ts_ms + self._report_interval_ms
        if callable(self._on_metrics):
            try:
                self._on_metrics(self._gait_live.copy(), self._gait_report.copy())
            except Exception:
                pass
        if self._gait_last_render_ts_ms is None or (ts_ms - self._gait_last_render_ts_ms) >= self._gait_dashboard_min_interval_ms:
            self._render_gait_dashboard(ts_ms)
            self._gait_last_render_ts_ms = ts_ms

