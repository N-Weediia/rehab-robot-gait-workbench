"""
Safety filter for gait metrics.

Purpose:
1) Prevent obvious outliers from polluting GUI and report.
2) Reuse the last stable-normal value when current value is implausible.
"""

import copy
import math
import numbers
import os
from typing import Any, Dict, Tuple


class GaitMetricSafetyFilter:
    def __init__(self):
        self._population_profile = str(os.environ.get("GAIT_POPULATION_PROFILE", "adult_general")).strip().lower()
        self._last_normal: Dict[str, float] = {}
        self._rules = self._build_rules(self._population_profile)

    @staticmethod
    def _build_rules(profile: str = "adult_general") -> Dict[str, Tuple[float, float]]:
        if profile == "healthy_strict":
            return {
                "live.cadence_spm": (80.0, 150.0),
                "live.step_l_s": (0.30, 0.90),
                "live.step_r_s": (0.30, 0.90),
                "live.stride_l_s": (0.65, 1.70),
                "live.stride_r_s": (0.65, 1.70),
                "live.stance_l_pct": (48.0, 70.0),
                "live.stance_r_pct": (48.0, 70.0),
                "live.double_support_pct": (8.0, 32.0),
                "live.step_width_m": (0.02, 0.30),
                "live.step_length_sur_m": (0.15, 1.10),
                "live.toe_clear_l_m": (0.004, 0.12),
                "live.toe_clear_r_m": (0.004, 0.12),
                "live.com_ml_std_m": (0.005, 0.12),
                "live.trunk_sway_std_deg": (0.5, 10.0),
                "live.com_speed_mps": (0.0, 2.30),
                "live.sym_step_pct": (0.0, 15.0),
                "live.sym_stride_pct": (0.0, 15.0),
                "live.hip_l_deg": (10.0, 210.0),
                "live.hip_r_deg": (10.0, 210.0),
                "live.knee_l_deg": (10.0, 210.0),
                "live.knee_r_deg": (10.0, 210.0),
                "live.ankle_l_deg": (10.0, 210.0),
                "live.ankle_r_deg": (10.0, 210.0),
                "report.rom.hip_l": (15.0, 120.0),
                "report.rom.hip_r": (15.0, 120.0),
                "report.rom.knee_l": (20.0, 150.0),
                "report.rom.knee_r": (20.0, 150.0),
                "report.rom.ankle_l": (10.0, 170.0),
                "report.rom.ankle_r": (10.0, 170.0),
                "report.stability.com_ml_std_m": (0.005, 0.14),
                "report.stability.trunk_sway_std_deg": (0.5, 12.0),
                "report.symmetry.step_time_si_pct": (0.0, 15.0),
                "report.symmetry.stride_time_si_pct": (0.0, 15.0),
                "report.symmetry.knee_rom_si_pct": (0.0, 25.0),
                "report.symmetry.hip_rom_si_pct": (0.0, 25.0),
                "report.fatigue.cadence_drift_pct": (-20.0, 20.0),
                "report.fatigue.toe_clear_l_drift_pct": (-25.0, 25.0),
                "report.fatigue.toe_clear_r_drift_pct": (-25.0, 25.0),
            }
        if profile == "young_strict":
            return {
                "live.cadence_spm": (75.0, 165.0),
                "live.step_l_s": (0.28, 0.95),
                "live.step_r_s": (0.28, 0.95),
                "live.stride_l_s": (0.60, 1.90),
                "live.stride_r_s": (0.60, 1.90),
                "live.stance_l_pct": (44.0, 74.0),
                "live.stance_r_pct": (44.0, 74.0),
                "live.double_support_pct": (6.0, 40.0),
                "live.step_width_m": (0.02, 0.40),
                "live.step_length_sur_m": (0.12, 1.30),
                "live.toe_clear_l_m": (0.003, 0.15),
                "live.toe_clear_r_m": (0.003, 0.15),
                "live.com_ml_std_m": (0.005, 0.16),
                "live.trunk_sway_std_deg": (0.5, 12.0),
                "live.com_speed_mps": (0.0, 2.30),
                "live.sym_step_pct": (0.0, 45.0),
                "live.sym_stride_pct": (0.0, 45.0),
                "live.hip_l_deg": (10.0, 210.0),
                "live.hip_r_deg": (10.0, 210.0),
                "live.knee_l_deg": (10.0, 210.0),
                "live.knee_r_deg": (10.0, 210.0),
                "live.ankle_l_deg": (10.0, 210.0),
                "live.ankle_r_deg": (10.0, 210.0),
                "report.rom.hip_l": (15.0, 120.0),
                "report.rom.hip_r": (15.0, 120.0),
                "report.rom.knee_l": (20.0, 150.0),
                "report.rom.knee_r": (20.0, 150.0),
                "report.rom.ankle_l": (10.0, 170.0),
                "report.rom.ankle_r": (10.0, 170.0),
                "report.stability.com_ml_std_m": (0.005, 0.14),
                "report.stability.trunk_sway_std_deg": (0.5, 12.0),
                "report.symmetry.step_time_si_pct": (0.0, 45.0),
                "report.symmetry.stride_time_si_pct": (0.0, 45.0),
                "report.symmetry.knee_rom_si_pct": (0.0, 45.0),
                "report.symmetry.hip_rom_si_pct": (0.0, 45.0),
                "report.fatigue.cadence_drift_pct": (-35.0, 35.0),
                "report.fatigue.toe_clear_l_drift_pct": (-40.0, 40.0),
                "report.fatigue.toe_clear_r_drift_pct": (-40.0, 40.0),
            }
        # General-adult treadmill walking bounds (covering healthy young to many older adults).
        # Keep bounds clinically plausible but not over-restrictive.
        return {
            # Live metrics
            "live.cadence_spm": (65.0, 160.0),
            "live.step_l_s": (0.30, 1.10),
            "live.step_r_s": (0.30, 1.10),
            "live.stride_l_s": (0.65, 2.20),
            "live.stride_r_s": (0.65, 2.20),
            "live.stance_l_pct": (40.0, 80.0),
            "live.stance_r_pct": (40.0, 80.0),
            "live.double_support_pct": (6.0, 45.0),
            "live.step_width_m": (0.02, 0.45),
            "live.step_length_sur_m": (0.10, 1.40),
            "live.toe_clear_l_m": (0.003, 0.16),
            "live.toe_clear_r_m": (0.003, 0.16),
            "live.com_ml_std_m": (0.005, 0.18),
            "live.trunk_sway_std_deg": (0.5, 14.0),
            "live.com_speed_mps": (0.0, 2.20),
            "live.sym_step_pct": (0.0, 55.0),
            "live.sym_stride_pct": (0.0, 55.0),
            "live.hip_l_deg": (10.0, 210.0),
            "live.hip_r_deg": (10.0, 210.0),
            "live.knee_l_deg": (10.0, 210.0),
            "live.knee_r_deg": (10.0, 210.0),
            "live.ankle_l_deg": (10.0, 210.0),
            "live.ankle_r_deg": (10.0, 210.0),

            # Periodic report
            "report.rom.hip_l": (15.0, 120.0),
            "report.rom.hip_r": (15.0, 120.0),
            "report.rom.knee_l": (20.0, 150.0),
            "report.rom.knee_r": (20.0, 150.0),
            "report.rom.ankle_l": (10.0, 170.0),
            "report.rom.ankle_r": (10.0, 170.0),
            "report.stability.com_ml_std_m": (0.005, 0.15),
            "report.stability.trunk_sway_std_deg": (0.5, 14.0),
            "report.symmetry.step_time_si_pct": (0.0, 55.0),
            "report.symmetry.stride_time_si_pct": (0.0, 55.0),
            "report.symmetry.knee_rom_si_pct": (0.0, 55.0),
            "report.symmetry.hip_rom_si_pct": (0.0, 55.0),
            "report.fatigue.cadence_drift_pct": (-40.0, 40.0),
            "report.fatigue.toe_clear_l_drift_pct": (-45.0, 45.0),
            "report.fatigue.toe_clear_r_drift_pct": (-45.0, 45.0),
        }

    def sanitize(self, live_metrics: Dict[str, Any], periodic_report: Dict[str, Any]):
        live = copy.deepcopy(live_metrics or {})
        report = copy.deepcopy(periodic_report or {})

        quality = {
            "replaced": [],
            "hidden": [],
            "total_flags": 0,
        }

        self._sanitize_dict(live, "live", quality)
        self._sanitize_dict(report, "report", quality)
        self._sanitize_consistency(live, report, quality)
        quality["total_flags"] = len(quality["replaced"]) + len(quality["hidden"])
        return live, report, quality

    @staticmethod
    def _estimate_stance_from_cadence(cadence_spm):
        if not isinstance(cadence_spm, numbers.Real):
            return 60.0
        stance = 62.0 - 0.10 * (float(cadence_spm) - 90.0)
        return float(min(max(stance, 52.0), 72.0))

    @staticmethod
    def _estimate_ds_from_cadence(cadence_spm):
        if not isinstance(cadence_spm, numbers.Real):
            return 22.0
        ds = 23.0 - 0.11 * (float(cadence_spm) - 85.0)
        return float(min(max(ds, 10.0), 34.0))

    def _sanitize_consistency(self, live: Dict[str, Any], report: Dict[str, Any], quality: Dict[str, Any]):
        motion_state = str(live.get("motion_state", "")).strip().upper()
        if motion_state in ("STANDING", "IDLE"):
            # Standing mode is not expected to satisfy treadmill walking consistency constraints.
            return

        # Cadence should be consistent with bilateral step times.
        l = live.get("step_l_s", None)
        r = live.get("step_r_s", None)
        cadence = live.get("cadence_spm", None)
        if isinstance(l, numbers.Real) and isinstance(r, numbers.Real):
            ssum = float(l) + float(r)
            if ssum > 1e-6:
                cadence_from_step = 120.0 / ssum
                c_lo, c_hi = self._rules["live.cadence_spm"]
                if c_lo <= cadence_from_step <= c_hi:
                    if not isinstance(cadence, numbers.Real):
                        live["cadence_spm"] = cadence_from_step
                        quality["replaced"].append(
                            {
                                "path": "live.cadence_spm",
                                "reason": "derived_from_step_time",
                                "raw": None,
                                "used": cadence_from_step,
                            }
                        )
                        self._last_normal["live.cadence_spm"] = cadence_from_step
                    else:
                        rel_err = abs(float(cadence) - cadence_from_step) / max(cadence_from_step, 1e-6)
                        if rel_err > 0.12:
                            prev = float(cadence)
                            live["cadence_spm"] = cadence_from_step
                            quality["replaced"].append(
                                {
                                    "path": "live.cadence_spm",
                                    "reason": "inconsistent_with_step_time",
                                    "raw": prev,
                                    "used": cadence_from_step,
                                }
                            )
                            self._last_normal["live.cadence_spm"] = cadence_from_step

        # Phase/state consistency for treadmill walking: both SWING in same frame is implausible.
        phase_l = str(live.get("phase_l", "")).upper()
        phase_r = str(live.get("phase_r", "")).upper()
        if phase_l == "SWING" and phase_r == "SWING":
            raw_l = live.get("phase_l", None)
            raw_r = live.get("phase_r", None)
            event_txt = str(live.get("last_event", "")).upper()
            if "L TOE OFF" in event_txt:
                live["phase_r"] = "STANCE"
            elif "R TOE OFF" in event_txt:
                live["phase_l"] = "STANCE"
            elif "L HEEL STRIKE" in event_txt:
                live["phase_l"] = "STANCE"
            elif "R HEEL STRIKE" in event_txt:
                live["phase_r"] = "STANCE"
            else:
                live["phase_r"] = "STANCE"
            quality["replaced"].append(
                {
                    "path": "live.phase",
                    "reason": "invalid_both_swing_phase",
                    "raw": {"phase_l": raw_l, "phase_r": raw_r},
                    "used": {"phase_l": live.get("phase_l"), "phase_r": live.get("phase_r")},
                }
            )

        # Spatiotemporal consistency: L_stance + R_stance - DS should be ~100% in walking.
        stance_l = live.get("stance_l_pct", None)
        stance_r = live.get("stance_r_pct", None)
        ds = live.get("double_support_pct", None)
        cadence_now = live.get("cadence_spm", None)
        stance_est = self._estimate_stance_from_cadence(cadence_now)
        ds_est = self._estimate_ds_from_cadence(cadence_now)

        if not isinstance(stance_l, numbers.Real):
            stance_l = stance_est
        if not isinstance(stance_r, numbers.Real):
            stance_r = stance_est
        if not isinstance(ds, numbers.Real):
            ds = ds_est
        stance_l = float(stance_l)
        stance_r = float(stance_r)
        ds = float(ds)

        corrected = False
        raw_triplet = {"stance_l_pct": live.get("stance_l_pct"), "stance_r_pct": live.get("stance_r_pct"), "double_support_pct": live.get("double_support_pct")}
        if stance_l < 40.0 or stance_r < 40.0:
            half_diff = max(min((stance_l - stance_r) * 0.5, 6.0), -6.0)
            stance_l = float(min(max(stance_est + half_diff, 45.0), 76.0))
            stance_r = float(min(max(stance_est - half_diff, 45.0), 76.0))
            corrected = True
        ds_lo = max(6.0, ds_est - 10.0)
        ds_hi = min(42.0, ds_est + 10.0, stance_l, stance_r)
        if ds_hi < ds_lo:
            ds_hi = ds_lo
        ds = float(min(max(ds, ds_lo), ds_hi))
        union_contact = stance_l + stance_r - ds
        if union_contact < 90.0 or union_contact > 108.0:
            ds = float(min(max(stance_l + stance_r - 100.0, ds_lo), min(stance_l, stance_r, ds_hi)))
            union_contact = stance_l + stance_r - ds
            corrected = True
            if union_contact < 90.0 or union_contact > 108.0:
                stance_l = float(stance_est)
                stance_r = float(stance_est)
                ds = float(min(max(ds_est, 10.0), min(stance_l, stance_r, 34.0)))
        if corrected:
            live["stance_l_pct"] = stance_l
            live["stance_r_pct"] = stance_r
            live["double_support_pct"] = ds
            quality["replaced"].append(
                {
                    "path": "live.stance_double_support",
                    "reason": "inconsistent_stance_double_support_triplet",
                    "raw": raw_triplet,
                    "used": {
                        "stance_l_pct": stance_l,
                        "stance_r_pct": stance_r,
                        "double_support_pct": ds,
                    },
                }
            )
            self._last_normal["live.stance_l_pct"] = stance_l
            self._last_normal["live.stance_r_pct"] = stance_r
            self._last_normal["live.double_support_pct"] = ds

        # Periodic step SI should be consistent with live step symmetry when both are valid.
        try:
            sym = report.get("symmetry", {}) if isinstance(report, dict) else {}
            step_si_report = sym.get("step_time_si_pct", None) if isinstance(sym, dict) else None
            step_si_live = live.get("sym_step_pct", None)
            if isinstance(step_si_report, numbers.Real) and isinstance(step_si_live, numbers.Real):
                if abs(float(step_si_report) - float(step_si_live)) > 15.0:
                    sym["step_time_si_pct"] = float(step_si_live)
                    quality["replaced"].append(
                        {
                            "path": "report.symmetry.step_time_si_pct",
                            "reason": "inconsistent_with_live_symmetry",
                            "raw": float(step_si_report),
                            "used": float(step_si_live),
                        }
                    )
        except Exception:
            pass

        # Healthy profile anti-jitter: limit frame-to-frame SI jumps.
        if self._population_profile == "healthy_strict":
            self._limit_healthy_symmetry_jumps(live, report, quality)

    def _limit_healthy_symmetry_jumps(self, live: Dict[str, Any], report: Dict[str, Any], quality: Dict[str, Any]):
        # Keep output stable for healthy treadmill tests; suppress pathological transient spikes.
        max_delta = 4.0
        for path, container, key in (
            ("live.sym_step_pct", live, "sym_step_pct"),
            ("live.sym_stride_pct", live, "sym_stride_pct"),
        ):
            val = container.get(key, None)
            if not isinstance(val, numbers.Real):
                continue
            prev = self._last_normal.get(path, None)
            if isinstance(prev, numbers.Real):
                delta = float(val) - float(prev)
                if abs(delta) > max_delta:
                    limited = float(prev) + max_delta * (1.0 if delta > 0 else -1.0)
                    lo, hi = self._rules.get(path, (None, None))
                    if isinstance(lo, numbers.Real) and isinstance(hi, numbers.Real):
                        limited = float(min(max(limited, float(lo)), float(hi)))
                    container[key] = limited
                    quality["replaced"].append(
                        {
                            "path": path,
                            "reason": "healthy_symmetry_jump_limited",
                            "raw": float(val),
                            "used": limited,
                        }
                    )
                    self._last_normal[path] = limited

        sym = report.get("symmetry", {}) if isinstance(report, dict) else {}
        if isinstance(sym, dict):
            for path, key in (
                ("report.symmetry.step_time_si_pct", "step_time_si_pct"),
                ("report.symmetry.stride_time_si_pct", "stride_time_si_pct"),
            ):
                val = sym.get(key, None)
                if not isinstance(val, numbers.Real):
                    continue
                prev = self._last_normal.get(path, None)
                if isinstance(prev, numbers.Real):
                    delta = float(val) - float(prev)
                    if abs(delta) > max_delta:
                        limited = float(prev) + max_delta * (1.0 if delta > 0 else -1.0)
                        lo, hi = self._rules.get(path, (None, None))
                        if isinstance(lo, numbers.Real) and isinstance(hi, numbers.Real):
                            limited = float(min(max(limited, float(lo)), float(hi)))
                        sym[key] = limited
                        quality["replaced"].append(
                            {
                                "path": path,
                                "reason": "healthy_symmetry_jump_limited",
                                "raw": float(val),
                                "used": limited,
                            }
                        )
                        self._last_normal[path] = limited

    def _sanitize_dict(self, data: Dict[str, Any], prefix: str, quality: Dict[str, Any]):
        if not isinstance(data, dict):
            return
        for key, value in list(data.items()):
            path = f"{prefix}.{key}"
            if isinstance(value, dict):
                self._sanitize_dict(value, path, quality)
                continue
            if not isinstance(value, numbers.Real):
                continue
            data[key] = self._sanitize_scalar(path, float(value), quality)

    def _sanitize_scalar(self, path: str, value: float, quality: Dict[str, Any]):
        if not math.isfinite(value):
            return self._fallback(path, "non_finite", value, quality)

        bounds = self._rules.get(path)
        if bounds is not None:
            lo, hi = bounds
            if value < lo or value > hi:
                return self._fallback(path, "out_of_range", value, quality)

        self._last_normal[path] = value
        return value

    def _fallback(self, path: str, reason: str, raw_value: float, quality: Dict[str, Any]):
        if path in self._last_normal:
            fallback = self._last_normal[path]
            quality["replaced"].append(
                {
                    "path": path,
                    "reason": reason,
                    "raw": raw_value,
                    "used": fallback,
                }
            )
            return fallback
        bounds = self._rules.get(path)
        if bounds is not None and math.isfinite(raw_value):
            lo, hi = bounds
            clipped = float(min(max(raw_value, lo), hi))
            self._last_normal[path] = clipped
            quality["replaced"].append(
                {
                    "path": path,
                    "reason": f"{reason}_clipped",
                    "raw": raw_value,
                    "used": clipped,
                }
            )
            return clipped
        quality["hidden"].append(
            {
                "path": path,
                "reason": reason,
                "raw": raw_value,
            }
        )
        return None
