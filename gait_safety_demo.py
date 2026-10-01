"""Small data-free example of the gait metric safety gate."""

from gait_metric_safety import GaitMetricSafetyFilter


def main() -> None:
    live = {
        "cadence_spm": 104.0,
        "step_l_s": 0.58,
        "step_r_s": 0.59,
        "stance_l_pct": 61.0,
        "stance_r_pct": 60.0,
        "double_support_pct": 21.0,
        "phase_l": "STANCE",
        "phase_r": "SWING",
    }
    report = {
        "rom": {"knee_l": 54.0, "knee_r": 56.0},
        "stability": {"trunk_sway_std_deg": 2.4},
    }
    safe_live, safe_report, quality = GaitMetricSafetyFilter().sanitize(live, report)
    print("sanitized live metrics:", safe_live)
    print("sanitized report:", safe_report)
    print("quality flags:", quality["total_flags"])


if __name__ == "__main__":
    main()
