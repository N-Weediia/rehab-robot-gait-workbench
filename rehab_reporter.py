import datetime
import json
import os
import webbrowser


class RehabReportExporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _fmt(v, nd=3, suffix=""):
        try:
            if v is None:
                return "--"
            return f"{float(v):.{nd}f}{suffix}"
        except Exception:
            return "--"

    @staticmethod
    def _ratio_score(a, b):
        try:
            a = float(a)
            b = float(b)
            if a <= 0.0 or b <= 0.0:
                return None
            return min(a, b) / max(a, b) * 100.0
        except Exception:
            return None

    @staticmethod
    def _first_finite(*values):
        for v in values:
            try:
                if v is None:
                    continue
                fv = float(v)
                if fv == fv and fv not in (float("inf"), float("-inf")):
                    return fv
            except Exception:
                continue
        return None

    def export_report(self, session_info, live_metrics, periodic_report):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"rehab_report_{ts}"
        json_path = os.path.join(self.output_dir, f"{stem}.json")
        html_zh_path = os.path.join(self.output_dir, f"{stem}_zh.html")
        html_en_path = os.path.join(self.output_dir, f"{stem}_en.html")

        payload = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "session": session_info or {},
            "live_metrics": live_metrics or {},
            "periodic_rehab": periodic_report or {},
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        html_zh = self._build_html(payload, lang="zh")
        with open(html_zh_path, "w", encoding="utf-8") as f:
            f.write(html_zh)

        html_en = self._build_html(payload, lang="en")
        with open(html_en_path, "w", encoding="utf-8") as f:
            f.write(html_en)

        return {
            "json_path": json_path,
            "html_zh_path": html_zh_path,
            "html_en_path": html_en_path,
        }

    def _build_html(self, payload, lang="en"):
        sess = payload.get("session", {})
        live = payload.get("live_metrics", {})
        report = payload.get("periodic_rehab", {})
        rom = report.get("rom", {}) if isinstance(report, dict) else {}
        stab = report.get("stability", {}) if isinstance(report, dict) else {}
        sym = report.get("symmetry", {}) if isinstance(report, dict) else {}
        functional = report.get("functional", {}) if isinstance(report, dict) else {}

        gait_stability_score = self._first_finite(
            functional.get("gait_stability_score_pct") if isinstance(functional, dict) else None,
            live.get("gait_stability_score_pct"),
            self._ratio_score(live.get("stance_l_pct"), live.get("stance_r_pct")),
        )
        gait_rhythm_score = self._first_finite(
            functional.get("gait_rhythm_score_pct") if isinstance(functional, dict) else None,
            live.get("gait_rhythm_score_pct"),
            self._ratio_score(live.get("step_l_s"), live.get("step_r_s")),
        )
        step_sym_si = self._first_finite(
            sym.get("step_time_si_pct") if isinstance(sym, dict) else None,
            live.get("sym_step_pct"),
        )

        is_zh = str(lang).lower().startswith("zh")
        t = {
            "page_title": "康复步态评估报告" if is_zh else "Rehab Gait Assessment Report",
            "main_title": "康复步态评估报告" if is_zh else "Rehabilitation Gait Assessment Report",
            "generated_at": "生成时间" if is_zh else "Generated",
            "session": "基本信息" if is_zh else "Basic Information",
            "realtime": "步态时空参数" if is_zh else "Spatiotemporal Gait Parameters",
            "periodic": "关节运动参数" if is_zh else "Joint Kinematic Parameters",
            "stability": "步态功能指标" if is_zh else "Gait Functional Indicators",
            "session_id": "测试者序号" if is_zh else "Session ID",
            "start": "开始时间" if is_zh else "Start",
            "end": "结束时间" if is_zh else "End",
            "duration": "持续时长" if is_zh else "Duration",
            "report_window": "报告窗口" if is_zh else "Report Window",
            "refresh_interval": "刷新间隔" if is_zh else "Refresh Interval",
            "step_cadence_speed": "步长/步频/步速" if is_zh else "Step Length/Cadence/Speed",
            "phase_lr": "支撑相/摆动相" if is_zh else "Stance/Swing Phase (Phase L/R)",
            "double_support": "双支撑相时间" if is_zh else "Double Support Time",
            "toe_lr": "足趾垂直位移" if is_zh else "Toe Vertical Displacement (Toe Clearance L/R)",
            "rom_hip_lr": "髋关节L/R" if is_zh else "Hip Joint L/R",
            "rom_knee_lr": "膝关节L/R" if is_zh else "Knee Joint L/R",
            "rom_ankle_lr": "踝关节L/R" if is_zh else "Ankle Joint L/R",
            "trunk_std": "躯干摆动标准差" if is_zh else "Trunk Sway Std",
            "gait_stability": "步态稳定性" if is_zh else "Gait Stability",
            "gait_symmetry": "步态对称性" if is_zh else "Gait Symmetry (Step Symmetry SI)",
            "gait_rhythm": "步态节律性" if is_zh else "Gait Rhythmicity",
            "unit_s": " s",
            "unit_spm": " spm",
            "unit_m": " m",
            "unit_deg": " deg",
        }

        css = """
        :root { color-scheme: light; }
        body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 24px; color: #1f2a37; background: linear-gradient(180deg, #e8eff7 0%, #f7fafc 100%); }
        .report { background: #ffffff; border: 1px solid #d5e1ef; border-radius: 14px; padding: 24px; box-shadow: 0 10px 24px rgba(24, 58, 95, 0.08); }
        .title { font-size: 36px; font-weight: 800; letter-spacing: 0.2px; color: #0b4d82; margin-bottom: 8px; }
        .subtitle { color: #4f5e70; margin-bottom: 20px; font-size: 16px; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
        .card { border: 1px solid #dbe5f3; border-radius: 10px; padding: 14px 16px; background: linear-gradient(180deg, #ffffff 0%, #fdfefe 100%); }
        .card h3 { margin: 0 0 8px 0; color: #0f4c81; font-size: 20px; }
        .kv { display: grid; grid-template-columns: 1fr auto; gap: 6px 12px; font-size: 14px; }
        .kv div:nth-child(odd) { color: #425466; }
        .kv div:nth-child(even) { color: #111827; font-weight: 600; }
        @media print { body { margin: 0; background: white; } .report { border: none; border-radius: 0; } }
        """

        html = f"""
<!doctype html>
<html lang='{"zh-CN" if is_zh else "en"}'>
<head>
  <meta charset='utf-8'>
  <title>{t["page_title"]}</title>
  <style>{css}</style>
</head>
<body>
  <section class='report'>
    <div class='title'>{t["main_title"]}</div>
    <div class='subtitle'>{t["generated_at"]}: {payload.get('generated_at', '--')}</div>

    <div class='row'>
      <div class='card'>
        <h3>{t["session"]}</h3>
        <div class='kv'>
          <div>{t["session_id"]}</div><div>{sess.get('session_id', '--')}</div>
          <div>{t["start"]}</div><div>{sess.get('start_time', '--')}</div>
          <div>{t["end"]}</div><div>{sess.get('end_time', '--')}</div>
          <div>{t["duration"]}</div><div>{self._fmt(sess.get('duration_s'), 1, t['unit_s'])}</div>
          <div>{t["report_window"]}</div><div>{self._fmt(live.get('report_window_s') or report.get('window_s'), 0, t['unit_s'])}</div>
          <div>{t["refresh_interval"]}</div><div>{self._fmt(live.get('report_interval_s') or report.get('report_interval_s'), 0, t['unit_s'])}</div>
        </div>
      </div>
      <div class='card'>
        <h3>{t["realtime"]}</h3>
        <div class='kv'>
          <div>{t["step_cadence_speed"]}</div><div>{self._fmt(live.get('step_length_speed_m') if live.get('step_length_speed_m') is not None else live.get('step_length_sur_m'), 3, t['unit_m'])} / {self._fmt(live.get('cadence_spm'), 1, t['unit_spm'])} / {self._fmt(live.get('gait_speed_mps') if live.get('gait_speed_mps') is not None else live.get('com_speed_mps'), 3, t['unit_m'] + '/s')}</div>
          <div>{t["phase_lr"]}</div><div>{live.get('phase_l', '--')} / {live.get('phase_r', '--')}</div>
          <div>{t["double_support"]}</div><div>{self._fmt(live.get('double_support_pct'), 1, '%')}</div>
          <div>{t["toe_lr"]}</div><div>{self._fmt(live.get('toe_clear_l_m'), 3, t['unit_m'])} / {self._fmt(live.get('toe_clear_r_m'), 3, t['unit_m'])}</div>
        </div>
      </div>
    </div>

    <div class='row'>
      <div class='card'>
        <h3>{t["periodic"]}</h3>
        <div class='kv'>
          <div>{t["rom_hip_lr"]}</div><div>{self._fmt(rom.get('hip_l'), 1, t['unit_deg'])} / {self._fmt(rom.get('hip_r'), 1, t['unit_deg'])}</div>
          <div>{t["rom_knee_lr"]}</div><div>{self._fmt(rom.get('knee_l'), 1, t['unit_deg'])} / {self._fmt(rom.get('knee_r'), 1, t['unit_deg'])}</div>
          <div>{t["rom_ankle_lr"]}</div><div>{self._fmt(rom.get('ankle_l'), 1, t['unit_deg'])} / {self._fmt(rom.get('ankle_r'), 1, t['unit_deg'])}</div>
          <div>{t["trunk_std"]}</div><div>{self._fmt(stab.get('trunk_sway_std_deg'), 2, t['unit_deg'])}</div>
        </div>
      </div>
      <div class='card'>
        <h3>{t["stability"]}</h3>
        <div class='kv'>
          <div>{t["gait_stability"]}</div><div>{self._fmt(gait_stability_score, 1, '%')}</div>
          <div>{t["gait_symmetry"]}</div><div>{self._fmt(step_sym_si, 1, '%')}</div>
          <div>{t["gait_rhythm"]}</div><div>{self._fmt(gait_rhythm_score, 1, '%')}</div>
        </div>
      </div>
    </div>

  </section>
</body>
</html>
"""
        return html

    def print_file(self, file_path):
        if os.name != "nt":
            return False, "Direct print is only supported on Windows."
        try:
            os.startfile(file_path, "print")
            return True, "Print command submitted to system printer queue."
        except Exception as e:
            if getattr(e, "winerror", None) == 1155:
                try:
                    os.startfile(file_path)
                except Exception:
                    try:
                        webbrowser.open_new_tab(f"file:///{os.path.abspath(file_path).replace(os.sep, '/')}")
                    except Exception:
                        return False, f"Print failed: {e}"
                return True, "系统未配置 HTML 直接打印，已打开报告文件，请在浏览器中按 Ctrl+P 打印。"
            return False, f"Print failed: {e}"

    def open_file(self, file_path):
        try:
            if os.name == "nt":
                os.startfile(file_path)
                return True
        except Exception:
            return False
        return False

