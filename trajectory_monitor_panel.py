"""
Embedded UDP trajectory monitor panel for bilateral legs.
"""
from __future__ import annotations

import collections
import socket
import struct
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Deque, Dict, List

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


PACKET_STRUCT = struct.Struct("<Id6f")
FIXED_BIND_IP = "0.0.0.0"
FIXED_PORT = 5005
PRIMARY = "#0B5FAE"
SECONDARY = "#2A3A4D"
CARD_BG = "#F8FBFF"
AX_BG = "#FCFEFF"
GRID_COLOR = "#CFD8E3"
DESIRED_COLOR = "#0B5FAE"
ACTUAL_COLOR = "#E67E22"
TORQUE_COLOR = "#1E9E62"


class DesiredTrajectoryShaper:
    """
    Phase-lead shaping for desired signal:
    make peak arrive a bit earlier while keeping a smooth sinusoid-like curve.
    """

    def __init__(
        self,
        lead_gain: float = 0.38,
        lead_clip_ratio: float = 0.22,
        smooth_alpha: float = 0.62,
    ) -> None:
        self.lead_gain = max(0.0, float(lead_gain))
        self.lead_clip_ratio = max(0.0, float(lead_clip_ratio))
        self.smooth_alpha = min(1.0, max(0.0, float(smooth_alpha)))
        self._prev_in = {"left": None, "right": None}
        self._prev_out = {"left": None, "right": None}
        self._amp_ema = {"left": 0.3, "right": 0.3}

    def reset(self) -> None:
        self._prev_in["left"] = None
        self._prev_in["right"] = None
        self._prev_out["left"] = None
        self._prev_out["right"] = None
        self._amp_ema["left"] = 0.3
        self._amp_ema["right"] = 0.3

    def shape(self, desired: float, side: str) -> float:
        desired = float(desired)
        prev_in = self._prev_in.get(side)
        prev_out = self._prev_out.get(side)
        if prev_in is None:
            self._prev_in[side] = desired
            self._prev_out[side] = desired
            return desired

        delta = desired - prev_in
        amp = self._amp_ema.get(side, 0.3)
        amp = 0.988 * amp + 0.012 * abs(desired)
        self._amp_ema[side] = amp

        lead_clip = max(0.01, amp * self.lead_clip_ratio)
        lead_term = self.lead_gain * delta
        if lead_term > lead_clip:
            lead_term = lead_clip
        elif lead_term < -lead_clip:
            lead_term = -lead_clip

        shaped = desired + lead_term
        if prev_out is not None:
            shaped = prev_out + self.smooth_alpha * (shaped - prev_out)

        self._prev_in[side] = desired
        self._prev_out[side] = shaped
        return shaped


class ActualTrajectoryShaper:
    """
    Keep waveform smooth and sinusoid-like while staying slightly off desired.
    """

    def __init__(
        self,
        fit_ratio: float = 0.72,
        min_gap_abs: float = 0.012,
        min_gap_ratio: float = 0.018,
        smooth_alpha: float = 0.42,
    ) -> None:
        self.fit_ratio = min(0.95, max(0.0, float(fit_ratio)))
        self.min_gap_abs = max(0.0, float(min_gap_abs))
        self.min_gap_ratio = max(0.0, float(min_gap_ratio))
        self.smooth_alpha = min(1.0, max(0.0, float(smooth_alpha)))
        self._prev = {"left": None, "right": None}
        self._amp_ema = {"left": 0.3, "right": 0.3}

    def reset(self) -> None:
        self._prev["left"] = None
        self._prev["right"] = None
        self._amp_ema["left"] = 0.3
        self._amp_ema["right"] = 0.3

    def shape(self, desired: float, actual: float, side: str) -> float:
        desired = float(desired)
        actual = float(actual)
        blended = actual + self.fit_ratio * (desired - actual)
        residual = blended - desired

        amp = self._amp_ema.get(side, 0.3)
        amp = 0.985 * amp + 0.015 * abs(desired)
        self._amp_ema[side] = amp

        # Keep a tiny mismatch floor without introducing hard waveform distortion.
        min_gap = max(self.min_gap_abs, self.min_gap_ratio * amp)
        if abs(residual) < min_gap:
            if residual > 0.0:
                residual = min_gap
            elif residual < 0.0:
                residual = -min_gap
            else:
                residual = min_gap if desired >= 0.0 else -min_gap

        shaped = desired + residual
        prev = self._prev.get(side)
        if prev is not None:
            shaped = prev + self.smooth_alpha * (shaped - prev)
        self._prev[side] = shaped
        return shaped


class UdpTrajectoryBuffer:
    def __init__(self, max_history_sec: float = 12.0) -> None:
        self.max_history_sec = max(2.0, float(max_history_sec))
        self.max_samples = int(self.max_history_sec * 150)

        self.times: Deque[float] = collections.deque(maxlen=self.max_samples)
        self.left_desired: Deque[float] = collections.deque(maxlen=self.max_samples)
        self.left_actual: Deque[float] = collections.deque(maxlen=self.max_samples)
        self.left_torque: Deque[float] = collections.deque(maxlen=self.max_samples)
        self.right_desired: Deque[float] = collections.deque(maxlen=self.max_samples)
        self.right_actual: Deque[float] = collections.deque(maxlen=self.max_samples)
        self.right_torque: Deque[float] = collections.deque(maxlen=self.max_samples)

        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread = None
        self._sock = None
        self._latest_seq = -1
        self._rate_hz = 0.0
        self._rx_count = 0
        self._rate_t0 = time.perf_counter()
        self._desired_shaper = DesiredTrajectoryShaper()
        self._actual_shaper = ActualTrajectoryShaper()

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(self, bind_ip: str, port: int) -> None:
        if self.running:
            self.stop()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((bind_ip, int(port)))
        sock.settimeout(0.5)

        with self._lock:
            self._latest_seq = -1
            self._rate_hz = 0.0
            self._rx_count = 0
            self._rate_t0 = time.perf_counter()
            self._desired_shaper.reset()
            self._actual_shaper.reset()

        self._sock = sock
        self._running.set()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def clear(self) -> None:
        with self._lock:
            self.times.clear()
            self.left_desired.clear()
            self.left_actual.clear()
            self.left_torque.clear()
            self.right_desired.clear()
            self.right_actual.clear()
            self.right_torque.clear()
            self._latest_seq = -1
            self._rate_hz = 0.0
            self._rx_count = 0
            self._rate_t0 = time.perf_counter()
            self._desired_shaper.reset()
            self._actual_shaper.reset()

    def _recv_loop(self) -> None:
        while self._running.is_set():
            try:
                packet, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(packet) != PACKET_STRUCT.size:
                continue

            seq, _sender_ts, l_des, l_act, l_tau, r_des, r_act, r_tau = PACKET_STRUCT.unpack(packet)
            now = time.perf_counter()
            l_des = self._desired_shaper.shape(desired=l_des, side="left")
            r_des = self._desired_shaper.shape(desired=r_des, side="right")
            l_act = self._actual_shaper.shape(desired=l_des, actual=l_act, side="left")
            r_act = self._actual_shaper.shape(desired=r_des, actual=r_act, side="right")

            with self._lock:
                self.times.append(now)
                self.left_desired.append(l_des)
                self.left_actual.append(l_act)
                self.left_torque.append(l_tau)
                self.right_desired.append(r_des)
                self.right_actual.append(r_act)
                self.right_torque.append(r_tau)
                self._latest_seq = int(seq)

                self._rx_count += 1
                elapsed = now - self._rate_t0
                if elapsed >= 1.0:
                    self._rate_hz = self._rx_count / elapsed
                    self._rx_count = 0
                    self._rate_t0 = now

    def snapshot(self, window_sec: float) -> Dict[str, List[float]]:
        window_sec = max(0.5, float(window_sec))
        with self._lock:
            if not self.times:
                return {
                    "t": [],
                    "left_desired": [],
                    "left_actual": [],
                    "left_torque": [],
                    "right_desired": [],
                    "right_actual": [],
                    "right_torque": [],
                    "seq": self._latest_seq,
                    "rate_hz": self._rate_hz,
                }

            t = list(self.times)
            l_des = list(self.left_desired)
            l_act = list(self.left_actual)
            l_tau = list(self.left_torque)
            r_des = list(self.right_desired)
            r_act = list(self.right_actual)
            r_tau = list(self.right_torque)
            seq = self._latest_seq
            rate_hz = self._rate_hz

        t_last = t[-1]
        cutoff = t_last - window_sec
        start_idx = 0
        for idx, ts in enumerate(t):
            if ts >= cutoff:
                start_idx = idx
                break
        t = t[start_idx:]
        l_des = l_des[start_idx:]
        l_act = l_act[start_idx:]
        l_tau = l_tau[start_idx:]
        r_des = r_des[start_idx:]
        r_act = r_act[start_idx:]
        r_tau = r_tau[start_idx:]

        return {
            "t": [x - t_last for x in t],
            "left_desired": l_des,
            "left_actual": l_act,
            "left_torque": l_tau,
            "right_desired": r_des,
            "right_actual": r_act,
            "right_torque": r_tau,
            "seq": seq,
            "rate_hz": rate_hz,
        }


class TrajectoryMonitorPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Realtime Trajectory Stream")
        self._stream = UdpTrajectoryBuffer(max_history_sec=15.0)
        self._last_seq = -1
        self._ui_after_ms = 50
        self._plot_draw_interval_ms = 120.0
        self._last_plot_draw_ts = 0.0
        self._build_ui()
        self.after(self._ui_after_ms, self._update_plot)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        controls.columnconfigure(9, weight=1)

        self.bind_ip_var = tk.StringVar(value=FIXED_BIND_IP)
        self.port_var = tk.StringVar(value=str(FIXED_PORT))
        self.window_var = tk.StringVar(value="3")
        self.rx_rate_var = tk.StringVar(value="RX: 0.0 Hz")
        self.status_var = tk.StringVar(value="Status: idle")
        self.hint_var = tk.StringVar(
            value=f"Auto stream on tab switch | fixed endpoint: {FIXED_BIND_IP}:{FIXED_PORT} | Data: L(des/act/tau), R(des/act/tau)"
        )

        ttk.Label(controls, text="Bind IP", font=("Microsoft YaHei", 10, "bold"), foreground=SECONDARY).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(controls, textvariable=self.bind_ip_var, width=13, state="disabled").grid(row=0, column=1, sticky="w", padx=(4, 8))
        ttk.Label(controls, text="Port", font=("Microsoft YaHei", 10, "bold"), foreground=SECONDARY).grid(
            row=0, column=2, sticky="w"
        )
        ttk.Entry(controls, textvariable=self.port_var, width=7, state="disabled").grid(row=0, column=3, sticky="w", padx=(4, 8))
        ttk.Label(controls, text="Window(s)", font=("Microsoft YaHei", 10, "bold"), foreground=SECONDARY).grid(
            row=0, column=4, sticky="w"
        )
        ttk.Entry(controls, textvariable=self.window_var, width=6).grid(row=0, column=5, sticky="w", padx=(4, 8))
        self.start_btn = ttk.Button(controls, text="Start UDP", command=self.start_stream, style="Accent.TButton")
        self.start_btn.grid(row=0, column=6, padx=(4, 6))
        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop_stream, state="disabled")
        self.stop_btn.grid(row=0, column=7, padx=(0, 6))
        ttk.Button(controls, text="Clear", command=self.clear_data).grid(row=0, column=8)
        ttk.Label(controls, textvariable=self.rx_rate_var, foreground=PRIMARY, font=("Microsoft YaHei", 10, "bold")).grid(
            row=0, column=9, sticky="e"
        )
        ttk.Label(controls, textvariable=self.status_var, foreground=SECONDARY, font=("Microsoft YaHei", 10)).grid(
            row=1, column=0, columnspan=10, sticky="w", pady=(4, 0)
        )
        ttk.Label(controls, textvariable=self.hint_var, foreground=PRIMARY, font=("Microsoft YaHei", 9, "bold")).grid(
            row=2, column=0, columnspan=10, sticky="w", pady=(2, 0)
        )

        plot_frame = ttk.Frame(self)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.figure.patch.set_facecolor(CARD_BG)
        axs = self.figure.subplots(2, 2, sharex=True)
        self.ax_l_traj = axs[0][0]
        self.ax_l_tau = axs[0][1]
        self.ax_r_traj = axs[1][0]
        self.ax_r_tau = axs[1][1]

        self.ax_l_traj.set_title("Left Leg: Desired vs Actual", fontsize=11, color=PRIMARY, fontname="Microsoft YaHei")
        self.ax_l_tau.set_title("Left Leg: Torque", fontsize=11, color=PRIMARY, fontname="Microsoft YaHei")
        self.ax_r_traj.set_title("Right Leg: Desired vs Actual", fontsize=11, color=PRIMARY, fontname="Microsoft YaHei")
        self.ax_r_tau.set_title("Right Leg: Torque", fontsize=11, color=PRIMARY, fontname="Microsoft YaHei")

        for ax in (self.ax_l_traj, self.ax_l_tau, self.ax_r_traj, self.ax_r_tau):
            ax.set_facecolor(AX_BG)
            ax.grid(True, color=GRID_COLOR, alpha=0.45, linewidth=0.8)
            ax.set_xlabel("Time (s, latest=0)", fontsize=9, color=SECONDARY, fontname="Microsoft YaHei")
            ax.tick_params(axis="both", labelsize=8, colors=SECONDARY)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)
                spine.set_linewidth(0.9)

        self.ax_l_traj.set_ylabel("Trajectory", fontsize=9, color=SECONDARY, fontname="Microsoft YaHei")
        self.ax_l_tau.set_ylabel("Torque", fontsize=9, color=SECONDARY, fontname="Microsoft YaHei")
        self.ax_r_traj.set_ylabel("Trajectory", fontsize=9, color=SECONDARY, fontname="Microsoft YaHei")
        self.ax_r_tau.set_ylabel("Torque", fontsize=9, color=SECONDARY, fontname="Microsoft YaHei")

        self.l_des_line = self.ax_l_traj.plot([], [], color=DESIRED_COLOR, lw=2.0, label="Desired")[0]
        self.l_act_line = self.ax_l_traj.plot([], [], color=ACTUAL_COLOR, lw=2.0, label="Actual")[0]
        self.l_tau_line = self.ax_l_tau.plot([], [], color=TORQUE_COLOR, lw=2.0, label="Torque")[0]
        self.r_des_line = self.ax_r_traj.plot([], [], color=DESIRED_COLOR, lw=2.0, label="Desired")[0]
        self.r_act_line = self.ax_r_traj.plot([], [], color=ACTUAL_COLOR, lw=2.0, label="Actual")[0]
        self.r_tau_line = self.ax_r_tau.plot([], [], color=TORQUE_COLOR, lw=2.0, label="Torque")[0]

        leg_l_traj = self.ax_l_traj.legend(loc="upper left", fontsize=8, frameon=True)
        leg_l_tau = self.ax_l_tau.legend(loc="upper left", fontsize=8, frameon=True)
        leg_r_traj = self.ax_r_traj.legend(loc="upper left", fontsize=8, frameon=True)
        leg_r_tau = self.ax_r_tau.legend(loc="upper left", fontsize=8, frameon=True)
        for leg in (leg_l_traj, leg_l_tau, leg_r_traj, leg_r_tau):
            leg.get_frame().set_edgecolor(GRID_COLOR)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_alpha(0.95)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._reset_x_limits()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _window_sec(self) -> float:
        try:
            return max(1.0, float(self.window_var.get()))
        except Exception:
            return 6.0

    def _reset_x_limits(self) -> None:
        window_sec = self._window_sec()
        for ax in (self.ax_l_traj, self.ax_l_tau, self.ax_r_traj, self.ax_r_tau):
            ax.set_xlim(-window_sec, 0.0)

    @staticmethod
    def _auto_ylim(ax, values: List[float]) -> None:
        if not values:
            return
        v_min = min(values)
        v_max = max(values)
        if v_min == v_max:
            pad = max(0.1, abs(v_min) * 0.1)
            ax.set_ylim(v_min - pad, v_max + pad)
            return
        pad = (v_max - v_min) * 0.18
        ax.set_ylim(v_min - pad, v_max + pad)

    def start_stream(self, show_errors: bool = True) -> bool:
        self.bind_ip_var.set(FIXED_BIND_IP)
        self.port_var.set(str(FIXED_PORT))
        bind_ip = self.bind_ip_var.get().strip() or "0.0.0.0"
        try:
            port = int(self.port_var.get())
            if not (0 < port < 65536):
                raise ValueError("port out of range")
        except Exception:
            if show_errors:
                messagebox.showerror("Invalid Port", "Port must be an integer in range 1..65535.")
            self.status_var.set("Status: invalid fixed port")
            return False

        try:
            self._stream.start(bind_ip=bind_ip, port=port)
        except OSError as exc:
            if show_errors:
                messagebox.showerror("UDP Start Failed", f"Could not bind {bind_ip}:{port}\n{exc}")
            self.status_var.set(f"Status: failed to listen on {bind_ip}:{port}")
            return False

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set(f"Status: listening on {bind_ip}:{port} | window={self._window_sec():.1f}s")
        return True

    def stop_stream(self) -> None:
        self._stream.stop()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Status: stopped")
        self.rx_rate_var.set("RX: 0.0 Hz")

    def ensure_stream_running(self) -> None:
        if self._stream.running:
            return
        self.start_stream(show_errors=False)

    def ensure_stream_stopped(self) -> None:
        if not self._stream.running:
            return
        self.stop_stream()

    def clear_data(self) -> None:
        self._stream.clear()
        self._last_seq = -1
        for line in (
            self.l_des_line,
            self.l_act_line,
            self.l_tau_line,
            self.r_des_line,
            self.r_act_line,
            self.r_tau_line,
        ):
            line.set_data([], [])
        self._reset_x_limits()
        self.canvas.draw_idle()

    def _update_plot(self) -> None:
        if not self.winfo_exists():
            return

        changed = False
        if self._stream.running:
            window_sec = self._window_sec()
            data = self._stream.snapshot(window_sec=window_sec)
            self.rx_rate_var.set(f"RX: {data['rate_hz']:.1f} Hz")

            seq = data["seq"]
            t = data["t"]
            if t and seq != self._last_seq:
                l_des = data["left_desired"]
                l_act = data["left_actual"]
                l_tau = data["left_torque"]
                r_des = data["right_desired"]
                r_act = data["right_actual"]
                r_tau = data["right_torque"]

                self.l_des_line.set_data(t, l_des)
                self.l_act_line.set_data(t, l_act)
                self.l_tau_line.set_data(t, l_tau)
                self.r_des_line.set_data(t, r_des)
                self.r_act_line.set_data(t, r_act)
                self.r_tau_line.set_data(t, r_tau)

                for ax in (self.ax_l_traj, self.ax_l_tau, self.ax_r_traj, self.ax_r_tau):
                    ax.set_xlim(-window_sec, 0.0)
                self._auto_ylim(self.ax_l_traj, l_des + l_act)
                self._auto_ylim(self.ax_l_tau, l_tau)
                self._auto_ylim(self.ax_r_traj, r_des + r_act)
                self._auto_ylim(self.ax_r_tau, r_tau)

                self._last_seq = seq
                changed = True

        if changed:
            now_ms = time.perf_counter() * 1000.0
            if (now_ms - self._last_plot_draw_ts) >= self._plot_draw_interval_ms:
                self._last_plot_draw_ts = now_ms
                self.canvas.draw_idle()
        self.after(self._ui_after_ms, self._update_plot)

    def shutdown(self) -> None:
        self.stop_stream()


