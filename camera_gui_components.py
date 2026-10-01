"""
GUI components for calibration panel and log display.
"""
import tkinter as tk
from tkinter import ttk
import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk


class CalibrationPanel(ttk.LabelFrame):
    """Calibration panel."""
    def __init__(self, parent, app):
        super().__init__(parent, text="Camera Calibration", style="PanelCard.TLabelframe")
        self.app = app
        self.setup_ui()

    def setup_ui(self):
        param_frame = ttk.Frame(self)
        param_frame.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(param_frame, text="Square Size").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.square_size_input = ttk.Spinbox(param_frame, from_=1.0, to=100.0, increment=1.0, width=8)
        self.square_size_input.set(35.0)  # mm
        self.square_size_input.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(param_frame, text="mm").grid(row=0, column=2, sticky="w", padx=5, pady=2)

        ttk.Label(param_frame, text="Inner Corners Width").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.board_width_input = ttk.Spinbox(param_frame, from_=3, to=20, increment=1, width=8)
        self.board_width_input.set(11)
        self.board_width_input.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(param_frame, text="Inner Corners Height").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.board_height_input = ttk.Spinbox(param_frame, from_=3, to=20, increment=1, width=8)
        self.board_height_input.set(8)
        self.board_height_input.grid(row=2, column=1, padx=5, pady=2)

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=4, pady=4)

        self.calibrate_button = ttk.Button(
            button_frame,
            text="Start Stereo Calibration",
            command=self.app.start_calibration
        )
        self.calibrate_button.pack(side=tk.LEFT, padx=5)

        self.delete_images_button = ttk.Button(
            button_frame,
            text="Delete Calibration Images",
            command=self.app.delete_calibration_images
        )
        self.delete_images_button.pack(side=tk.LEFT, padx=5)

        self.calibration_status = ttk.Label(self, text="Waiting to start calibration...", foreground="#6B7280")
        self.calibration_status.pack(fill=tk.X, padx=4, pady=4)


class LogDisplay(ttk.LabelFrame):
    """Log display widget."""
    def __init__(self, parent):
        super().__init__(parent, text="System Logs", style="PanelCard.TLabelframe")
        self.setup_ui()

    def setup_ui(self):
        text_frame = ttk.Frame(self)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.log_text = tk.Text(
            text_frame,
            height=8,
            width=60,
            state="disabled",
            font=("Consolas", 10),
            bg="#F8FAFD",
            fg="#213447",
            relief="flat",
            padx=6,
            pady=6,
        )
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=4, pady=4)

        self.clear_log_button = ttk.Button(button_frame, text="Clear Logs", command=self.clear_log, style="Secondary.TButton")
        self.clear_log_button.pack(side=tk.LEFT)

    def add_log(self, message):
        def update_ui():
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, formatted_message)
            self.log_text.see(tk.END)
            self.log_text.config(state="disabled")

        if hasattr(self, "log_text") and self.log_text.master.winfo_exists():
            self.log_text.master.after(0, update_ui)

    def clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")


class StereoPreviewPanel(ttk.LabelFrame):
    """Stacked 2D recognition previews for left/right camera."""

    def __init__(self, parent):
        super().__init__(parent, text="2D Recognition Preview", style="PanelCard.TLabelframe")
        self._left_photo = None
        self._right_photo = None
        self._last_left_bgr = None
        self._last_right_bgr = None
        self._paused = False
        self._pause_msg = "Preview paused on Trajectory tab (reduced load)"
        # Match app card background (near #F8FBFF) for 16:9 letterbox padding.
        self._letterbox_bg_rgb = (248, 251, 255)
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(3, weight=1)

        ttk.Label(self, text="Left Camera", font=("Microsoft YaHei", 10, "bold")).grid(
            row=0, column=0, sticky="w", padx=6, pady=(6, 2)
        )
        self.left_label = ttk.Label(self, anchor="center")
        self.left_label.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 8))

        ttk.Label(self, text="Right Camera", font=("Microsoft YaHei", 10, "bold")).grid(
            row=2, column=0, sticky="w", padx=6, pady=(0, 2)
        )
        self.right_label = ttk.Label(self, anchor="center")
        self.right_label.grid(row=3, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.pause_label = ttk.Label(self, text="", foreground="#1f5d99", font=("Microsoft YaHei", 9, "bold"))
        self.pause_label.grid(row=4, column=0, sticky="w", padx=6, pady=(0, 6))

    def clear(self):
        self._left_photo = None
        self._right_photo = None
        self._last_left_bgr = None
        self._last_right_bgr = None
        self._paused = False
        self.left_label.configure(image="")
        self.right_label.configure(image="")
        self.pause_label.configure(text="")

    def update_frames(self, left_bgr, right_bgr):
        if left_bgr is not None:
            self._last_left_bgr = left_bgr.copy()
        if right_bgr is not None:
            self._last_right_bgr = right_bgr.copy()
        if self._paused:
            return
        if left_bgr is not None:
            self._left_photo = self._to_photo(left_bgr)
            if self._left_photo is not None:
                self.left_label.configure(image=self._left_photo)
        if right_bgr is not None:
            self._right_photo = self._to_photo(right_bgr)
            if self._right_photo is not None:
                self.right_label.configure(image=self._right_photo)

    def set_paused(self, paused, message=None):
        self._paused = bool(paused)
        if message is not None:
            self._pause_msg = str(message)
        if self._paused:
            self.pause_label.configure(text=self._pause_msg)
            self._render_paused_snapshot()
        else:
            self.pause_label.configure(text="")

    def _render_paused_snapshot(self):
        if self._last_left_bgr is not None:
            stamped_left = self._stamp_paused(self._last_left_bgr)
            self._left_photo = self._to_photo(stamped_left)
            if self._left_photo is not None:
                self.left_label.configure(image=self._left_photo)
        if self._last_right_bgr is not None:
            stamped_right = self._stamp_paused(self._last_right_bgr)
            self._right_photo = self._to_photo(stamped_right)
            if self._right_photo is not None:
                self.right_label.configure(image=self._right_photo)

    @staticmethod
    def _stamp_paused(frame_bgr):
        draw = frame_bgr.copy()
        if draw.ndim != 3 or draw.shape[0] <= 0 or draw.shape[1] <= 0:
            return draw
        h, w = draw.shape[:2]
        band_h = max(30, int(h * 0.11))
        y0 = max(0, h - band_h)
        overlay = draw.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), (35, 35, 35), -1)
        alpha = 0.45
        draw = cv2.addWeighted(overlay, alpha, draw, 1.0 - alpha, 0.0)
        cv2.putText(
            draw,
            "PAUSED",
            (12, h - max(10, int(band_h * 0.3))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        return draw

    def _to_photo(self, frame_bgr):
        if frame_bgr is None:
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        slot_w = max(180, int(self.winfo_width()) - 20)
        slot_h = max(120, int((self.winfo_height() - 60) * 0.5))
        # Enforce strict 16:9 preview area and letterbox source frames (no crop).
        target_w = min(slot_w, int(slot_h * 16 / 9))
        target_h = int(target_w * 9 / 16)
        if target_h > slot_h:
            target_h = slot_h
            target_w = int(target_h * 16 / 9)
        target_w = max(1, target_w)
        target_h = max(1, target_h)

        scale = min(target_w / max(float(w), 1.0), target_h / max(float(h), 1.0))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((target_h, target_w, 3), self._letterbox_bg_rgb, dtype=np.uint8)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return ImageTk.PhotoImage(Image.fromarray(canvas))


