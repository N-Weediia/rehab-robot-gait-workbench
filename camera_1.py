import cv2
import threading
import numpy as np
import time
import datetime
import os
from pyorbbecsdk import Context, Config, OBError, OBSensorType, Pipeline, OBFormat


ESC_KEY = 27
APP_DIR = os.path.dirname(os.path.abspath(__file__))


class CameraThread(threading.Thread):
    def __init__(self, device, device_idx, image_list, lock):
        super().__init__()
        self.device = device
        self.device_idx = device_idx
        self.image_list = image_list
        self.lock = lock
        self.running = True
        self.last_valid_frame = None
        self.actual_width = None
        self.actual_height = None
        self.actual_fps = None
        self.frame_seq = 0

    def run(self):
        config = Config()
        pipeline = None
        try:
            pipeline = Pipeline(self.device)
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = self.select_best_profile(profile_list)

            if not color_profile:
                raise RuntimeError("No valid color stream profile found.")

            print(f"\nCamera {self.device_idx} profile: {self.actual_width}x{self.actual_height} @ {self.actual_fps}fps")

            config.enable_stream(color_profile)
            pipeline.start(config)
            print(f"Camera {self.device_idx} started.")

            while self.running:
                try:
                    frames = pipeline.wait_for_frames(500)
                    if not frames:
                        continue
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    image = self.process_frame(color_frame)
                    with self.lock:
                        if image is not None:
                            self.image_list[self.device_idx] = image.copy()
                            self.last_valid_frame = image.copy()
                            self.frame_seq += 1
                except Exception as e:
                    print(f"Camera {self.device_idx} read error: {str(e)}")
                    break
        except Exception as e:
            print(f"\nCamera {self.device_idx} init failed: {str(e)}")
        finally:
            if pipeline:
                pipeline.stop()
            print(f"Camera {self.device_idx} stopped.")

    def select_best_profile(self, profile_list):
        target_config = (1280, 720, 30)
        try:
            profile = profile_list.get_video_stream_profile(
                target_config[0], target_config[1], OBFormat.MJPG, target_config[2]
            )
            self.actual_width = target_config[0]
            self.actual_height = target_config[1]
            self.actual_fps = target_config[2]
            return profile
        except (OBError, AttributeError):
            print(f"Camera {self.device_idx}: 720p30 MJPG not available, fallback to default profile.")
            default_profile = profile_list.get_default_video_stream_profile()
            self.actual_width = default_profile.get_width()
            self.actual_height = default_profile.get_height()
            self.actual_fps = default_profile.get_fps()
            return default_profile

    def process_frame(self, frame):
        try:
            data = np.frombuffer(frame.get_data(), dtype=np.uint8)
            decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if decoded is not None and not decoded.flags.c_contiguous:
                return np.ascontiguousarray(decoded)
            return decoded
        except Exception:
            error_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(error_img, "Decode Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return error_img


def camera_main(control_state=None, exit_callback=None, start_recording_callback=None, stop_recording_callback=None):
    context = Context()
    device_list = context.query_devices()

    if device_list.get_count() < 2:
        print(f"Error: need 2 cameras, found {device_list.get_count()}.")
        if exit_callback:
            exit_callback()
        return

    devices = [device_list.get_device_by_index(i) for i in range(2)]
    image_list = [None] * 2
    lock = threading.Lock()
    threads = [CameraThread(devices[i], i, image_list, lock) for i in range(2)]
    for t in threads:
        t.start()

    is_recording = False
    video_writers = [None] * 2
    recording_start_time = 0
    current_recording_dir = None
    last_written_seq = [-1] * 2

    try:
        cv2.namedWindow("Binocular Camera System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Binocular Camera System", 1280, 360)
        print("\n=== Binocular Camera Window Started ===")
        print("Recording is controlled from GUI.")
        print("Press S to save synchronized calibration images.")
        print("Press Q or ESC to quit camera window.")
        print("=======================================\n")

        while True:
            with lock:
                current_frames = [
                    image_list[i].copy()
                    if image_list[i] is not None
                    else np.zeros(
                        (
                            int(threads[i].actual_height or 720),
                            int(threads[i].actual_width or 1280),
                            3,
                        ),
                        dtype=np.uint8,
                    )
                    for i in range(2)
                ]
                current_seqs = [t.frame_seq for t in threads]

            if control_state:
                command = control_state.get("recording_command")
                if command == "START" and not is_recording:
                    try:
                        recording_timestamp = control_state.get("recording_timestamp")
                        current_recording_dir = control_state.get("recording_dir")
                        if not recording_timestamp:
                            recording_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        if not current_recording_dir:
                            current_recording_dir = os.path.join(APP_DIR, "recording", recording_timestamp)
                        os.makedirs(current_recording_dir, exist_ok=True)

                        for i in range(2):
                            filename = os.path.join(current_recording_dir, f"cam_{i}.mp4")
                            frame_h = int(threads[i].actual_height or current_frames[i].shape[0])
                            frame_w = int(threads[i].actual_width or current_frames[i].shape[1])
                            frame_fps = int(threads[i].actual_fps or 30)
                            video_writers[i] = cv2.VideoWriter(
                                filename, cv2.VideoWriter_fourcc(*"avc1"), frame_fps, (frame_w, frame_h)
                            )
                        is_recording = True
                        recording_start_time = time.time()
                        last_written_seq = [t.frame_seq for t in threads]
                        if start_recording_callback:
                            start_recording_callback()
                        print(f"Recording started -> {current_recording_dir}")
                    except Exception as e:
                        print(f"Failed to start recording: {e}")
                        is_recording = False
                    finally:
                        control_state["recording_command"] = None

                elif command == "STOP" and is_recording:
                    for i in range(2):
                        if video_writers[i]:
                            video_writers[i].release()
                    video_writers = [None] * 2
                    is_recording = False
                    if stop_recording_callback:
                        stop_recording_callback()
                    print(f"Recording stopped -> {current_recording_dir}")
                    control_state["recording_command"] = None

            display_frames = [cv2.resize(f, (640, 360)) for f in current_frames]
            combined = cv2.hconcat(display_frames)

            if is_recording:
                elapsed = time.time() - recording_start_time
                time_str = f"Recording {datetime.timedelta(seconds=int(elapsed))}"
                cv2.putText(combined, time_str, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                for i in range(2):
                    if video_writers[i] and current_seqs[i] != last_written_seq[i]:
                        video_writers[i].write(current_frames[i])
                        last_written_seq[i] = current_seqs[i]

            cv2.imshow("Binocular Camera System", combined)
            key = cv2.waitKey(1)

            if key in (ord("q"), ord("Q"), ESC_KEY):
                break
            elif key in (ord("s"), ord("S")):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                base_sync_dir = os.path.join(APP_DIR, "calib_images", "sync")
                os.makedirs(base_sync_dir, exist_ok=True)
                for i in range(2):
                    if current_frames[i] is not None:
                        cam_sync_dir = os.path.join(base_sync_dir, f"cam{i}")
                        os.makedirs(cam_sync_dir, exist_ok=True)
                        filename = os.path.join(cam_sync_dir, f"sync_{timestamp}.png")
                        cv2.imwrite(filename, current_frames[i])
                print(f"Synchronized stereo images saved ({timestamp})")

    finally:
        if is_recording and stop_recording_callback:
            stop_recording_callback()
            print("Stopped recording automatically on exit.")
        for thread in threads:
            thread.running = False
            thread.join(timeout=1)
        for writer in video_writers:
            if writer:
                writer.release()
        cv2.destroyAllWindows()
        if exit_callback:
            exit_callback()


if __name__ == "__main__":
    camera_main()
