"""Passive frame sources: a recorded video file, or a read-only screen capture.

Neither source ever sends input anywhere. VideoSource just decodes frames with
OpenCV; ScreenSource just grabs pixels with mss. Nothing here touches the
mouse, keyboard, or any process other than reading its own screen buffer.
"""
import time

import cv2
import numpy as np


class VideoSource:
    """Yields frames from a recorded .mov/.mp4 file, read-only."""

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._index = -1

    def frames(self):
        """Yields (frame_index, timestamp_seconds, frame_bgr)."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self._index += 1
            timestamp = self._index / self.fps
            yield self._index, timestamp, frame

    def release(self):
        self.cap.release()


class ScreenSource:
    """Yields frames captured from a monitor, read-only. Requires `mss`."""

    def __init__(self, monitor_index=1, target_fps=30):
        import mss  # imported lazily so video-only usage never needs it installed active
        mss_cls = getattr(mss, "MSS", mss.mss)
        self.sct = mss_cls()
        monitors = self.sct.monitors
        if monitor_index >= len(monitors):
            monitor_index = 1
        self.monitor = monitors[monitor_index]
        self.target_fps = target_fps
        self._index = -1
        self._start_time = None

    def frames(self):
        self._start_time = time.time()
        min_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        while True:
            loop_start = time.time()
            shot = self.sct.grab(self.monitor)
            frame = np.array(shot)[:, :, :3]  # BGRA -> BGR
            frame = np.ascontiguousarray(frame)
            self._index += 1
            timestamp = loop_start - self._start_time
            yield self._index, timestamp, frame
            elapsed = time.time() - loop_start
            sleep_time = min_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def release(self):
        self.sct.close()
