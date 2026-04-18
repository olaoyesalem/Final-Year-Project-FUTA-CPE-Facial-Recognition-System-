"""Shared mutable state between the Flask server and the main recognition loop."""
import threading
from typing import Optional

import cv2
import numpy as np


class FrameBuffer:
    """Thread-safe single-slot frame buffer for MJPEG stream and recognition."""

    def __init__(self):
        self._raw: Optional[np.ndarray] = None
        self._jpeg: Optional[bytes] = None
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        """Store a raw frame and its JPEG encoding."""
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        with self._lock:
            self._raw = frame
            self._jpeg = buf.tobytes()

    def put_annotated(self, frame: np.ndarray) -> None:
        """Update the JPEG only — used by enrollment on_frame to show overlay."""
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        with self._lock:
            self._jpeg = buf.tobytes()

    def get_raw(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._raw

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg


class AppState:
    """
    State shared between Flask routes and the main attendance loop.

    mode values
    -----------
    "idle"        — system just started, camera not yet warm
    "attendance"  — main loop is running recognition
    "enrollment"  — DatasetCapture is collecting images
    "training"    — LBPH Trainer is running
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._mode = "idle"

        self.frame_buffer = FrameBuffer()
        self.recognizer = None  # Recognizer instance; set by main.py after load

        # Enrollment
        self.enroll_face_label: int = 0
        self.enroll_stop_event = threading.Event()
        self.enroll_saved: int = 0
        self.enroll_total: int = 0
        self.enroll_done: bool = False
        self.enroll_error: Optional[str] = None

        # Training
        self.train_running: bool = False
        self.train_done: bool = False
        self.train_error: Optional[str] = None
        self.train_result = None  # TrainingResult dataclass

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode

    def get_mode(self) -> str:
        with self._lock:
            return self._mode
