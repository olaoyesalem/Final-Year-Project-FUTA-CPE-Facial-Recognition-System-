"""
T7 — Camera Driver
Unified camera abstraction for FRAMS.

On a Raspberry Pi with Pi Camera Module 2, picamera2 is used automatically.
On any other machine (dev/test), it falls back to OpenCV VideoCapture.

Usage
-----
    from hardware.camera import Camera

    with Camera() as cam:
        frame = cam.capture_frame()          # numpy BGR array
        cam.save_image(frame, "/tmp/test.jpg")
"""

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional picamera2 import — only present on Raspberry Pi OS
# ---------------------------------------------------------------------------
try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False


class CameraError(Exception):
    """Raised when the camera cannot be opened or a frame cannot be read."""


class Camera:
    """
    Wraps picamera2 (Pi) or OpenCV VideoCapture (dev/fallback) behind a
    single interface.  Supports use as a context manager.

    Attributes
    ----------
    width, height : int
        Capture resolution in pixels.
    framerate : int
        Target frames per second (best-effort; OpenCV may not honour it).
    """

    def __init__(
        self,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        framerate: int = config.CAMERA_FRAMERATE,
        camera_index: int = config.CAMERA_INDEX,
    ) -> None:
        self.width = width
        self.height = height
        self.framerate = framerate
        self._camera_index = camera_index

        self._picam: Optional["Picamera2"] = None
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open and configure the camera.  Must be called before capture_frame."""
        if _PICAMERA2_AVAILABLE:
            self._start_picamera2()
        else:
            self._start_opencv()

    def stop(self) -> None:
        """Release camera resources."""
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing picamera2: %s", exc)
            finally:
                self._picam = None
            logger.debug("picamera2 released.")

        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("OpenCV VideoCapture released.")

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def capture_frame(self) -> np.ndarray:
        """
        Capture one frame and return it as a numpy array in BGR colour order
        (compatible with OpenCV and Haar Cascade detector).

        Returns
        -------
        np.ndarray
            Shape (height, width, 3), dtype uint8, BGR.

        Raises
        ------
        CameraError
            If the camera is not started or the frame cannot be read.
        """
        if self._picam is not None:
            return self._capture_picamera2()
        if self._cap is not None:
            return self._capture_opencv()
        raise CameraError("Camera is not started. Call start() first.")

    def save_image(self, frame: np.ndarray, path: str) -> None:
        """
        Write a BGR frame to disk as a JPEG.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array as returned by capture_frame().
        path : str
            Destination file path.  Parent directory is created if missing.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ok = cv2.imwrite(path, frame)
        if not ok:
            raise CameraError(f"cv2.imwrite failed for path: {path}")
        logger.debug("Frame saved → %s", path)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Internal — picamera2 backend
    # ------------------------------------------------------------------

    def _start_picamera2(self) -> None:
        logger.info(
            "Starting picamera2 (%dx%d @ %d fps).",
            self.width, self.height, self.framerate,
        )
        try:
            self._picam = Picamera2()
            cfg = self._picam.create_still_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"},
            )
            self._picam.configure(cfg)
            self._picam.start()
            # Short warm-up so the sensor auto-exposure settles
            time.sleep(0.5)
            logger.info("picamera2 ready.")
        except Exception as exc:
            self._picam = None
            raise CameraError(f"picamera2 init failed: {exc}") from exc

    def _capture_picamera2(self) -> np.ndarray:
        try:
            frame = self._picam.capture_array("main")
            # picamera2 returns (H, W, 3) BGR888 — already correct colour order
            return frame
        except Exception as exc:
            raise CameraError(f"picamera2 capture failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal — OpenCV backend
    # ------------------------------------------------------------------

    def _start_opencv(self) -> None:
        logger.info(
            "picamera2 not available — using OpenCV VideoCapture (index=%d, %dx%d @ %d fps).",
            self._camera_index, self.width, self.height, self.framerate,
        )
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            raise CameraError(
                f"OpenCV could not open camera at index {self._camera_index}."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.framerate)
        self._cap = cap
        logger.info("OpenCV VideoCapture ready.")

    def _capture_opencv(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("OpenCV VideoCapture.read() returned no frame.")
        return frame
