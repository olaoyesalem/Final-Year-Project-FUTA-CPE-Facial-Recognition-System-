"""
T11 — Face Detector
Haar Cascade face detection wrapper for FRAMS.

Provides:
  - detect(frame)           → list of (x, y, w, h) for every face found
  - detect_largest(frame)   → single largest (x, y, w, h) or None
  - crop_face(frame, rect)  → grayscale, histogram-equalised, resized ROI
                              ready to feed directly into LBPH train/predict

Usage
-----
    from recognition.face_detector import FaceDetector, FaceDetectorError

    detector = FaceDetector()
    rects = detector.detect(frame)
    if rects:
        roi = detector.crop_face(frame, rects[0])   # (100, 100) uint8 gray
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Type alias for a face rectangle
Rect = Tuple[int, int, int, int]   # (x, y, w, h)


class FaceDetectorError(Exception):
    """Raised when the cascade file cannot be loaded."""


class FaceDetector:
    """
    Thin wrapper around OpenCV's Haar Cascade detector.

    Parameters
    ----------
    cascade_path : str
        Path to ``haarcascade_frontalface_default.xml``.
    scale_factor : float
        Image pyramid scale step (>1.0).  Smaller = slower but more thorough.
    min_neighbors : int
        Minimum number of overlapping detections to retain a candidate.
    min_size : tuple[int, int]
        Smallest face rectangle the detector will report.
    roi_size : tuple[int, int]
        Output size for ``crop_face()``.  Must match the size used during
        training so that LBPH histogram grids are consistent.
    """

    def __init__(
        self,
        cascade_path: str = config.CASCADE_PATH,
        scale_factor: float = config.HAAR_SCALE_FACTOR,
        min_neighbors: int = config.HAAR_MIN_NEIGHBORS,
        min_size: Tuple[int, int] = config.HAAR_MIN_SIZE,
        roi_size: Tuple[int, int] = config.FACE_ROI_SIZE,
    ) -> None:
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        self._roi_size = roi_size

        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise FaceDetectorError(
                f"Failed to load Haar Cascade from: {cascade_path}\n"
                "Make sure haarcascade_frontalface_default.xml is in frams/cascades/."
            )
        logger.debug("Haar Cascade loaded from %s.", cascade_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Rect]:
        """
        Detect all faces in a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as returned by ``Camera.capture_frame()``.

        Returns
        -------
        list of (x, y, w, h)
            Every detected face rectangle, sorted largest-first by area.
            Empty list if no faces are found.
        """
        gray = self._to_gray(frame)
        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )

        if len(detections) == 0:
            logger.debug("No faces detected.")
            return []

        # Sort largest face first so callers can simply take [0]
        rects: List[Rect] = sorted(
            [tuple(d) for d in detections],
            key=lambda r: r[2] * r[3],   # w * h
            reverse=True,
        )
        logger.debug("%d face(s) detected.", len(rects))
        return rects

    def detect_largest(self, frame: np.ndarray) -> Optional[Rect]:
        """
        Return the largest face rectangle, or None if no face is found.

        This is the primary entry point for the attendance scan where only
        one face is expected in front of the camera.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.

        Returns
        -------
        (x, y, w, h) or None
        """
        rects = self.detect(frame)
        return rects[0] if rects else None

    def crop_face(
        self,
        frame: np.ndarray,
        rect: Rect,
        roi_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Extract and normalise a face ROI for LBPH.

        Steps:
          1. Crop the bounding box from the frame.
          2. Convert to grayscale (if the frame is BGR).
          3. Apply histogram equalisation to reduce lighting variance.
          4. Resize to ``roi_size`` (default: ``config.FACE_ROI_SIZE``).

        Parameters
        ----------
        frame : np.ndarray
            BGR image the rect was detected in.
        rect : (x, y, w, h)
            Bounding box as returned by ``detect()`` / ``detect_largest()``.
        roi_size : (width, height), optional
            Override the default ROI size.

        Returns
        -------
        np.ndarray
            Grayscale uint8 array of shape ``(roi_size[1], roi_size[0])``.

        Raises
        ------
        ValueError
            If the rect falls (partially) outside the frame bounds.
        """
        size = roi_size or self._roi_size
        x, y, w, h = rect
        fh, fw = frame.shape[:2]

        if x < 0 or y < 0 or x + w > fw or y + h > fh:
            raise ValueError(
                f"Rect {rect} is out of frame bounds ({fw}×{fh})."
            )

        crop = frame[y : y + h, x : x + w]
        gray = self._to_gray(crop)
        equalised = cv2.equalizeHist(gray)
        resized = cv2.resize(equalised, size, interpolation=cv2.INTER_AREA)
        return resized

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert BGR or already-grayscale image to single-channel grayscale."""
        if len(image.shape) == 2:
            return image          # already grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
