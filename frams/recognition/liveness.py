"""
T15 — Liveness Detector
Blink-based liveness check using the Haar eye cascade.

No dlib required.  EAR (Eye Aspect Ratio) is approximated from the number
of eyes the cascade detects within the upper half of the face ROI:

    ear = eyes_found / 2.0      →  0.0 (both closed) … 1.0 (both open)

A frame is a *blink frame* when ``ear < EYE_AR_THRESHOLD`` (default 0.25),
meaning zero eyes were detected.  Liveness is confirmed when at least
``LIVENESS_BLINK_FRAMES`` blink frames occur within a rolling window of
``LIVENESS_WINDOW_FRAMES`` frames.

When ``LIVENESS_ENABLED = False`` every call immediately returns ``True``
so the rest of the pipeline is unaffected.

Public API
----------
    update(frame, face_rect)  →  True | False | None
        Feed one frame.  Returns a decision when the window is full,
        ``None`` while still collecting.  This is the primary interface
        used by the main attendance loop and the Flask MJPEG preview.

    reset()
        Clear the rolling window between scans.

    check(frame_iter, rect_iter)  →  bool
        Convenience wrapper that drives an iterator until a decision is
        reached or the iterators are exhausted.

Usage
-----
    from recognition.liveness import LivenessDetector

    detector = LivenessDetector()
    detector.reset()

    for frame, rect in zip(camera_frames, face_rects):
        decision = detector.update(frame, rect)
        if decision is not None:
            print("Live!" if decision else "Spoof detected.")
            break
"""

import collections
import logging
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Type alias
Rect = Tuple[int, int, int, int]


class LivenessDetector:
    """
    Blink-based liveness detector driven by the Haar eye cascade.

    Parameters
    ----------
    eye_cascade_path : str
        Path to ``haarcascade_eye.xml``.
    window_frames : int
        Number of frames in the rolling blink buffer.
    blink_frames : int
        Minimum blink frames needed to pass liveness.
    ear_threshold : float
        EAR value below which a frame counts as a blink frame.
        With the Haar proxy (``ear = eyes / 2``), 0.25 means
        zero eyes detected.
    enabled : bool
        When ``False``, every ``update()`` call immediately returns ``True``.
    """

    def __init__(
        self,
        eye_cascade_path: str = config.EYE_CASCADE_PATH,
        window_frames: int = config.LIVENESS_WINDOW_FRAMES,
        blink_frames: int = config.LIVENESS_BLINK_FRAMES,
        ear_threshold: float = config.EYE_AR_THRESHOLD,
        enabled: bool = config.LIVENESS_ENABLED,
    ) -> None:
        self._window_frames = window_frames
        self._blink_frames = blink_frames
        self._ear_threshold = ear_threshold
        self._enabled = enabled

        self._window: collections.deque = collections.deque(maxlen=window_frames)

        self._cascade: Optional[cv2.CascadeClassifier] = None
        if enabled:
            self._cascade = cv2.CascadeClassifier(eye_cascade_path)
            if self._cascade.empty():
                logger.warning(
                    "Could not load eye cascade from %s — liveness will pass "
                    "all frames (disabled).",
                    eye_cascade_path,
                )
                self._cascade = None
                self._enabled = False
            else:
                logger.debug("Eye cascade loaded from %s.", eye_cascade_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        face_rect: Rect,
    ) -> Optional[bool]:
        """
        Feed one frame into the rolling window and return a decision when
        the window is full.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame from the camera.
        face_rect : (x, y, w, h)
            The face bounding box detected in ``frame`` by FaceDetector.

        Returns
        -------
        True
            Window is full and enough blink frames were observed → live.
        False
            Window is full but insufficient blinks → likely a spoof.
        None
            Window is not yet full — keep feeding frames.
        """
        if not self._enabled:
            return True

        ear = self._compute_ear(frame, face_rect)
        is_blink_frame = ear < self._ear_threshold
        self._window.append(is_blink_frame)

        logger.debug(
            "Liveness frame: ear=%.2f blink=%s  buffer=%d/%d  blinks_so_far=%d",
            ear, is_blink_frame, len(self._window),
            self._window_frames, sum(self._window),
        )

        if len(self._window) == self._window_frames:
            blink_count = sum(self._window)
            passed = blink_count >= self._blink_frames
            logger.info(
                "Liveness decision: %s  (blink_frames=%d/%d, required>=%d)",
                "PASS" if passed else "FAIL",
                blink_count, self._window_frames, self._blink_frames,
            )
            return passed

        return None

    def reset(self) -> None:
        """Clear the rolling window.  Call between successive scans."""
        self._window.clear()
        logger.debug("Liveness window reset.")

    def check(
        self,
        frame_iter: Iterator[np.ndarray],
        rect_iter: Iterator[Rect],
    ) -> bool:
        """
        Convenience wrapper: drive iterators until a decision is made.

        Returns
        -------
        bool
            ``True`` if liveness confirmed, ``False`` otherwise.
            Returns ``True`` immediately if liveness is disabled or if
            either iterator is exhausted before the window fills
            (treats incomplete data as a pass to avoid blocking attendance).
        """
        self.reset()
        for frame, rect in zip(frame_iter, rect_iter):
            decision = self.update(frame, rect)
            if decision is not None:
                return decision
        # Iterators exhausted before window filled — treat as pass
        logger.warning(
            "Liveness check exhausted frame supply before window filled "
            "(%d/%d frames). Treating as PASS.",
            len(self._window), self._window_frames,
        )
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_ear(self, frame: np.ndarray, face_rect: Rect) -> float:
        """
        Detect eyes inside the upper 60 % of the face bounding box and
        return a proxy EAR:

            ear = eyes_detected / 2.0

        Constraining the search to the upper portion of the face prevents
        the mouth and nostrils from generating false eye detections.

        Returns
        -------
        float
            0.0 (no eyes), 0.5 (one eye), or 1.0 (both eyes).
        """
        x, y, w, h = face_rect
        fh, fw = frame.shape[:2]

        # Clamp to frame bounds before slicing
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        # Upper 60 % of the face for the eye search region
        y2 = min(fh, y + int(h * 0.60))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        face_crop = frame[y1:y2, x1:x2]
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop

        eyes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
        )

        # Cap at 2 — occasional false positives should not inflate the EAR
        eye_count = min(len(eyes) if len(eyes) > 0 else 0, 2)
        return eye_count / 2.0
