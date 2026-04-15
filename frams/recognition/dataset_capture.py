"""
T12 — Dataset Capture
Enrollment image collector for FRAMS.

Drives the Camera and FaceDetector to gather DATASET_IMAGES_PER_STUDENT
grayscale face ROIs per student and write them to:

    DATASET_DIR/{face_label}/face_{n:04d}.jpg

The directory name IS the LBPH integer label — the Trainer (T13) reads it
directly, so the naming convention here and in the trainer must stay in sync.

Public API
----------
    capture(face_label, target_count, on_frame, stop_event) → int
        Run the capture loop.  Returns the number of images saved.

    clear_dataset(face_label)  → int
        Remove all existing images for a student (used before re-enrollment).

    count_images(face_label)   → int
        How many images are already on disk for a student.

    dataset_path(face_label)   → str
        Absolute path to the student's image directory.

Usage
-----
    from hardware.camera import Camera
    from recognition.face_detector import FaceDetector
    from recognition.dataset_capture import DatasetCapture

    with Camera() as cam:
        detector = FaceDetector()
        capture  = DatasetCapture(cam, detector)
        saved    = capture.capture(face_label=3)
        print(f"Saved {saved} images.")
"""

import logging
import os
import time
import threading
from typing import Callable, Optional

import cv2
import numpy as np

import config
from hardware.camera import Camera
from recognition.face_detector import FaceDetector

logger = logging.getLogger(__name__)


class DatasetCapture:
    """
    Capture enrollment images for one student.

    Parameters
    ----------
    camera : Camera
        Already-started Camera instance.
    detector : FaceDetector
        FaceDetector instance (shared with the recognition pipeline is fine).
    dataset_dir : str
        Root directory under which per-student subdirectories are created.
        Defaults to ``config.DATASET_DIR``.
    capture_delay : float
        Seconds to wait between successive successful captures.
        Defaults to ``config.CAPTURE_DELAY_MS / 1000``.
    """

    def __init__(
        self,
        camera: Camera,
        detector: FaceDetector,
        dataset_dir: str = config.DATASET_DIR,
        capture_delay: float = config.CAPTURE_DELAY_MS / 1000.0,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._dataset_dir = dataset_dir
        self._capture_delay = capture_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(
        self,
        face_label: int,
        target_count: int = config.DATASET_IMAGES_PER_STUDENT,
        on_frame: Optional[Callable[[np.ndarray, int, int], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> int:
        """
        Run the capture loop until ``target_count`` face images are saved
        or ``stop_event`` is set.

        Parameters
        ----------
        face_label : int
            LBPH integer label from ``students.face_label``.  Used as the
            subdirectory name so the Trainer can recover the label without
            any extra metadata file.
        target_count : int
            Number of images to collect.
        on_frame : callable(annotated_frame, saved, total), optional
            Called after every camera read (hit or miss).  ``annotated_frame``
            is a BGR copy of the frame with the detected face rectangle drawn
            on it (green = saved, yellow = face found but not yet saved,
            red = no face).  Useful for feeding the Flask MJPEG stream.
        stop_event : threading.Event, optional
            Set this event from another thread to abort the capture early
            (e.g. the Flask UI cancel button).

        Returns
        -------
        int
            Number of images successfully written to disk.
        """
        out_dir = self.dataset_path(face_label)
        os.makedirs(out_dir, exist_ok=True)

        # Start index from the next available number so re-runs append
        saved = self.count_images(face_label)
        start_n = saved
        logger.info(
            "Starting capture for face_label=%d — target=%d, already have=%d, dir=%s",
            face_label, target_count, saved, out_dir,
        )

        while saved - start_n < target_count:
            if stop_event is not None and stop_event.is_set():
                logger.info("Capture stopped by stop_event after %d images.", saved)
                break

            try:
                frame = self._camera.capture_frame()
            except Exception as exc:
                logger.warning("Camera read failed: %s — retrying.", exc)
                time.sleep(0.1)
                continue

            rect = self._detector.detect_largest(frame)

            if rect is not None:
                try:
                    roi = self._detector.crop_face(frame, rect)
                except ValueError as exc:
                    logger.debug("crop_face skipped: %s", exc)
                    rect = None
                else:
                    img_path = os.path.join(out_dir, f"face_{saved:04d}.jpg")
                    ok = cv2.imwrite(img_path, roi)
                    if ok:
                        saved += 1
                        logger.debug(
                            "Saved image %d/%d → %s",
                            saved - start_n, target_count, img_path,
                        )
                    else:
                        logger.warning("cv2.imwrite failed for %s", img_path)

            if on_frame is not None:
                annotated = self._annotate(frame, rect, saved - start_n, target_count)
                try:
                    on_frame(annotated, saved - start_n, target_count)
                except Exception as exc:
                    logger.debug("on_frame callback error (ignored): %s", exc)

            if rect is not None:
                time.sleep(self._capture_delay)

        captured_this_run = saved - start_n
        logger.info(
            "Capture complete for face_label=%d — captured=%d, total on disk=%d.",
            face_label, captured_this_run, saved,
        )
        return captured_this_run

    def clear_dataset(self, face_label: int) -> int:
        """
        Delete all JPEG images in the student's dataset directory.

        Parameters
        ----------
        face_label : int
            Student's LBPH label.

        Returns
        -------
        int
            Number of files deleted.
        """
        out_dir = self.dataset_path(face_label)
        if not os.path.isdir(out_dir):
            logger.debug("clear_dataset: directory not found — %s", out_dir)
            return 0

        deleted = 0
        for fname in os.listdir(out_dir):
            if fname.lower().endswith(".jpg"):
                try:
                    os.remove(os.path.join(out_dir, fname))
                    deleted += 1
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", fname, exc)

        logger.info(
            "Cleared %d image(s) for face_label=%d.", deleted, face_label
        )
        return deleted

    def count_images(self, face_label: int) -> int:
        """
        Return the number of JPEG images already on disk for ``face_label``.
        Returns 0 if the directory doesn't exist.
        """
        out_dir = self.dataset_path(face_label)
        if not os.path.isdir(out_dir):
            return 0
        return sum(1 for f in os.listdir(out_dir) if f.lower().endswith(".jpg"))

    def dataset_path(self, face_label: int) -> str:
        """Absolute path to the image directory for ``face_label``."""
        return os.path.join(self._dataset_dir, str(face_label))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        rect: Optional[tuple],
        saved: int,
        total: int,
    ) -> np.ndarray:
        """
        Draw a status overlay on a copy of the frame for the preview stream.

        Colours
        -------
        Green  — face detected and just saved
        Yellow — face detected (not yet saved, e.g. waiting for delay)
        Red    — no face detected
        """
        out = frame.copy()

        if rect is not None:
            x, y, w, h = rect
            colour = (0, 255, 0) if saved > 0 else (0, 255, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        else:
            h_frame = out.shape[0]
            cv2.putText(
                out, "No face", (10, h_frame - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1,
            )

        # Progress counter in top-left corner
        cv2.putText(
            out, f"{saved}/{total}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        return out
