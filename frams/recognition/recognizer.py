"""
T14 — Recognizer
LBPH face recognition inference for FRAMS.

Loads the trained ``trainer.yml`` model and predicts the identity of a
face ROI.  The caller is responsible for extracting the ROI — use
``FaceDetector.crop_face()`` (T11) to get a normalised 100×100 grayscale
array before passing it here.

Public API
----------
    load()                           Load (or reload) model from disk.
    predict_roi(roi, threshold)  →   RecognitionResult
    is_loaded                    →   bool
    reload()                         Re-read trainer.yml after retraining.

Usage
-----
    from recognition.recognizer import Recognizer, RecognitionResult

    rec = Recognizer()
    rec.load()

    # roi is a (100, 100) grayscale array from FaceDetector.crop_face()
    result = rec.predict_roi(roi)
    if result.is_recognized:
        print(f"face_label={result.face_label}, confidence={result.confidence:.1f}")
    else:
        print("Unknown face")
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional cv2.face import — needs opencv-contrib-python
# ---------------------------------------------------------------------------
try:
    _lbph_create = cv2.face.LBPHFaceRecognizer_create
    _LBPH_AVAILABLE = True
except AttributeError:
    _LBPH_AVAILABLE = False


@dataclass
class RecognitionResult:
    """
    Result of one LBPH prediction.

    Attributes
    ----------
    face_label : int
        The integer label predicted by LBPH.  -1 if the confidence score
        did not meet the threshold (i.e. face is unknown).
    confidence : float
        LBPH dissimilarity score.  **Lower is better.**  A perfect match
        scores 0; anything above ``threshold`` is treated as unknown.
    is_recognized : bool
        ``True`` when ``confidence < threshold``.
    threshold : float
        The threshold that was applied to produce this result.
    """
    face_label: int
    confidence: float
    is_recognized: bool
    threshold: float


class RecognizerError(Exception):
    """Raised when the model cannot be loaded or a prediction fails."""


class Recognizer:
    """
    Wraps an OpenCV LBPHFaceRecognizer for inference.

    Parameters
    ----------
    model_path : str
        Path to the ``trainer.yml`` file written by :class:`Trainer`.
    threshold : float
        Default confidence threshold.  Predictions with a score **at or
        above** this value are classified as unknown.  Can be overridden
        per-call via ``predict_roi(threshold_override=...)``.
    """

    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        threshold: float = config.RECOGNITION_THRESHOLD,
    ) -> None:
        self._model_path = model_path
        self.threshold = threshold
        self._recognizer = None   # populated by load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Read the model from ``model_path`` and prepare it for inference.

        Call this once on startup, and again after the Trainer writes a
        new ``trainer.yml`` (i.e. after re-enrollment).

        Raises
        ------
        RecognizerError
            If opencv-contrib-python is not installed, the model file is
            missing, or the file cannot be parsed.
        """
        self._check_available()

        if not os.path.isfile(self._model_path):
            raise RecognizerError(
                f"Model file not found: '{self._model_path}'\n"
                "Run enrollment and training first."
            )

        try:
            rec = _lbph_create()
            rec.read(self._model_path)
            self._recognizer = rec
            logger.info("LBPH model loaded from %s.", self._model_path)
        except Exception as exc:
            self._recognizer = None
            raise RecognizerError(
                f"Failed to load model from '{self._model_path}': {exc}"
            ) from exc

    def reload(self) -> None:
        """Re-read trainer.yml from disk — call after the Trainer finishes."""
        logger.info("Reloading LBPH model.")
        self.load()

    @property
    def is_loaded(self) -> bool:
        """``True`` if a model has been successfully loaded."""
        return self._recognizer is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_roi(
        self,
        roi: np.ndarray,
        threshold_override: Optional[float] = None,
    ) -> RecognitionResult:
        """
        Predict the identity of a single face ROI.

        Parameters
        ----------
        roi : np.ndarray
            Grayscale uint8 array, shape ``(H, W)``.  Should be the output
            of ``FaceDetector.crop_face()`` — already histogram-equalised
            and resized to ``config.FACE_ROI_SIZE``.
        threshold_override : float, optional
            Override the instance-level threshold for this call only.
            Pass ``db.get_recognition_threshold()`` here so the live
            setting from the Flask dashboard is always honoured.

        Returns
        -------
        RecognitionResult

        Raises
        ------
        RecognizerError
            If :meth:`load` has not been called yet.
        """
        if not self.is_loaded:
            raise RecognizerError(
                "Model not loaded. Call load() before predict_roi()."
            )

        threshold = threshold_override if threshold_override is not None else self.threshold

        # Ensure grayscale — defend against a caller passing a BGR array
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        face_label, confidence = self._recognizer.predict(roi)
        is_recognized = confidence < threshold

        logger.debug(
            "LBPH predict → label=%d, confidence=%.2f, threshold=%.2f, recognised=%s",
            face_label, confidence, threshold, is_recognized,
        )

        return RecognitionResult(
            face_label=face_label if is_recognized else -1,
            confidence=confidence,
            is_recognized=is_recognized,
            threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _check_available() -> None:
        if not _LBPH_AVAILABLE:
            raise RecognizerError(
                "cv2.face is not available. "
                "Install opencv-contrib-python:\n"
                "    pip install opencv-contrib-python\n"
                "(or: pip install -r requirements.txt)"
            )
