"""
T13 — Trainer
LBPH face recognition model trainer for FRAMS.

Walks ``DATASET_DIR``, loads every student's grayscale ROIs, trains an
OpenCV LBPHFaceRecognizer, and writes the model to ``MODEL_PATH``.

Dataset layout expected (produced by DatasetCapture — T12):

    DATASET_DIR/
        {face_label}/          ← directory name IS the LBPH integer label
            face_0000.jpg
            face_0001.jpg
            ...

Usage
-----
    from recognition.trainer import Trainer, TrainerError

    trainer = Trainer()
    result  = trainer.train()
    print(f"Trained on {result.num_images} images for {result.num_students} student(s).")
    print(f"Model saved → {result.model_path}")
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

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
class TrainingResult:
    """Summary returned by :meth:`Trainer.train`."""
    num_students: int   # number of distinct face labels trained
    num_images: int     # total images used for training
    model_path: str     # absolute path where trainer.yml was written


class TrainerError(Exception):
    """Raised when training cannot proceed."""


class Trainer:
    """
    Loads the on-disk dataset and produces a trained LBPH model.

    Parameters
    ----------
    dataset_dir : str
        Root directory produced by DatasetCapture.
    model_path : str
        Destination for the trained ``trainer.yml``.
    roi_size : tuple[int, int]
        Expected (width, height) of each image.  Images that don't match
        are resized so LBPH histogram grids stay consistent.
    lbph_radius : int
        Radius of the circular LBP pattern.
    lbph_neighbors : int
        Number of sample points on the circle.
    lbph_grid_x : int
        Columns in the spatial histogram grid.
    lbph_grid_y : int
        Rows in the spatial histogram grid.
    """

    def __init__(
        self,
        dataset_dir: str = config.DATASET_DIR,
        model_path: str = config.MODEL_PATH,
        roi_size: Tuple[int, int] = config.FACE_ROI_SIZE,
        lbph_radius: int = config.LBPH_RADIUS,
        lbph_neighbors: int = config.LBPH_NEIGHBORS,
        lbph_grid_x: int = config.LBPH_GRID_X,
        lbph_grid_y: int = config.LBPH_GRID_Y,
    ) -> None:
        self._dataset_dir = dataset_dir
        self._model_path = model_path
        self._roi_size = roi_size
        self._lbph_radius = lbph_radius
        self._lbph_neighbors = lbph_neighbors
        self._lbph_grid_x = lbph_grid_x
        self._lbph_grid_y = lbph_grid_y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        """
        Load the full dataset, train LBPH, and save the model.

        Returns
        -------
        TrainingResult
            Contains ``num_students``, ``num_images``, and ``model_path``.

        Raises
        ------
        TrainerError
            If opencv-contrib-python is not installed, the dataset directory
            is missing, or no usable images are found.
        """
        self._check_available()

        images, labels = self._load_dataset()

        if not images:
            raise TrainerError(
                f"No training images found in '{self._dataset_dir}'. "
                "Enroll at least one student first."
            )

        num_students = len(set(labels))
        num_images = len(images)
        logger.info(
            "Training LBPH on %d image(s) for %d student(s).",
            num_images, num_students,
        )

        recognizer = _lbph_create(
            radius=self._lbph_radius,
            neighbors=self._lbph_neighbors,
            grid_x=self._lbph_grid_x,
            grid_y=self._lbph_grid_y,
        )
        recognizer.train(images, np.array(labels, dtype=np.int32))

        self._save_model(recognizer)

        logger.info(
            "Model saved → %s  (students=%d, images=%d).",
            self._model_path, num_students, num_images,
        )
        return TrainingResult(
            num_students=num_students,
            num_images=num_images,
            model_path=self._model_path,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_dataset(self) -> Tuple[List[np.ndarray], List[int]]:
        """
        Walk ``dataset_dir`` and return parallel lists of (images, labels).

        Each subdirectory whose name is a valid integer is treated as one
        student with that LBPH label.  Non-integer directory names and
        non-JPEG files are silently skipped.
        """
        if not os.path.isdir(self._dataset_dir):
            raise TrainerError(
                f"Dataset directory not found: '{self._dataset_dir}'"
            )

        images: List[np.ndarray] = []
        labels: List[int] = []

        for entry in sorted(os.listdir(self._dataset_dir)):
            subdir = os.path.join(self._dataset_dir, entry)
            if not os.path.isdir(subdir):
                continue

            # Directory name must be a plain integer (the face_label)
            try:
                face_label = int(entry)
            except ValueError:
                logger.debug("Skipping non-integer directory: %s", entry)
                continue

            loaded = self._load_student_images(face_label, subdir)
            if not loaded:
                logger.warning(
                    "No usable images for face_label=%d in %s — skipping.",
                    face_label, subdir,
                )
                continue

            images.extend(loaded)
            labels.extend([face_label] * len(loaded))
            logger.debug(
                "face_label=%d: loaded %d image(s).", face_label, len(loaded)
            )

        return images, labels

    def _load_student_images(
        self, face_label: int, directory: str
    ) -> List[np.ndarray]:
        """
        Read all JPEG images from ``directory``, returning grayscale arrays
        resized to ``roi_size``.
        """
        result: List[np.ndarray] = []

        for fname in sorted(os.listdir(directory)):
            if not fname.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(directory, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                logger.warning("Could not read image: %s — skipping.", img_path)
                continue

            # Ensure consistent size in case images were saved at wrong dimensions
            if (img.shape[1], img.shape[0]) != self._roi_size:
                img = cv2.resize(img, self._roi_size, interpolation=cv2.INTER_AREA)
                logger.debug(
                    "Resized %s to %s.", fname, self._roi_size
                )

            result.append(img)

        return result

    def _save_model(self, recognizer) -> None:
        """Write the trained model to ``model_path``, creating dirs as needed."""
        parent = os.path.dirname(self._model_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        recognizer.write(self._model_path)

    @staticmethod
    def _check_available() -> None:
        if not _LBPH_AVAILABLE:
            raise TrainerError(
                "cv2.face is not available. "
                "Install opencv-contrib-python:\n"
                "    pip install opencv-contrib-python\n"
                "(or: pip install -r requirements.txt)"
            )
