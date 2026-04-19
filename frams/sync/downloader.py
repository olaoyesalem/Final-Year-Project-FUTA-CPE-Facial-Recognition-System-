"""
Pi Face-Image Downloader (Supabase → local dataset).

Runs as a background daemon thread.  Every DOWNLOADER_POLL_SECONDS it:
  1. Queries Supabase face_images for rows where is_downloaded = False.
  2. Downloads each image from Supabase Storage.
  3. Saves it to  dataset/{face_label}/{image_id}.jpg
  4. Marks the row is_downloaded = True in Supabase.
  5. If any images were saved, triggers LBPH retrain and reloads the
     recognizer so the Pi starts recognising the new student immediately.
"""

import logging
import os
import threading
import time

import config
from database import supabase_client as supa
from recognition.trainer import Trainer, TrainerError

logger = logging.getLogger(__name__)


class Downloader:
    """Background thread that syncs face images from Supabase to the Pi."""

    def __init__(self, app_state=None):
        self._app_state = app_state   # optional — used to reload recognizer
        self._thread: threading.Thread = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------

    def start(self) -> None:
        if not config.SUPABASE_ENABLED:
            logger.info("Supabase not configured — downloader disabled.")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="downloader",
        )
        self._thread.start()
        logger.info("Downloader started (poll every %ds).", config.DOWNLOADER_POLL_SECONDS)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sync_once()
            except Exception as exc:
                logger.error("Downloader iteration error: %s", exc)
            self._stop.wait(config.DOWNLOADER_POLL_SECONDS)

    def _sync_once(self) -> None:
        rows = supa.get_undownloaded_images()
        if not rows:
            return

        logger.info("Downloader: %d new image(s) to fetch.", len(rows))
        saved = 0

        for row in rows:
            image_id    = row["id"]
            face_label  = row["face_label"]
            storage_path = row["storage_path"]

            img_bytes = supa.download_image_bytes(storage_path)
            if img_bytes is None:
                logger.warning("Could not download %s — skipping.", storage_path)
                continue

            dest_dir = os.path.join(config.DATASET_DIR, str(face_label))
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, f"{image_id}.jpg")

            with open(dest_path, "wb") as fh:
                fh.write(img_bytes)

            if supa.mark_image_downloaded(image_id):
                saved += 1
                logger.debug("Saved %s → %s", storage_path, dest_path)
            else:
                logger.warning("Downloaded but failed to mark id=%d.", image_id)

        if saved:
            logger.info("Downloader: saved %d image(s). Triggering retrain…", saved)
            self._retrain()

    def _retrain(self) -> None:
        try:
            result = Trainer().train()
            logger.info(
                "Retrain complete — %d student(s), %d image(s).",
                result.num_students, result.num_images,
            )
            if self._app_state and self._app_state.recognizer:
                self._app_state.recognizer.reload()
                logger.info("Recognizer reloaded after retrain.")
        except TrainerError as exc:
            logger.error("Retrain failed: %s", exc)
        except Exception as exc:
            logger.error("Unexpected retrain error: %s", exc)
