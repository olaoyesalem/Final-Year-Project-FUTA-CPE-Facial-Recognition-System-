"""T22 — Network Sync Manager
Periodically POSTs unsynced attendance records to a remote server endpoint.
"""
import logging
import threading
import time

import requests

import config
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class SyncManager:
    """
    Background thread that uploads unsynced attendance logs to SYNC_REMOTE_URL.

    Skips silently if SYNC_ENABLED is False or SYNC_REMOTE_URL is empty.
    Uses exponential back-off (2^attempt seconds) between retry attempts.
    """

    def __init__(self, db: DatabaseManager = None):
        self._db = db or DatabaseManager()
        self._stop_event = threading.Event()
        self._thread: threading.Thread = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not config.SYNC_ENABLED:
            logger.info("Sync disabled (SYNC_ENABLED=False).")
            return
        if not config.SYNC_REMOTE_URL:
            logger.info("Sync disabled (SYNC_REMOTE_URL not configured).")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="sync-manager"
        )
        self._thread.start()
        logger.info(
            "SyncManager started — interval=%ds, url=%s",
            config.SYNC_INTERVAL_SECONDS,
            config.SYNC_REMOTE_URL,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("SyncManager stopped.")

    def sync_now(self) -> dict:
        """Force an immediate sync outside the scheduled interval."""
        return self._do_sync()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.wait(config.SYNC_INTERVAL_SECONDS):
            self._do_sync()

    def _do_sync(self) -> dict:
        logs = self._db.get_unsynced_logs()
        if not logs:
            logger.debug("Sync: nothing to upload.")
            return {"synced": 0, "failed": 0}

        payload = {
            "records": [
                {
                    "id": row["id"],
                    "student_name": row["student_name"],
                    "matric_no": row["matric_no"],
                    "timestamp": row["timestamp"],
                    "confidence": row["confidence"],
                }
                for row in logs
            ]
        }

        for attempt in range(1, config.SYNC_RETRY_ATTEMPTS + 1):
            try:
                resp = requests.post(
                    config.SYNC_REMOTE_URL,
                    json=payload,
                    timeout=config.SYNC_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                ids = [row["id"] for row in logs]
                self._db.mark_logs_synced(ids)
                logger.info("Synced %d record(s).", len(ids))
                return {"synced": len(ids), "failed": 0}
            except requests.RequestException as exc:
                logger.warning(
                    "Sync attempt %d/%d failed: %s",
                    attempt, config.SYNC_RETRY_ATTEMPTS, exc,
                )
                if attempt < config.SYNC_RETRY_ATTEMPTS:
                    time.sleep(2 ** attempt)

        logger.error("All sync attempts failed for %d record(s).", len(logs))
        return {"synced": 0, "failed": len(logs)}
