"""
Supabase client wrapper for FRAMS.

Provides a thin, import-safe layer over supabase-py.  All methods
return None / [] gracefully when Supabase is not configured so the
rest of the app can call them unconditionally.
"""

import logging
import os
from typing import List, Optional

import config

logger = logging.getLogger(__name__)

# Lazy singleton — created on first use so the import never hard-crashes
# when supabase-py is absent (Pi offline-only deployments).
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not config.SUPABASE_ENABLED:
        return None
    try:
        from supabase import create_client
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        logger.info("Supabase client initialised.")
    except Exception as exc:
        logger.error("Could not create Supabase client: %s", exc)
        _client = None
    return _client


# ---------------------------------------------------------------------------
# Face-image records
# ---------------------------------------------------------------------------

def insert_face_image(student_id: int, face_label: int, storage_path: str) -> Optional[dict]:
    """Insert a face_images row after uploading to Storage."""
    client = _get_client()
    if client is None:
        return None
    try:
        res = (
            client.table("face_images")
            .insert({
                "student_id": student_id,
                "face_label": face_label,
                "storage_path": storage_path,
                "is_downloaded": False,
            })
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as exc:
        logger.error("insert_face_image failed: %s", exc)
        return None


def get_undownloaded_images() -> List[dict]:
    """Return all face_images rows where is_downloaded = False."""
    client = _get_client()
    if client is None:
        return []
    try:
        res = (
            client.table("face_images")
            .select("*")
            .eq("is_downloaded", False)
            .order("created_at")
            .execute()
        )
        return res.data or []
    except Exception as exc:
        logger.error("get_undownloaded_images failed: %s", exc)
        return []


def mark_image_downloaded(image_id: int) -> bool:
    """Set is_downloaded = True for the given row."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("face_images").update({"is_downloaded": True}).eq("id", image_id).execute()
        return True
    except Exception as exc:
        logger.error("mark_image_downloaded failed (id=%d): %s", image_id, exc)
        return False


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def upload_image_bytes(storage_path: str, image_bytes: bytes,
                       content_type: str = "image/jpeg") -> bool:
    """Upload raw bytes to the Supabase Storage bucket."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.storage.from_(config.SUPABASE_BUCKET).upload(
            storage_path,
            image_bytes,
            {"content-type": content_type, "upsert": "true"},
        )
        return True
    except Exception as exc:
        logger.error("upload_image_bytes failed (%s): %s", storage_path, exc)
        return False


def download_image_bytes(storage_path: str) -> Optional[bytes]:
    """Download raw bytes from the Supabase Storage bucket."""
    client = _get_client()
    if client is None:
        return None
    try:
        data = client.storage.from_(config.SUPABASE_BUCKET).download(storage_path)
        return data
    except Exception as exc:
        logger.error("download_image_bytes failed (%s): %s", storage_path, exc)
        return None


# ---------------------------------------------------------------------------
# Student sync (optional — keeps Supabase students table in sync)
# ---------------------------------------------------------------------------

def upsert_student(student_id: int, face_label: int,
                   name: str, matric_no: str, department: str) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.table("students").upsert({
            "id": student_id,
            "face_label": face_label,
            "name": name,
            "matric_no": matric_no,
            "department": department,
        }).execute()
        return True
    except Exception as exc:
        logger.error("upsert_student failed: %s", exc)
        return False
