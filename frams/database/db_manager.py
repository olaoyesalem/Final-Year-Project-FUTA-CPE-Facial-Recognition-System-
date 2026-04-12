"""
T5 — Database Manager
Thread-safe SQLite access layer for FRAMS.

Usage
-----
    from database.db_manager import DatabaseManager

    db = DatabaseManager()

    # as a one-shot call
    student = db.get_student_by_label(3)

    # or as a context manager (auto-commits / rolls back)
    with DatabaseManager() as db:
        db.log_attendance(student_id=1, course_id=2, session_id=1, confidence=42.5)
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Optional

import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Wraps every SQLite operation needed by FRAMS."""

    def __init__(self, db_path: str = config.DATABASE_PATH):
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row          # rows behave like dicts
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")   # safe concurrent reads
        return conn

    @contextmanager
    def _cursor(self):
        """Yield a cursor; commit on success, rollback on exception."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Context manager support so callers can use `with DatabaseManager() as db:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Student CRUD
    # ------------------------------------------------------------------

    def add_student(self, name: str, matric_no: str,
                    department: str = "Computer Engineering") -> int:
        """
        Register a new student and assign the next available face_label.
        Returns the new student's primary key (id).
        """
        with self._cursor() as cur:
            # Next LBPH label = max existing label + 1  (starts at 1)
            cur.execute("SELECT COALESCE(MAX(face_label), 0) + 1 FROM students")
            face_label = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO students (face_label, name, matric_no, department)
                VALUES (?, ?, ?, ?)
                """,
                (face_label, name.strip(), matric_no.strip().upper(), department),
            )
            student_id = cur.lastrowid
            logger.info(
                "Enrolled student '%s' (%s) → id=%d, face_label=%d",
                name, matric_no, student_id, face_label,
            )
            return student_id

    def get_student_by_id(self, student_id: int) -> Optional[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM students WHERE id = ? AND is_active = 1",
                (student_id,),
            )
            return cur.fetchone()

    def get_student_by_label(self, face_label: int) -> Optional[sqlite3.Row]:
        """Look up a student by their LBPH integer label."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM students WHERE face_label = ? AND is_active = 1",
                (face_label,),
            )
            return cur.fetchone()

    def get_student_by_matric(self, matric_no: str) -> Optional[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM students WHERE matric_no = ? AND is_active = 1",
                (matric_no.strip().upper(),),
            )
            return cur.fetchone()

    def list_students(self, active_only: bool = True) -> List[sqlite3.Row]:
        with self._cursor() as cur:
            if active_only:
                cur.execute(
                    "SELECT * FROM students WHERE is_active = 1 ORDER BY name"
                )
            else:
                cur.execute("SELECT * FROM students ORDER BY name")
            return cur.fetchall()

    def deactivate_student(self, student_id: int) -> bool:
        """Soft-delete: marks is_active = 0.  Does NOT remove from DB."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE students SET is_active = 0 WHERE id = ?",
                (student_id,),
            )
            changed = cur.rowcount > 0
            if changed:
                logger.info("Deactivated student id=%d", student_id)
            return changed

    def get_all_face_labels(self) -> List[int]:
        """Returns all active face_labels — used by the LBPH trainer."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT face_label FROM students WHERE is_active = 1 ORDER BY face_label"
            )
            return [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Course CRUD
    # ------------------------------------------------------------------

    def add_course(self, course_code: str, course_name: str,
                   department: str = "Computer Engineering",
                   semester: str = "First") -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO courses (course_code, course_name, department, semester)
                VALUES (?, ?, ?, ?)
                """,
                (course_code.strip().upper(), course_name.strip(), department, semester),
            )
            return cur.lastrowid

    def list_courses(self, active_only: bool = True) -> List[sqlite3.Row]:
        with self._cursor() as cur:
            if active_only:
                cur.execute(
                    "SELECT * FROM courses WHERE is_active = 1 ORDER BY course_code"
                )
            else:
                cur.execute("SELECT * FROM courses ORDER BY course_code")
            return cur.fetchall()

    def get_course_by_id(self, course_id: int) -> Optional[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM courses WHERE id = ?", (course_id,))
            return cur.fetchone()

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def list_sessions(self) -> List[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM sessions ORDER BY id")
            return cur.fetchall()

    def get_session_by_name(self, name: str) -> Optional[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM sessions WHERE name = ?", (name,))
            return cur.fetchone()

    # ------------------------------------------------------------------
    # Attendance Logging
    # ------------------------------------------------------------------

    def is_duplicate(self, student_id: int,
                     window_seconds: int = config.DUPLICATE_WINDOW_SECONDS) -> bool:
        """
        Return True if the student already has an attendance record within
        the last `window_seconds` seconds.
        """
        cutoff = (
            datetime.now() - timedelta(seconds=window_seconds)
        ).strftime("%Y-%m-%d %H:%M:%S")

        with self._cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM attendance_logs
                WHERE student_id = ? AND timestamp >= ?
                LIMIT 1
                """,
                (student_id, cutoff),
            )
            return cur.fetchone() is not None

    def log_attendance(self, student_id: int, confidence: float,
                       course_id: Optional[int] = None,
                       session_id: Optional[int] = None) -> Optional[int]:
        """
        Write one attendance record.  Checks for duplicates first.
        Returns the new log id, or None if it was a duplicate.
        """
        if self.is_duplicate(student_id):
            logger.warning(
                "Duplicate attendance blocked for student_id=%d", student_id
            )
            return None

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO attendance_logs (student_id, course_id, session_id, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (student_id, course_id, session_id, round(confidence, 4)),
            )
            log_id = cur.lastrowid
            logger.info(
                "Attendance logged: student_id=%d, log_id=%d, confidence=%.2f",
                student_id, log_id, confidence,
            )
            return log_id

    # ------------------------------------------------------------------
    # Attendance Queries (used by Flask dashboard)
    # ------------------------------------------------------------------

    def get_today_attendance(self) -> List[sqlite3.Row]:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    al.id,
                    s.name        AS student_name,
                    s.matric_no,
                    c.course_code,
                    c.course_name,
                    se.name       AS session,
                    al.timestamp,
                    al.confidence
                FROM attendance_logs al
                JOIN students s  ON s.id  = al.student_id
                LEFT JOIN courses  c  ON c.id  = al.course_id
                LEFT JOIN sessions se ON se.id = al.session_id
                WHERE DATE(al.timestamp) = ?
                ORDER BY al.timestamp DESC
                """,
                (today,),
            )
            return cur.fetchall()

    def get_attendance_filtered(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        course_id: Optional[int] = None,
        session_id: Optional[int] = None,
    ) -> List[sqlite3.Row]:
        """
        Flexible filter used by the reports page.
        date_from / date_to should be 'YYYY-MM-DD' strings.
        """
        clauses = []
        params: List = []

        if date_from:
            clauses.append("DATE(al.timestamp) >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("DATE(al.timestamp) <= ?")
            params.append(date_to)
        if course_id:
            clauses.append("al.course_id = ?")
            params.append(course_id)
        if session_id:
            clauses.append("al.session_id = ?")
            params.append(session_id)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    al.id,
                    s.name        AS student_name,
                    s.matric_no,
                    c.course_code,
                    c.course_name,
                    se.name       AS session,
                    al.timestamp,
                    al.confidence
                FROM attendance_logs al
                JOIN students s  ON s.id  = al.student_id
                LEFT JOIN courses  c  ON c.id  = al.course_id
                LEFT JOIN sessions se ON se.id = al.session_id
                {where}
                ORDER BY al.timestamp DESC
                """,
                params,
            )
            return cur.fetchall()

    def get_attendance_summary(self, date_from: Optional[str] = None,
                               date_to: Optional[str] = None) -> List[sqlite3.Row]:
        """Per-student attendance count — useful for the dashboard summary card."""
        clauses = []
        params: List = []
        if date_from:
            clauses.append("DATE(al.timestamp) >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("DATE(al.timestamp) <= ?")
            params.append(date_to)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    s.matric_no,
                    s.name   AS student_name,
                    COUNT(*) AS total_present
                FROM attendance_logs al
                JOIN students s ON s.id = al.student_id
                {where}
                GROUP BY al.student_id
                ORDER BY total_present DESC
                """,
                params,
            )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # Sync support (used by T22)
    # ------------------------------------------------------------------

    def get_unsynced_logs(self) -> List[sqlite3.Row]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT al.*, s.name AS student_name, s.matric_no
                FROM attendance_logs al
                JOIN students s ON s.id = al.student_id
                WHERE al.is_synced = 0
                ORDER BY al.timestamp
                """
            )
            return cur.fetchall()

    def mark_logs_synced(self, log_ids: List[int]) -> None:
        if not log_ids:
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        placeholders = ",".join("?" * len(log_ids))
        with self._cursor() as cur:
            cur.execute(
                f"""
                UPDATE attendance_logs
                SET is_synced = 1, synced_at = ?
                WHERE id IN ({placeholders})
                """,
                [now, *log_ids],
            )
        logger.info("Marked %d log(s) as synced.", len(log_ids))

    # ------------------------------------------------------------------
    # System Settings
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT value FROM system_settings WHERE key = ?", (key,)
            )
            row = cur.fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO system_settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value)),
            )
        logger.debug("Setting updated: %s = %s", key, value)

    def get_recognition_threshold(self) -> float:
        """Returns the live threshold (may have been changed via the settings page)."""
        raw = self.get_setting("recognition_threshold",
                               str(config.RECOGNITION_THRESHOLD))
        return float(raw)
