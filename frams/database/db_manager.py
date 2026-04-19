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

    def get_course_student_stats(self, course_id: int) -> List[sqlite3.Row]:
        """
        For every active student: how many distinct days they attended this course
        vs total distinct days the course ran (any student attended).
        Ordered highest attendance first.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                WITH course_days AS (
                    SELECT COUNT(DISTINCT DATE(timestamp)) AS total_days
                    FROM attendance_logs
                    WHERE course_id = ?
                ),
                student_days AS (
                    SELECT student_id,
                           COUNT(DISTINCT DATE(timestamp)) AS attended_days
                    FROM attendance_logs
                    WHERE course_id = ?
                    GROUP BY student_id
                )
                SELECT
                    s.id,
                    s.name           AS student_name,
                    s.matric_no,
                    COALESCE(sd.attended_days, 0)                      AS attended,
                    (SELECT total_days FROM course_days)                AS total,
                    ROUND(100.0 * COALESCE(sd.attended_days, 0)
                          / MAX((SELECT total_days FROM course_days), 1)) AS pct
                FROM students s
                LEFT JOIN student_days sd ON sd.student_id = s.id
                WHERE s.is_active = 1
                ORDER BY pct DESC, s.name
                """,
                (course_id, course_id),
            )
            return cur.fetchall()

    def get_course_daily_totals(self, course_id: int) -> List[sqlite3.Row]:
        """Distinct attendance count per day for this course — used for the sparkline."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT DATE(timestamp) AS day, COUNT(DISTINCT student_id) AS count
                FROM attendance_logs
                WHERE course_id = ?
                GROUP BY day
                ORDER BY day DESC
                LIMIT 30
                """,
                (course_id,),
            )
            return cur.fetchall()

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
                       session_id: Optional[int] = None,
                       timestamp: Optional[str] = None) -> Optional[int]:
        """
        Write one attendance record.  Checks for duplicates first.
        Returns the new log id, or None if it was a duplicate.

        Parameters
        ----------
        timestamp : str, optional
            'YYYY-MM-DD HH:MM:SS' string.  Defaults to current local time.
        """
        if self.is_duplicate(student_id):
            logger.warning(
                "Duplicate attendance blocked for student_id=%d", student_id
            )
            return None

        with self._cursor() as cur:
            if timestamp:
                cur.execute(
                    """
                    INSERT INTO attendance_logs
                        (student_id, course_id, session_id, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (student_id, course_id, session_id, round(confidence, 4), timestamp),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO attendance_logs (student_id, course_id, session_id, confidence)
                    VALUES (?, ?, ?, ?)
                    """,
                    (student_id, course_id, session_id, round(confidence, 4)),
                )
            log_id = cur.lastrowid
            logger.info(
                "Attendance logged: student_id=%d, log_id=%d, confidence=%.2f, ts=%s",
                student_id, log_id, confidence, timestamp or "now",
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
    # Student Profile (attendance analytics)
    # ------------------------------------------------------------------

    def get_student_attendance_stats(self, student_id: int) -> dict:
        """
        Returns overall attendance stats for one student.
        Denominator = unique (course_id, date) pairs across ALL students
        (proxy for "total scheduled classes").
        """
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(DISTINCT course_id || '_' || DATE(timestamp))
                FROM attendance_logs
                WHERE course_id IS NOT NULL
                """
            )
            total = cur.fetchone()[0] or 0

            cur.execute(
                """
                SELECT COUNT(DISTINCT course_id || '_' || DATE(timestamp))
                FROM attendance_logs
                WHERE student_id = ? AND course_id IS NOT NULL
                """,
                (student_id,),
            )
            attended = cur.fetchone()[0] or 0

        pct = round(attended / total * 100) if total else 0
        return {"total": total, "attended": attended, "pct": pct}

    def get_student_streak(self, student_id: int) -> int:
        """Consecutive-day streak (counts back from today or yesterday)."""
        from datetime import date, timedelta

        with self._cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT DATE(timestamp) AS day
                FROM attendance_logs
                WHERE student_id = ?
                ORDER BY day DESC
                """,
                (student_id,),
            )
            days = [row[0] for row in cur.fetchall()]

        if not days:
            return 0

        today = date.today()
        anchor = today if days[0] == today.isoformat() else today - timedelta(days=1)
        if days[0] != anchor.isoformat():
            return 0

        streak = 0
        check  = anchor
        for d in days:
            if d == check.isoformat():
                streak += 1
                check -= timedelta(days=1)
            else:
                break
        return streak

    def get_student_course_stats(self, student_id: int) -> List[sqlite3.Row]:
        """
        Per-course attendance: student's attended days vs total class days.
        Only includes courses that have at least one attendance record globally.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                WITH course_totals AS (
                    SELECT course_id,
                           COUNT(DISTINCT DATE(timestamp)) AS total_days
                    FROM attendance_logs
                    WHERE course_id IS NOT NULL
                    GROUP BY course_id
                ),
                student_days AS (
                    SELECT course_id,
                           COUNT(DISTINCT DATE(timestamp)) AS attended_days
                    FROM attendance_logs
                    WHERE student_id = ? AND course_id IS NOT NULL
                    GROUP BY course_id
                )
                SELECT
                    c.id,
                    c.course_code,
                    c.course_name,
                    COALESCE(sd.attended_days, 0)          AS attended,
                    ct.total_days                           AS total,
                    ROUND(100.0 * COALESCE(sd.attended_days, 0)
                          / ct.total_days)                 AS pct
                FROM course_totals ct
                JOIN courses c ON c.id = ct.course_id
                LEFT JOIN student_days sd ON sd.course_id = ct.course_id
                ORDER BY c.course_code
                """,
                (student_id,),
            )
            return cur.fetchall()

    def get_student_attendance_dates(self, student_id: int) -> List[str]:
        """All distinct dates (YYYY-MM-DD) the student has an attendance record."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT DATE(timestamp) AS day
                FROM attendance_logs
                WHERE student_id = ?
                ORDER BY day DESC
                """,
                (student_id,),
            )
            return [row[0] for row in cur.fetchall()]

    def get_student_daily_presence(self, student_id: int, days: int = 14) -> List[int]:
        """
        Returns a list of length `days` (oldest → newest) where
        1 = student had at least one attendance record that day, 0 = absent.
        Used to render the sparkline and compute the weekly trend.
        """
        from datetime import date, timedelta
        present = set(self.get_student_attendance_dates(student_id))
        today   = date.today()
        return [
            1 if (today - timedelta(days=i)).isoformat() in present else 0
            for i in range(days - 1, -1, -1)
        ]

    def get_student_scan_log(self, student_id: int,
                             limit: int = 60) -> List[sqlite3.Row]:
        """Recent attendance records for the scan-log tab."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    al.id,
                    c.course_code,
                    c.course_name,
                    se.name       AS session,
                    al.timestamp,
                    al.confidence
                FROM attendance_logs al
                LEFT JOIN courses  c  ON c.id  = al.course_id
                LEFT JOIN sessions se ON se.id = al.session_id
                WHERE al.student_id = ?
                ORDER BY al.timestamp DESC
                LIMIT ?
                """,
                (student_id, limit),
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
