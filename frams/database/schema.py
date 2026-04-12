"""
T4 — Database Schema
Defines all CREATE TABLE and CREATE INDEX statements for FRAMS.

Table relationships:
    students ──< attendance_logs >── courses
                        │
                     sessions

    system_settings  (standalone key-value store)
"""

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

CREATE_STUDENTS = """
CREATE TABLE IF NOT EXISTS students (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    face_label   INTEGER UNIQUE NOT NULL,   -- Integer ID used by LBPH trainer
    name         TEXT    NOT NULL,
    matric_no    TEXT    UNIQUE NOT NULL,
    department   TEXT    NOT NULL DEFAULT 'Computer Engineering',
    enrolled_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    is_active    INTEGER NOT NULL DEFAULT 1 -- 1 = active, 0 = soft-deleted
);
"""

CREATE_COURSES = """
CREATE TABLE IF NOT EXISTS courses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT    UNIQUE NOT NULL,   -- e.g. CPE 401
    course_name TEXT    NOT NULL,          -- e.g. Embedded Systems Design
    department  TEXT    NOT NULL DEFAULT 'Computer Engineering',
    semester    TEXT    NOT NULL DEFAULT 'First',  -- First | Second
    is_active   INTEGER NOT NULL DEFAULT 1
);
"""

CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT    UNIQUE NOT NULL            -- Morning | Afternoon | Evening
);
"""

CREATE_ATTENDANCE_LOGS = """
CREATE TABLE IF NOT EXISTS attendance_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    INTEGER NOT NULL REFERENCES students(id)  ON DELETE CASCADE,
    course_id     INTEGER          REFERENCES courses(id)   ON DELETE SET NULL,
    session_id    INTEGER          REFERENCES sessions(id)  ON DELETE SET NULL,
    timestamp     TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    confidence    REAL    NOT NULL,         -- LBPH dissimilarity score (lower = better)
    is_synced     INTEGER NOT NULL DEFAULT 0,   -- 0 = pending upload, 1 = synced
    synced_at     TEXT                          -- ISO timestamp of successful sync
);
"""

CREATE_SYSTEM_SETTINGS = """
CREATE TABLE IF NOT EXISTS system_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Indexes  (added after tables to keep CREATE TABLE blocks readable)
# ---------------------------------------------------------------------------

CREATE_INDEXES = [
    # Attendance queries filter heavily on date and student
    "CREATE INDEX IF NOT EXISTS idx_attendance_student   ON attendance_logs(student_id);",
    "CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance_logs(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_attendance_course    ON attendance_logs(course_id);",
    # Sync module only needs unsynced rows
    "CREATE INDEX IF NOT EXISTS idx_attendance_synced    ON attendance_logs(is_synced);",
    # Enrollment lookups by matric number
    "CREATE INDEX IF NOT EXISTS idx_students_matric      ON students(matric_no);",
]

# ---------------------------------------------------------------------------
# Ordered list used by db_init.py
# ---------------------------------------------------------------------------

ALL_TABLES = [
    CREATE_STUDENTS,
    CREATE_COURSES,
    CREATE_SESSIONS,
    CREATE_ATTENDANCE_LOGS,
    CREATE_SYSTEM_SETTINGS,
]
