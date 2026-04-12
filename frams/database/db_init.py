"""
T6 — Database Initialiser
Creates frams.db, applies the schema, and seeds default reference data.

Run once before first use:
    python -m database.db_init

Safe to re-run — all CREATE statements use IF NOT EXISTS, and seed data
uses INSERT OR IGNORE so existing rows are never overwritten.
"""

import logging
import os
import sqlite3
import sys

# Allow running as `python -m database.db_init` from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from database.schema import ALL_TABLES, CREATE_INDEXES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default seed data
# ---------------------------------------------------------------------------

DEFAULT_SESSIONS = [
    ("Morning",),
    ("Afternoon",),
    ("Evening",),
]

DEFAULT_COURSES = [
    ("CPE 401", "Embedded Systems Design",          "Computer Engineering", "First"),
    ("CPE 403", "Computer Architecture",             "Computer Engineering", "First"),
    ("CPE 405", "Digital Signal Processing",         "Computer Engineering", "First"),
    ("CPE 407", "Software Engineering",              "Computer Engineering", "First"),
    ("CPE 499", "Final Year Project",                "Computer Engineering", "Second"),
]

DEFAULT_SETTINGS = [
    ("recognition_threshold", str(config.RECOGNITION_THRESHOLD)),
    ("duplicate_window_seconds", str(config.DUPLICATE_WINDOW_SECONDS)),
    ("liveness_enabled", str(int(config.LIVENESS_ENABLED))),
    ("dataset_images_per_student", str(config.DATASET_IMAGES_PER_STUDENT)),
    ("system_version", "1.0.0"),
]


# ---------------------------------------------------------------------------
# Initialisation routine
# ---------------------------------------------------------------------------

def initialise_database(db_path: str = config.DATABASE_PATH) -> None:
    """
    Create the database file (if absent), apply schema, seed reference data.
    Idempotent — safe to call on every application startup.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_existed = os.path.isfile(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    try:
        with conn:
            # 1. Create tables
            for ddl in ALL_TABLES:
                conn.execute(ddl)
            logger.info("Tables verified.")

            # 2. Create indexes
            for idx_sql in CREATE_INDEXES:
                conn.execute(idx_sql)
            logger.info("Indexes verified.")

            # 3. Seed sessions
            conn.executemany(
                "INSERT OR IGNORE INTO sessions (name) VALUES (?)",
                DEFAULT_SESSIONS,
            )

            # 4. Seed courses
            conn.executemany(
                """
                INSERT OR IGNORE INTO courses
                    (course_code, course_name, department, semester)
                VALUES (?, ?, ?, ?)
                """,
                DEFAULT_COURSES,
            )

            # 5. Seed system settings (never overwrite user-changed values)
            conn.executemany(
                "INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)",
                DEFAULT_SETTINGS,
            )

        action = "Opened existing" if db_existed else "Created new"
        logger.info("%s database at: %s", action, db_path)
        _verify(conn)

    finally:
        conn.close()


def _verify(conn: sqlite3.Connection) -> None:
    """Log a quick sanity check — counts rows in each table."""
    tables = [
        "students", "courses", "sessions",
        "attendance_logs", "system_settings",
    ]
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info("  %-22s %d row(s)", table, count)


def reset_database(db_path: str = config.DATABASE_PATH) -> None:
    """
    DESTRUCTIVE — drops and recreates the database file.
    Only call this during development / testing.
    """
    if os.path.isfile(db_path):
        os.remove(db_path)
        logger.warning("Deleted existing database: %s", db_path)
    initialise_database(db_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FRAMS database initialiser")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DROP and recreate the database (all data will be lost!)",
    )
    args = parser.parse_args()

    if args.reset:
        confirm = input(
            "WARNING: This will delete all data. Type YES to confirm: "
        )
        if confirm.strip() == "YES":
            reset_database()
        else:
            print("Aborted.")
    else:
        initialise_database()
