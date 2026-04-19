"""Student attendance profile, export, and compare."""
import csv
import io
import logging
from datetime import datetime
from typing import Optional

from flask import Blueprint, Response, jsonify, render_template, request

from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)
bp = Blueprint('student', __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_profile(student_id: int, db: DatabaseManager) -> Optional[dict]:
    student = db.get_student_by_id(student_id)
    if not student:
        return None

    stats    = db.get_student_attendance_stats(student_id)
    streak   = db.get_student_streak(student_id)
    courses  = db.get_student_course_stats(student_id)
    dates    = db.get_student_attendance_dates(student_id)
    log      = db.get_student_scan_log(student_id)
    presence = db.get_student_daily_presence(student_id, days=14)

    # Trend: compare last 7 days vs previous 7 days
    recent = sum(presence[7:])
    prior  = sum(presence[:7])
    if recent > prior:
        trend = "up"
    elif recent < prior:
        trend = "down"
    else:
        trend = "flat"

    initials = ''.join(p[0].upper() for p in student['name'].split()[:2])

    return dict(
        student={
            'id':         student['id'],
            'name':       student['name'],
            'matric_no':  student['matric_no'],
            'department': student['department'],
            'initials':   initials,
        },
        stats={
            'pct':      stats['pct'],
            'attended': stats['attended'],
            'total':    stats['total'],
            'streak':   streak,
            'presence': presence,
            'trend':    trend,
        },
        courses=[
            {
                'code':     r['course_code'],
                'name':     r['course_name'],
                'attended': r['attended'],
                'total':    r['total'],
                'pct':      int(r['pct'] or 0),
            }
            for r in courses
        ],
        dates=dates,
        scan_log=[
            {
                'id':         r['id'],
                'course':     r['course_code'] or '—',
                'session':    r['session'] or '—',
                'timestamp':  r['timestamp'],
                'confidence': round(float(r['confidence']), 2),
            }
            for r in log
        ],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route('/')
def index():
    db       = DatabaseManager()
    students = db.list_students()
    return render_template('student.html', students=students)


@bp.route('/<int:student_id>/profile')
def profile(student_id: int):
    db   = DatabaseManager()
    data = _build_profile(student_id, db)
    if data is None:
        return jsonify(ok=False, error='Student not found.'), 404
    return jsonify(ok=True, **data)


@bp.route('/<int:student_id>/export')
def export(student_id: int):
    db      = DatabaseManager()
    student = db.get_student_by_id(student_id)
    if not student:
        return "Student not found.", 404

    fmt = request.args.get('format', 'csv')

    # ── CSV ─────────────────────────────────────────────────────────────
    if fmt == 'csv':
        stats  = db.get_student_attendance_stats(student_id)
        streak = db.get_student_streak(student_id)
        log    = db.get_student_scan_log(student_id, limit=10_000)
        courses = db.get_student_course_stats(student_id)

        buf = io.StringIO()
        w   = csv.writer(buf)

        w.writerow(['FRAMS Attendance Report'])
        w.writerow(['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        w.writerow([])
        w.writerow(['Student',     student['name']])
        w.writerow(['Matric No',   student['matric_no']])
        w.writerow(['Department',  student['department']])
        w.writerow([])
        w.writerow(['Overall Attendance', f"{stats['pct']}%"])
        w.writerow(['Classes Attended', f"{stats['attended']} / {stats['total']}"])
        w.writerow(['Streak', f"{streak} day(s)"])
        w.writerow([])

        w.writerow(['Per-Course Breakdown'])
        w.writerow(['Course Code', 'Course Name', 'Attended', 'Total', '%'])
        for r in courses:
            w.writerow([r['course_code'], r['course_name'],
                        r['attended'], r['total'], int(r['pct'] or 0)])
        w.writerow([])

        w.writerow(['Full Attendance Log'])
        w.writerow(['#', 'Date & Time', 'Course', 'Session', 'Confidence'])
        for i, r in enumerate(log, 1):
            w.writerow([i, r['timestamp'], r['course_code'] or '-',
                        r['session'] or '-', round(float(r['confidence']), 2)])

        buf.seek(0)
        filename = f"FRAMS_{student['matric_no'].replace('/', '_')}_Report.csv"
        return Response(
            buf.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'},
        )

    # ── Print (browser → PDF) ────────────────────────────────────────────
    elif fmt == 'print':
        data = _build_profile(student_id, db)
        return render_template('student_report.html', **data,
                               generated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return "Unknown format.", 400
