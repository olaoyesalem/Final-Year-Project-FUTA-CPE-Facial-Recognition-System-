"""Attendance log — filter, view, and export."""
import io
from flask import Blueprint, render_template, request, send_file
from database.db_manager import DatabaseManager

bp = Blueprint('attendance', __name__)


@bp.route('/')
def index():
    db = DatabaseManager()
    date_from  = request.args.get('date_from', '')
    date_to    = request.args.get('date_to', '')
    course_id  = request.args.get('course_id', type=int)
    session_id = request.args.get('session_id', type=int)

    logs = db.get_attendance_filtered(
        date_from  or None,
        date_to    or None,
        course_id,
        session_id,
    )
    return render_template(
        'attendance.html',
        logs=logs,
        courses=db.list_courses(),
        sessions=db.list_sessions(),
        date_from=date_from,
        date_to=date_to,
        course_id=course_id,
        session_id=session_id,
    )


@bp.route('/export')
def export():
    import openpyxl

    db = DatabaseManager()
    date_from  = request.args.get('date_from', '') or None
    date_to    = request.args.get('date_to', '')   or None
    course_id  = request.args.get('course_id', type=int)
    session_id = request.args.get('session_id', type=int)

    rows = db.get_attendance_filtered(date_from, date_to, course_id, session_id)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Attendance'
    ws.append(['#', 'Student Name', 'Matric No', 'Course Code', 'Course Name',
               'Session', 'Timestamp', 'Confidence'])
    for i, row in enumerate(rows, 1):
        ws.append([
            i,
            row['student_name'],
            row['matric_no'],
            row['course_code'] or '-',
            row['course_name'] or '-',
            row['session']     or '-',
            row['timestamp'],
            round(float(row['confidence']), 2),
        ])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    parts = ['attendance']
    if date_from:
        parts.append(date_from)
    if date_to:
        parts.append(date_to)
    filename = '_'.join(parts) + '.xlsx'

    return send_file(
        buf,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
