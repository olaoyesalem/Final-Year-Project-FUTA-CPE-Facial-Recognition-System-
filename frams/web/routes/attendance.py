"""Attendance log — mark, filter, view, and export."""
import io
import logging
from datetime import datetime

from flask import (Blueprint, render_template, request,
                   send_file, jsonify, current_app)

from database.db_manager import DatabaseManager
from recognition.face_detector import FaceDetector

logger = logging.getLogger(__name__)
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
        students=db.list_students(),
        courses=db.list_courses(),
        sessions=db.list_sessions(),
        date_from=date_from,
        date_to=date_to,
        course_id=course_id,
        session_id=session_id,
    )


@bp.route('/mark', methods=['POST'])
def mark():
    data       = request.get_json(force=True)
    student_id = data.get('student_id')
    course_id  = data.get('course_id')  or None
    session_id = data.get('session_id') or None
    raw_ts     = data.get('timestamp')  or None   # 'YYYY-MM-DDTHH:MM' from datetime-local

    if not student_id:
        return jsonify(ok=False, error='No student selected.'), 400

    # Parse and normalise timestamp
    timestamp = None
    if raw_ts:
        try:
            timestamp = datetime.strptime(raw_ts, '%Y-%m-%dT%H:%M').strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            return jsonify(ok=False, error='Invalid date/time format.'), 400

    app_state = current_app.config.get('APP_STATE')
    if app_state is None:
        return jsonify(ok=False, error='System not ready.'), 503

    frame = app_state.frame_buffer.get_raw()
    if frame is None:
        return jsonify(ok=False, error='No camera frame — ensure main.py is running.'), 503

    recognizer = app_state.recognizer
    if recognizer is None or not recognizer.is_loaded:
        return jsonify(ok=False,
                       error='Model not loaded. Complete enrollment and train first.'), 503

    db      = DatabaseManager()
    student = db.get_student_by_id(student_id)
    if not student:
        return jsonify(ok=False, error='Student not found.'), 404

    detector = FaceDetector()
    rect     = detector.detect_largest(frame)
    if rect is None:
        return jsonify(ok=False,
                       error='No face detected. Position the student in front of the camera.'), 400

    try:
        roi = detector.crop_face(frame, rect)
    except ValueError as exc:
        return jsonify(ok=False, error=f'Face crop failed: {exc}'), 400

    threshold = db.get_recognition_threshold()
    result    = recognizer.predict_roi(roi, threshold_override=threshold)

    if not result.is_recognized:
        return jsonify(
            ok=False,
            error=f'Face not recognised (score {result.confidence:.1f}, threshold {threshold:.1f}). '
                  'Ask the student to face the camera directly.',
        ), 400

    if result.face_label != student['face_label']:
        actual      = db.get_student_by_label(result.face_label)
        actual_name = actual['name'] if actual else 'unknown'
        return jsonify(
            ok=False,
            error=f'Face matched {actual_name}, not {student["name"]}. Wrong student?',
        ), 400

    log_id = db.log_attendance(student_id, result.confidence,
                               course_id=course_id, session_id=session_id,
                               timestamp=timestamp)
    if log_id is None:
        return jsonify(ok=False,
                       error='Attendance already recorded for this student within the hour.'), 409

    display_ts = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info("Manual attendance: %s (%s), ts=%s, confidence=%.2f, log_id=%d.",
                student['name'], student['matric_no'], display_ts, result.confidence, log_id)

    return jsonify(
        ok=True,
        message=f"Attendance recorded for {student['name']} ({student['matric_no']}) at {display_ts}.",
        confidence=round(result.confidence, 2),
        log_id=log_id,
    )


@bp.route('/export')
def export():
    import openpyxl

    db         = DatabaseManager()
    date_from  = request.args.get('date_from', '') or None
    date_to    = request.args.get('date_to', '')   or None
    course_id  = request.args.get('course_id',  type=int)
    session_id = request.args.get('session_id', type=int)

    rows = db.get_attendance_filtered(date_from, date_to, course_id, session_id)

    # Resolve names for the filename
    course_label  = ''
    session_label = ''
    if course_id:
        c = db.get_course_by_id(course_id)
        if c:
            course_label = c['course_code'].replace(' ', '')
    if session_id:
        for s in db.list_sessions():
            if s['id'] == session_id:
                session_label = s['name']
                break

    # Build descriptive filename
    parts = ['FRAMS_Attendance']
    if date_from:
        parts.append(date_from)
    if date_to:
        parts.append('to_' + date_to)
    if course_label:
        parts.append(course_label)
    if session_label:
        parts.append(session_label)
    filename = '_'.join(parts) + '.xlsx'

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Attendance'
    ws.append(['#', 'Student Name', 'Matric No', 'Course Code', 'Course Name',
               'Session', 'Date & Time', 'Confidence'])
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

    return send_file(
        buf,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
