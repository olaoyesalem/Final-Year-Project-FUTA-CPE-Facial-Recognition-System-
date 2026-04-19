"""Course management."""
import io, csv
from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, flash, request, Response
from database.db_manager import DatabaseManager

bp = Blueprint('courses', __name__)


@bp.route('/')
def index():
    db = DatabaseManager()
    return render_template('courses.html', courses=db.list_courses(active_only=False))


@bp.route('/add', methods=['POST'])
def add():
    code = request.form.get('course_code', '').strip()
    name = request.form.get('course_name', '').strip()
    dept = request.form.get('department', 'Computer Engineering').strip()
    sem  = request.form.get('semester', 'First').strip()

    if not code or not name:
        flash('Course code and name are required.', 'warning')
        return redirect(url_for('courses.index'))

    db = DatabaseManager()
    try:
        db.add_course(code, name, dept, sem)
        flash(f'Course {code} added.', 'success')
    except Exception as exc:
        flash(f'Error: {exc}', 'danger')
    return redirect(url_for('courses.index'))


@bp.route('/<int:course_id>')
def detail(course_id):
    db     = DatabaseManager()
    course = db.get_course_by_id(course_id)
    if not course:
        flash('Course not found.', 'danger')
        return redirect(url_for('courses.index'))

    rows   = db.get_course_student_stats(course_id)
    totals = db.get_course_daily_totals(course_id)

    # Threshold from query string (default 75 %)
    threshold = request.args.get('threshold', 75, type=int)

    below = [r for r in rows if r['pct'] < threshold]

    return render_template(
        'course_detail.html',
        course=course,
        rows=rows,
        below=below,
        threshold=threshold,
        totals=totals,
    )


@bp.route('/<int:course_id>/export')
def export(course_id):
    db     = DatabaseManager()
    course = db.get_course_by_id(course_id)
    if not course:
        return "Course not found.", 404

    rows      = db.get_course_student_stats(course_id)
    threshold = request.args.get('threshold', 75, type=int)

    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(['FRAMS Course Attendance Report'])
    w.writerow(['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    w.writerow([])
    w.writerow(['Course Code', course['course_code']])
    w.writerow(['Course Name', course['course_name']])
    w.writerow(['Department',  course['department']])
    w.writerow(['Semester',    course['semester']])
    w.writerow(['Threshold',   f'{threshold}%'])
    w.writerow([])
    w.writerow(['Rank', 'Student Name', 'Matric No', 'Attended', 'Total Classes', '%', 'Status'])
    for i, r in enumerate(rows, 1):
        status = 'AT RISK' if r['pct'] < threshold else 'OK'
        w.writerow([i, r['student_name'], r['matric_no'],
                    r['attended'], r['total'], int(r['pct'] or 0), status])

    buf.seek(0)
    fname = f"FRAMS_{course['course_code'].replace(' ','_')}_Attendance.csv"
    return Response(buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename="{fname}"'})


@bp.route('/<int:course_id>/delete', methods=['POST'])
def delete(course_id):
    db = DatabaseManager()
    course = db.get_course_by_id(course_id)
    if not course:
        flash('Course not found.', 'danger')
        return redirect(url_for('courses.index'))
    with db._cursor() as cur:
        cur.execute("UPDATE courses SET is_active = 0 WHERE id = ?", (course_id,))
    flash(f'Course {course["course_code"]} deactivated.', 'success')
    return redirect(url_for('courses.index'))
