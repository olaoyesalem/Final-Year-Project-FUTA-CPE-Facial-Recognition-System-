"""Student enrollment, dataset capture, and model training."""
import logging
import os
import threading

from flask import (Blueprint, jsonify, redirect, render_template,
                   request, url_for, flash, current_app)

import config
from database.db_manager import DatabaseManager
from recognition.face_detector import FaceDetector
from recognition.dataset_capture import DatasetCapture
from recognition.trainer import Trainer, TrainerError

logger = logging.getLogger(__name__)
bp = Blueprint('enrollment', __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_images(face_label: int) -> int:
    d = os.path.join(config.DATASET_DIR, str(face_label))
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith('.jpg'))


class _BufferedCamera:
    """Proxy camera that reads the latest frame from the shared FrameBuffer."""

    def __init__(self, frame_buffer):
        self._buf = frame_buffer

    def capture_frame(self):
        import time
        frame = self._buf.get_raw()
        if frame is None:
            time.sleep(0.05)
            frame = self._buf.get_raw()
        if frame is None:
            raise RuntimeError(
                "No frame in buffer — ensure main.py is running with camera active."
            )
        return frame

    # Stub lifecycle so DatasetCapture context-manager idioms work
    def start(self): pass
    def stop(self):  pass
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Background worker functions
# ---------------------------------------------------------------------------

def _run_capture(app_state, face_label: int, target: int) -> None:
    app_state.enroll_saved = 0
    app_state.enroll_total = target
    app_state.enroll_done  = False
    app_state.enroll_error = None
    app_state.enroll_stop_event.clear()
    app_state.set_mode("enrollment")

    try:
        camera   = _BufferedCamera(app_state.frame_buffer)
        detector = FaceDetector()
        capture  = DatasetCapture(camera, detector)

        def on_frame(frame, saved, total):
            app_state.frame_buffer.put_annotated(frame)
            app_state.enroll_saved = saved
            app_state.enroll_total = total

        capture.capture(
            face_label,
            target_count=target,
            on_frame=on_frame,
            stop_event=app_state.enroll_stop_event,
        )
        app_state.enroll_done = True
        logger.info("Capture complete for face_label=%d.", face_label)
    except Exception as exc:
        logger.error("Capture thread error: %s", exc)
        app_state.enroll_error = str(exc)
        app_state.enroll_done  = True
    finally:
        app_state.set_mode("attendance")


def _run_training(app_state) -> None:
    app_state.train_running = True
    app_state.train_done    = False
    app_state.train_error   = None
    app_state.train_result  = None
    app_state.set_mode("training")

    try:
        result = Trainer().train()
        app_state.train_result = result
        if app_state.recognizer is not None:
            app_state.recognizer.reload()
        logger.info(
            "Training complete — students=%d, images=%d.",
            result.num_students, result.num_images,
        )
    except TrainerError as exc:
        logger.error("Training error: %s", exc)
        app_state.train_error = str(exc)
    except Exception as exc:
        logger.error("Training unexpected error: %s", exc)
        app_state.train_error = str(exc)
    finally:
        app_state.train_running = False
        app_state.train_done    = True
        app_state.set_mode("attendance")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@bp.route('/')
def index():
    db       = DatabaseManager()
    students = db.list_students()
    counts   = {s['id']: _count_images(s['face_label']) for s in students}
    return render_template('enrollment.html', students=students, counts=counts,
                           target=config.DATASET_IMAGES_PER_STUDENT)


@bp.route('/add', methods=['POST'])
def add():
    name   = request.form.get('name',   '').strip()
    matric = request.form.get('matric_no', '').strip()
    dept   = request.form.get('department', 'Computer Engineering').strip()

    if not name or not matric:
        flash('Name and matric number are required.', 'warning')
        return redirect(url_for('enrollment.index'))

    db = DatabaseManager()
    if db.get_student_by_matric(matric):
        flash(f'Matric number {matric} is already registered.', 'danger')
        return redirect(url_for('enrollment.index'))

    try:
        db.add_student(name, matric, dept)
        flash(f'Student {name} ({matric}) registered successfully.', 'success')
    except Exception as exc:
        flash(f'Database error: {exc}', 'danger')
    return redirect(url_for('enrollment.index'))


@bp.route('/<int:student_id>/delete', methods=['POST'])
def delete(student_id):
    db      = DatabaseManager()
    student = db.get_student_by_id(student_id)
    if not student:
        flash('Student not found.', 'danger')
        return redirect(url_for('enrollment.index'))
    db.deactivate_student(student_id)
    flash(f'Student {student["name"]} deactivated.', 'success')
    return redirect(url_for('enrollment.index'))


# -- Capture page --

@bp.route('/capture/<int:student_id>')
def capture_page(student_id):
    db      = DatabaseManager()
    student = db.get_student_by_id(student_id)
    if not student:
        flash('Student not found.', 'danger')
        return redirect(url_for('enrollment.index'))
    existing = _count_images(student['face_label'])
    return render_template('capture.html', student=student, existing=existing,
                           target=config.DATASET_IMAGES_PER_STUDENT)


@bp.route('/capture/<int:student_id>/start', methods=['POST'])
def capture_start(student_id):
    app_state = current_app.config.get('APP_STATE')
    if app_state is None:
        return jsonify(ok=False, error='System not ready (no app state)'), 503

    if app_state.get_mode() == 'enrollment':
        return jsonify(ok=False, error='A capture session is already running'), 409

    db      = DatabaseManager()
    student = db.get_student_by_id(student_id)
    if not student:
        return jsonify(ok=False, error='Student not found'), 404

    face_label = student['face_label']
    target     = config.DATASET_IMAGES_PER_STUDENT

    app_state.enroll_face_label = face_label
    threading.Thread(
        target=_run_capture,
        args=(app_state, face_label, target),
        daemon=True,
        name='capture-thread',
    ).start()

    return jsonify(ok=True, face_label=face_label, target=target)


@bp.route('/capture/status')
def capture_status():
    app_state = current_app.config.get('APP_STATE')
    if app_state is None:
        return jsonify(ok=False, error='System not ready'), 503
    return jsonify(
        ok=True,
        mode=app_state.get_mode(),
        saved=app_state.enroll_saved,
        total=app_state.enroll_total,
        done=app_state.enroll_done,
        error=app_state.enroll_error,
    )


@bp.route('/capture/stop', methods=['POST'])
def capture_stop():
    app_state = current_app.config.get('APP_STATE')
    if app_state:
        app_state.enroll_stop_event.set()
    return jsonify(ok=True)


# -- Training --

@bp.route('/train', methods=['POST'])
def train():
    app_state = current_app.config.get('APP_STATE')
    if app_state is None:
        return jsonify(ok=False, error='System not ready'), 503
    if app_state.train_running:
        return jsonify(ok=False, error='Training already in progress'), 409

    threading.Thread(
        target=_run_training,
        args=(app_state,),
        daemon=True,
        name='trainer-thread',
    ).start()

    return jsonify(ok=True)


@bp.route('/train/status')
def train_status():
    app_state = current_app.config.get('APP_STATE')
    if app_state is None:
        return jsonify(ok=False, error='System not ready'), 503

    result = None
    if app_state.train_result:
        r = app_state.train_result
        result = {'num_students': r.num_students, 'num_images': r.num_images,
                  'model_path': r.model_path}

    return jsonify(
        ok=True,
        running=app_state.train_running,
        done=app_state.train_done,
        error=app_state.train_error,
        result=result,
    )
