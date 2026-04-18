"""MJPEG live camera stream endpoint."""
import time
import cv2
import numpy as np
from flask import Blueprint, Response, current_app

bp = Blueprint('stream', __name__)

_BOUNDARY = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'


def _make_placeholder() -> bytes:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, "Camera Offline",   (55,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
    cv2.putText(img, "Run main.py first", (45, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


_PLACEHOLDER = _make_placeholder()


def _generate(frame_buffer):
    while True:
        jpeg = frame_buffer.get_jpeg() or _PLACEHOLDER
        yield _BOUNDARY + jpeg + b'\r\n'
        time.sleep(1 / 15)  # ~15 fps


@bp.route('/video_feed')
def video_feed():
    app_state = current_app.config.get('APP_STATE')
    if app_state is None:
        return 'Camera not available', 503
    return Response(
        _generate(app_state.frame_buffer),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )
