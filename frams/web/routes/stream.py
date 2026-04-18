"""MJPEG live camera stream endpoint."""
import time
from flask import Blueprint, Response, current_app

bp = Blueprint('stream', __name__)

_BOUNDARY = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'


def _generate(frame_buffer):
    while True:
        jpeg = frame_buffer.get_jpeg()
        if jpeg:
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
