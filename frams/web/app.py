"""Flask application factory for FRAMS."""
import config
from flask import Flask
from web.routes import dashboard, enrollment, attendance, courses, settings, stream


def create_app(app_state=None) -> Flask:
    """
    Create and configure the Flask application.

    Parameters
    ----------
    app_state : AppState, optional
        Shared state object from main.py.  Stored in app.config['APP_STATE']
        so all route handlers can access it via current_app.config.
    """
    app = Flask(__name__)
    app.secret_key = config.FLASK_SECRET_KEY

    if app_state is not None:
        app.config['APP_STATE'] = app_state

    app.register_blueprint(dashboard.bp)
    app.register_blueprint(enrollment.bp, url_prefix='/enrollment')
    app.register_blueprint(attendance.bp, url_prefix='/attendance')
    app.register_blueprint(courses.bp,    url_prefix='/courses')
    app.register_blueprint(settings.bp,   url_prefix='/settings')
    app.register_blueprint(stream.bp,     url_prefix='/stream')

    return app
