"""Dashboard — system overview and today's attendance summary."""
from flask import Blueprint, render_template, current_app
from database.db_manager import DatabaseManager

bp = Blueprint('dashboard', __name__)


@bp.route('/')
def index():
    db = DatabaseManager()
    app_state = current_app.config.get('APP_STATE')
    today_logs = db.get_today_attendance()
    return render_template(
        'dashboard.html',
        total_students=len(db.list_students()),
        total_courses=len(db.list_courses()),
        today_logs=today_logs,
        today_count=len(today_logs),
        mode=app_state.get_mode() if app_state else 'unknown',
    )
