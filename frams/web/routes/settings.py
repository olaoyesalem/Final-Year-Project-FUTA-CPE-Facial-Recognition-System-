"""System settings — recognition threshold and sync controls."""
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from database.db_manager import DatabaseManager
import config

bp = Blueprint('settings', __name__)


@bp.route('/', methods=['GET'])
def index():
    db = DatabaseManager()
    threshold  = float(db.get_setting('recognition_threshold', str(config.RECOGNITION_THRESHOLD)))
    app_state  = current_app.config.get('APP_STATE')
    mode       = app_state.get_mode() if app_state else 'unknown'
    return render_template('settings.html',
                           threshold=threshold,
                           dup_window=config.DUPLICATE_WINDOW_SECONDS,
                           sync_enabled=config.SYNC_ENABLED,
                           sync_url=config.SYNC_REMOTE_URL,
                           mode=mode)


@bp.route('/', methods=['POST'])
def update():
    db = DatabaseManager()
    raw = request.form.get('recognition_threshold', '')
    try:
        threshold = float(raw)
        if not (20.0 <= threshold <= 150.0):
            raise ValueError
        db.set_setting('recognition_threshold', str(threshold))
        flash(f'Recognition threshold updated to {threshold}.', 'success')
    except (ValueError, TypeError):
        flash('Threshold must be a number between 20 and 150.', 'danger')
    return redirect(url_for('settings.index'))


@bp.route('/sync_now', methods=['POST'])
def sync_now():
    """Force an immediate sync — JSON response for AJAX call."""
    try:
        from sync.sync_manager import SyncManager
        result = SyncManager().sync_now()
        return jsonify(ok=True, **result)
    except Exception as exc:
        return jsonify(ok=False, error=str(exc)), 500
