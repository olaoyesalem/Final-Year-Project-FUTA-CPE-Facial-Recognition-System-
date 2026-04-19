"""
T21 — Main Integration Loop
Entry point for FRAMS on Raspberry Pi.

Start order
-----------
1. Logging
2. Database init
3. Hardware: Camera, RTC, LCD, GPIO
4. Recognition pipeline: FaceDetector, Recognizer, LivenessDetector
5. Camera feed thread  → keeps FrameBuffer current
6. Flask web server    → background thread
7. Sync manager        → background thread
8. Attendance loop     → main thread (button → detect → recognise → log)
"""

import logging
import os
import sys
import threading
import time

import config
from database.db_init import initialise_database
from database.db_manager import DatabaseManager
from hardware.camera import Camera, CameraError
from hardware.gpio_handler import GPIOHandler
from hardware.lcd import LCD
from hardware.rtc import RTC
from recognition.face_detector import FaceDetector
from recognition.liveness import LivenessDetector
from recognition.recognizer import Recognizer, RecognizerError
from sync.sync_manager import SyncManager
from sync.downloader import Downloader
from web.app import create_app
from web.state import AppState


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    handlers = [logging.FileHandler(config.LOG_FILE)]
    if config.LOG_TO_CONSOLE:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

def _camera_feed(camera: Camera, app_state: AppState,
                 stop_event: threading.Event) -> None:
    """Continuously capture frames into the shared FrameBuffer."""
    logger = logging.getLogger("camera-feed")
    while not stop_event.is_set():
        try:
            frame = camera.capture_frame()
            # During enrollment the on_frame callback owns the JPEG overlay;
            # always update the raw frame so DatasetCapture can read it.
            if app_state.get_mode() == "enrollment":
                app_state.frame_buffer._raw = frame   # raw only, no JPEG overwrite
            else:
                app_state.frame_buffer.put(frame)
        except CameraError as exc:
            logger.warning("Frame capture error: %s", exc)
            time.sleep(0.05)
        except Exception as exc:
            logger.error("Unexpected camera error: %s", exc)
            time.sleep(0.1)


def _flask_server(app, host: str, port: int) -> None:
    app.run(host=host, port=port, threaded=True, use_reloader=False)


# ---------------------------------------------------------------------------
# Liveness helper — collects frames until a decision is reached
# ---------------------------------------------------------------------------

def _check_liveness(
    app_state: AppState,
    detector: FaceDetector,
    liveness: LivenessDetector,
    max_frames: int = None,
) -> bool:
    if not config.LIVENESS_ENABLED:
        return True

    max_frames = max_frames or (config.LIVENESS_WINDOW_FRAMES + 10)
    liveness.reset()

    for _ in range(max_frames):
        frame = app_state.frame_buffer.get_raw()
        if frame is None:
            time.sleep(0.05)
            continue
        rect = detector.detect_largest(frame)
        if rect is None:
            time.sleep(0.033)
            continue
        decision = liveness.update(frame, rect)
        if decision is not None:
            return decision
        time.sleep(0.033)

    # Ran out of frames before window filled — treat as pass
    return True


# ---------------------------------------------------------------------------
# Attendance scan — one full recognition attempt
# ---------------------------------------------------------------------------

def _run_attendance_scan(
    app_state: AppState,
    db: DatabaseManager,
    detector: FaceDetector,
    recognizer: Recognizer,
    liveness: LivenessDetector,
    lcd: LCD,
    gpio: GPIOHandler,
) -> None:
    logger = logging.getLogger("scan")
    lcd.show(*config.LCD_MSG_SCANNING)

    frame = app_state.frame_buffer.get_raw()
    if frame is None:
        lcd.show(*config.LCD_MSG_NO_FACE)
        gpio.beep_failure()
        time.sleep(1)
        return

    # Face detection
    rect = detector.detect_largest(frame)
    if rect is None:
        lcd.show(*config.LCD_MSG_NO_FACE)
        gpio.beep_failure()
        time.sleep(1)
        return

    # Liveness check (collects window of frames)
    if config.LIVENESS_ENABLED:
        lcd.show("Liveness Check", "Blink please…")
        if not _check_liveness(app_state, detector, liveness):
            lcd.show("Liveness Failed", "Try Again")
            gpio.beep_failure()
            time.sleep(2)
            return

    # Recognition
    if not recognizer.is_loaded:
        lcd.show("No Model", "Enroll First")
        time.sleep(2)
        return

    # Refresh frame after liveness window
    latest = app_state.frame_buffer.get_raw()
    if latest is not None:
        frame = latest
    rect  = detector.detect_largest(frame)
    if rect is None:
        lcd.show(*config.LCD_MSG_NO_FACE)
        time.sleep(1)
        return

    try:
        roi = detector.crop_face(frame, rect)
    except ValueError:
        lcd.show(*config.LCD_MSG_NO_FACE)
        time.sleep(1)
        return

    threshold = db.get_recognition_threshold()
    result    = recognizer.predict_roi(roi, threshold_override=threshold)

    if not result.is_recognized:
        logger.info("Unknown face (confidence=%.2f, threshold=%.2f).",
                    result.confidence, result.threshold)
        lcd.show(*config.LCD_MSG_UNKNOWN)
        gpio.beep_failure()
        time.sleep(2)
        return

    student = db.get_student_by_label(result.face_label)
    if student is None:
        logger.warning("face_label=%d not in DB.", result.face_label)
        lcd.show(*config.LCD_MSG_UNKNOWN)
        gpio.beep_failure()
        time.sleep(2)
        return

    log_id = db.log_attendance(student["id"], result.confidence)
    if log_id is None:
        lcd.show(*config.LCD_MSG_DUPLICATE)
        gpio.beep_failure()
        logger.info("Duplicate attendance blocked for %s.", student["name"])
    else:
        first_name = student["name"].split()[0]
        lcd.show(
            config.LCD_MSG_RECOGNIZED[0],
            config.LCD_MSG_RECOGNIZED[1].format(name=first_name),
        )
        gpio.beep_success()
        logger.info(
            "Attendance logged — %s (%s), confidence=%.2f, log_id=%d.",
            student["name"], student["matric_no"], result.confidence, log_id,
        )

    time.sleep(2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()
    logger = logging.getLogger("main")
    logger.info("=== FRAMS starting ===")

    # --- Database ---
    initialise_database()
    db = DatabaseManager()

    # --- Shared state ---
    app_state = AppState()

    # --- Hardware ---
    camera = Camera()
    camera.start()

    rtc = RTC()
    if config.RTC_SYNC_SYSTEM:
        try:
            rtc.sync_system_clock()
        except Exception as exc:
            logger.warning("RTC sync failed: %s", exc)

    lcd = LCD()
    lcd.start()
    lcd.show(*config.LCD_MSG_READY)

    gpio = GPIOHandler()
    gpio.start()

    # --- Recognition pipeline ---
    detector  = FaceDetector()
    liveness  = LivenessDetector()
    recognizer = Recognizer()
    app_state.recognizer = recognizer
    try:
        recognizer.load()
    except RecognizerError as exc:
        logger.warning("Model not loaded (train after enrollment): %s", exc)

    # --- Camera feed thread ---
    stop_feed = threading.Event()
    threading.Thread(
        target=_camera_feed,
        args=(camera, app_state, stop_feed),
        daemon=True,
        name="camera-feed",
    ).start()

    # --- Flask web server ---
    flask_app = create_app(app_state)
    flask_port = int(os.environ.get("FRAMS_PORT", config.FLASK_PORT))
    threading.Thread(
        target=_flask_server,
        args=(flask_app, config.FLASK_HOST, flask_port),
        daemon=True,
        name="flask",
    ).start()
    logger.info("Flask dashboard at http://%s:%d", config.FLASK_HOST, flask_port)

    # --- Sync manager ---
    sync = SyncManager(db)
    sync.start()

    # --- Supabase downloader (Pi ← cloud face images) ---
    downloader = Downloader(app_state)
    downloader.start()

    # --- Attendance loop ---
    app_state.set_mode("attendance")
    logger.info("Entering attendance loop. Waiting for button press…")

    try:
        while True:
            mode = app_state.get_mode()

            if mode != "attendance":
                # Yield CPU while enrollment/training is running
                time.sleep(0.1)
                continue

            lcd.show(*config.LCD_MSG_READY)
            pressed = gpio.wait_for_press(timeout=1.0)
            if not pressed:
                continue

            # Run a full scan in attendance mode only
            if app_state.get_mode() == "attendance":
                _run_attendance_scan(
                    app_state, db, detector, recognizer, liveness, lcd, gpio
                )
                time.sleep(1)  # debounce — prevents tight loop in GPIO no-op mode

    except KeyboardInterrupt:
        logger.info("Shutdown requested via Ctrl+C.")
    finally:
        stop_feed.set()
        sync.stop()
        downloader.stop()
        camera.stop()
        gpio.stop()
        lcd.stop()
        logger.info("=== FRAMS shut down ===")


if __name__ == "__main__":
    main()
