"""
FRAMS - Facial Recognition Attendance Management System
Central configuration. All tunable parameters live here.
Edit this file to adapt the system to your hardware wiring and preferences.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH   = os.path.join(BASE_DIR, "database", "frams.db")
MODEL_PATH      = os.path.join(BASE_DIR, "models",   "trainer.yml")
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
CASCADE_PATH    = os.path.join(BASE_DIR, "cascades", "haarcascade_frontalface_default.xml")
LOG_FILE        = os.path.join(BASE_DIR, "logs",     "frams.log")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL       = "DEBUG"   # DEBUG | INFO | WARNING | ERROR
LOG_TO_CONSOLE  = True

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_WIDTH        = 640
CAMERA_HEIGHT       = 480
CAMERA_FRAMERATE    = 30
# Use index 0 for USB webcam / OpenCV fallback on non-Pi hardware.
# picamera2 ignores this; it auto-selects the Pi Camera Module.
CAMERA_INDEX        = 0

# ---------------------------------------------------------------------------
# Face Detection (Haar Cascade)
# ---------------------------------------------------------------------------
HAAR_SCALE_FACTOR   = 1.3   # How much the image is scaled down at each step
HAAR_MIN_NEIGHBORS  = 5     # Minimum neighbours for a positive detection
HAAR_MIN_SIZE       = (60, 60)  # Smallest face rectangle to detect (px)

# Path to the eye cascade used by the liveness detector (T15)
EYE_CASCADE_PATH    = os.path.join(BASE_DIR, "cascades", "haarcascade_eye.xml")

# Normalised face ROI size fed to LBPH (must be consistent across capture & recognition)
FACE_ROI_SIZE       = (100, 100)

# ---------------------------------------------------------------------------
# Face Recognition (LBPH)
# ---------------------------------------------------------------------------
# Confidence is a *dissimilarity* score — lower means more similar.
# A recognised face must score BELOW this threshold.
RECOGNITION_THRESHOLD   = 70.0  # Recommended range: 50–85

LBPH_RADIUS     = 1   # Radius of the circular LBP pattern
LBPH_NEIGHBORS  = 8   # Number of sample points on the circle
LBPH_GRID_X     = 8   # Columns in the spatial histogram grid
LBPH_GRID_Y     = 8   # Rows in the spatial histogram grid

# ---------------------------------------------------------------------------
# Dataset Capture (Enrollment)
# ---------------------------------------------------------------------------
DATASET_IMAGES_PER_STUDENT  = 30   # Images captured per enrollment session
CAPTURE_DELAY_MS            = 200  # Milliseconds between successive captures

# ---------------------------------------------------------------------------
# Liveness Detection
# ---------------------------------------------------------------------------
LIVENESS_ENABLED        = True
# Blink-based liveness: minimum eye-blink frames required in the window
LIVENESS_BLINK_FRAMES   = 2
LIVENESS_WINDOW_FRAMES  = 40   # Frames to inspect for blink activity
EYE_AR_THRESHOLD        = 0.25  # Eye Aspect Ratio below which eye is "closed"

# ---------------------------------------------------------------------------
# Attendance Logic
# ---------------------------------------------------------------------------
# Prevent a student from being logged more than once within this window.
DUPLICATE_WINDOW_SECONDS    = 3600   # 1 hour
# Session names used in the course/session table
SESSIONS = ["Morning", "Afternoon", "Evening"]

# ---------------------------------------------------------------------------
# GPIO Pin Assignments  (BCM numbering)
# ---------------------------------------------------------------------------
GPIO_BUTTON_PIN     = 18   # Tactile push button (active LOW with pull-up)
GPIO_BUZZER_PIN     = 23   # Active buzzer

# Buzzer feedback durations (seconds)
BUZZER_SUCCESS_DURATION = 0.2   # Short beep on successful recognition
BUZZER_FAILURE_DURATION = 0.5   # Longer beep on unknown face

# ---------------------------------------------------------------------------
# LCD Display (I2C 16x02)
# ---------------------------------------------------------------------------
LCD_I2C_ADDRESS     = 0x27   # Common address for PCF8574-based backpacks
LCD_I2C_BUS         = 1      # /dev/i2c-1 on Raspberry Pi
LCD_COLS            = 16
LCD_ROWS            = 2
LCD_BACKLIGHT       = True

# Messages shown on the LCD
LCD_MSG_READY       = ("FRAMS Ready", "Press button...")
LCD_MSG_SCANNING    = ("Scanning...", "Please wait")
LCD_MSG_RECOGNIZED  = ("Welcome!", "{name}")          # {name} replaced at runtime
LCD_MSG_UNKNOWN     = ("Face Unknown", "See Admin")
LCD_MSG_DUPLICATE   = ("Already Logged", "Today")
LCD_MSG_NO_FACE     = ("No Face Found", "Try Again")

# ---------------------------------------------------------------------------
# DS3231 Real-Time Clock (I2C)
# ---------------------------------------------------------------------------
RTC_I2C_ADDRESS     = 0x68
RTC_I2C_BUS         = 1
# Set to True to sync Pi system clock from RTC on startup
RTC_SYNC_SYSTEM     = True

# ---------------------------------------------------------------------------
# Flask Web Dashboard
# ---------------------------------------------------------------------------
FLASK_HOST          = "0.0.0.0"   # Listen on all interfaces (LAN access)
FLASK_PORT          = 5000
FLASK_DEBUG         = False        # Never True in production
FLASK_SECRET_KEY    = "change-this-to-a-random-secret-key"

# MJPEG stream
STREAM_FPS          = 15   # Frames per second for live enrollment preview

# ---------------------------------------------------------------------------
# Export / Reports
# ---------------------------------------------------------------------------
EXPORT_DIR          = os.path.join(BASE_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Network Synchronisation (T22)
# ---------------------------------------------------------------------------
SYNC_ENABLED            = True
SYNC_REMOTE_URL         = ""       # Set to departmental server endpoint
SYNC_INTERVAL_SECONDS   = 300      # Check every 5 minutes
SYNC_TIMEOUT_SECONDS    = 10
SYNC_RETRY_ATTEMPTS     = 3
