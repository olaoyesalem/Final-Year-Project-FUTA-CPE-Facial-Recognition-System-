"""
T10 — GPIO Handler
Push-button input and buzzer output via gpiozero (preferred) or RPi.GPIO fallback.

Provides:
  - wait_for_press(timeout)  → bool   block until button pressed (or timeout)
  - is_pressed()             → bool   non-blocking button state
  - beep(duration)                    activate buzzer for `duration` seconds
  - beep_success()                    short confirmation beep
  - beep_failure()                    longer rejection beep

On non-Pi machines (gpiozero / RPi.GPIO unavailable) every GPIO operation
is silently skipped and logged so the rest of the system runs in dev mode.

Usage
-----
    from hardware.gpio_handler import GPIOHandler

    with GPIOHandler() as gpio:
        gpio.beep_success()
        if gpio.wait_for_press(timeout=10):
            print("Button pressed!")
"""

import logging
import time

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional GPIO imports — gpiozero preferred, RPi.GPIO as fallback
# ---------------------------------------------------------------------------
try:
    from gpiozero import Button as _GZButton, OutputDevice as _GZOutput
    _GPIOZERO_AVAILABLE = True
except ImportError:
    _GPIOZERO_AVAILABLE = False

try:
    import RPi.GPIO as _GPIO
    _RPIGPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    _RPIGPIO_AVAILABLE = False


class GPIOHandler:
    """
    Manages the push button (input) and buzzer (output).

    Backend selection priority:
        1. gpiozero  (cleanest API, best for Pi OS Bullseye/Bookworm)
        2. RPi.GPIO  (legacy fallback)
        3. No-op log mode (dev machines without GPIO)
    """

    def __init__(
        self,
        button_pin: int = config.GPIO_BUTTON_PIN,
        buzzer_pin: int = config.GPIO_BUZZER_PIN,
        success_duration: float = config.BUZZER_SUCCESS_DURATION,
        failure_duration: float = config.BUZZER_FAILURE_DURATION,
    ) -> None:
        self._button_pin = button_pin
        self._buzzer_pin = buzzer_pin
        self._success_dur = success_duration
        self._failure_dur = failure_duration

        # gpiozero objects
        self._gz_button = None
        self._gz_buzzer = None

        # RPi.GPIO mode flag
        self._rpigpio_ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Set up GPIO pins.  Call once before using button/buzzer methods."""
        if _GPIOZERO_AVAILABLE:
            self._start_gpiozero()
        elif _RPIGPIO_AVAILABLE:
            self._start_rpigpio()
        else:
            logger.warning(
                "No GPIO library available (gpiozero / RPi.GPIO). "
                "GPIO operations will be no-ops. "
                "Run: pip install gpiozero  (or: pip install -r requirements.txt)"
            )

    def stop(self) -> None:
        """Release GPIO resources."""
        if self._gz_button is not None:
            self._gz_button.close()
            self._gz_button = None
        if self._gz_buzzer is not None:
            self._gz_buzzer.off()
            self._gz_buzzer.close()
            self._gz_buzzer = None

        if self._rpigpio_ready:
            try:
                _GPIO.cleanup()
            except Exception as exc:
                logger.debug("RPi.GPIO cleanup error (ignored): %s", exc)
            self._rpigpio_ready = False

        logger.debug("GPIO released.")

    # ------------------------------------------------------------------
    # Button
    # ------------------------------------------------------------------

    def is_pressed(self) -> bool:
        """
        Non-blocking check of the button state.

        Returns
        -------
        bool
            True if the button is currently held down.
        """
        if self._gz_button is not None:
            return self._gz_button.is_pressed

        if self._rpigpio_ready:
            # Button is active-LOW (pull-up), so LOW == pressed
            return _GPIO.input(self._button_pin) == _GPIO.LOW

        return False   # no-op mode

    def wait_for_press(self, timeout: float = 0.0) -> bool:
        """
        Block until the button is pressed.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait.  0 (default) = wait forever.

        Returns
        -------
        bool
            True if the button was pressed, False if timeout expired.
        """
        logger.debug("Waiting for button press (timeout=%.1fs)…", timeout)

        if self._gz_button is not None:
            # gpiozero wait_for_press returns None on success, raises on timeout
            self._gz_button.wait_for_press(timeout=timeout if timeout > 0 else None)
            pressed = self._gz_button.is_pressed
            logger.debug("Button %s.", "pressed" if pressed else "timeout")
            return pressed

        if self._rpigpio_ready:
            deadline = time.monotonic() + timeout if timeout > 0 else None
            while True:
                if _GPIO.input(self._button_pin) == _GPIO.LOW:
                    logger.debug("Button pressed.")
                    return True
                if deadline is not None and time.monotonic() >= deadline:
                    logger.debug("Button wait timed out.")
                    return False
                time.sleep(0.02)   # 20 ms poll interval

        # No-op mode — simulate immediate press so the main loop isn't blocked
        logger.debug("GPIO no-op: simulating button press.")
        return True

    # ------------------------------------------------------------------
    # Buzzer
    # ------------------------------------------------------------------

    def beep(self, duration: float) -> None:
        """
        Activate the buzzer for `duration` seconds.

        Parameters
        ----------
        duration : float
            How long to sound the buzzer (seconds).
        """
        logger.debug("Buzzer ON for %.2fs.", duration)

        if self._gz_buzzer is not None:
            self._gz_buzzer.on()
            time.sleep(duration)
            self._gz_buzzer.off()
            return

        if self._rpigpio_ready:
            _GPIO.output(self._buzzer_pin, _GPIO.HIGH)
            time.sleep(duration)
            _GPIO.output(self._buzzer_pin, _GPIO.LOW)
            return

        # No-op mode
        time.sleep(duration)   # keep timing realistic in tests

    def beep_success(self) -> None:
        """Short beep — played on successful face recognition."""
        self.beep(self._success_dur)

    def beep_failure(self) -> None:
        """Longer beep — played on unknown face or duplicate attendance."""
        self.beep(self._failure_dur)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GPIOHandler":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_gpiozero(self) -> None:
        try:
            # pull_up=True: button connects GPIO 18 → GND (active LOW)
            self._gz_button = _GZButton(self._button_pin, pull_up=True)
            # active_high=True: buzzer activates on HIGH
            self._gz_buzzer = _GZOutput(self._buzzer_pin, active_high=True,
                                        initial_value=False)
            logger.info(
                "gpiozero ready — button=BCM%d, buzzer=BCM%d.",
                self._button_pin, self._buzzer_pin,
            )
        except Exception as exc:
            logger.warning("gpiozero init failed (%s) — GPIO will be no-op.", exc)
            self._gz_button = None
            self._gz_buzzer = None

    def _start_rpigpio(self) -> None:
        try:
            _GPIO.setmode(_GPIO.BCM)
            _GPIO.setwarnings(False)
            _GPIO.setup(self._button_pin, _GPIO.IN, pull_up_down=_GPIO.PUD_UP)
            _GPIO.setup(self._buzzer_pin, _GPIO.OUT, initial=_GPIO.LOW)
            self._rpigpio_ready = True
            logger.info(
                "RPi.GPIO ready — button=BCM%d, buzzer=BCM%d.",
                self._button_pin, self._buzzer_pin,
            )
        except Exception as exc:
            logger.warning("RPi.GPIO init failed (%s) — GPIO will be no-op.", exc)
            self._rpigpio_ready = False
