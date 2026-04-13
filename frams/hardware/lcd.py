"""
T9 — LCD Driver
16x02 character LCD via I2C PCF8574 backpack (RPLCD library).

Provides:
  - show(line1, line2)   display two lines, auto-truncated to 16 chars
  - clear()              blank the display
  - show_message(key)    display a pre-defined message from config
  - backlight(on)        turn backlight on/off

On non-Pi machines (RPLCD / RPi.GPIO unavailable) every method silently
logs the message instead of raising, so the rest of the system keeps
running during development.

Usage
-----
    from hardware.lcd import LCD

    with LCD() as lcd:
        lcd.show("FRAMS Ready", "Press button...")
        lcd.show_message("ready")
"""

import logging
from typing import Tuple

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional RPLCD import
# ---------------------------------------------------------------------------
try:
    from RPLCD.i2c import CharLCD
    _RPLCD_AVAILABLE = True
except ImportError:
    _RPLCD_AVAILABLE = False


# Map friendly key names to config tuples so callers don't import config
_MESSAGES = {
    "ready":      config.LCD_MSG_READY,
    "scanning":   config.LCD_MSG_SCANNING,
    "recognized": config.LCD_MSG_RECOGNIZED,
    "unknown":    config.LCD_MSG_UNKNOWN,
    "duplicate":  config.LCD_MSG_DUPLICATE,
    "no_face":    config.LCD_MSG_NO_FACE,
}


class LCD:
    """
    Thin wrapper around RPLCD CharLCD.

    Falls back to logging when the library or hardware is unavailable
    so the rest of FRAMS runs normally on a development machine.
    """

    def __init__(
        self,
        i2c_address: int = config.LCD_I2C_ADDRESS,
        i2c_bus: int = config.LCD_I2C_BUS,
        cols: int = config.LCD_COLS,
        rows: int = config.LCD_ROWS,
        backlight: bool = config.LCD_BACKLIGHT,
    ) -> None:
        self._address = i2c_address
        self._bus = i2c_bus
        self._cols = cols
        self._rows = rows
        self._backlight = backlight
        self._lcd = None   # CharLCD instance, set in start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise the LCD.  Call once before using show()."""
        if not _RPLCD_AVAILABLE:
            logger.warning(
                "RPLCD not installed — LCD output will be logged only. "
                "Run: pip install RPLCD  (or: pip install -r requirements.txt)"
            )
            return

        try:
            self._lcd = CharLCD(
                i2c_expander="PCF8574",
                address=self._address,
                port=self._bus,
                cols=self._cols,
                rows=self._rows,
                backlight_enabled=self._backlight,
            )
            self._lcd.clear()
            logger.info(
                "LCD initialised (I2C 0x%02X, bus %d, %dx%d).",
                self._address, self._bus, self._cols, self._rows,
            )
        except Exception as exc:
            logger.warning("LCD init failed (%s) — falling back to log mode.", exc)
            self._lcd = None

    def stop(self) -> None:
        """Clear the display and release resources."""
        if self._lcd is not None:
            try:
                self._lcd.clear()
                self._lcd.backlight_enabled = False
                self._lcd.close(clear=True)
            except Exception as exc:
                logger.debug("LCD close error (ignored): %s", exc)
            finally:
                self._lcd = None
            logger.debug("LCD released.")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def show(self, line1: str = "", line2: str = "") -> None:
        """
        Write two lines to the LCD.  Each line is truncated / padded to
        exactly `cols` characters so the display is always cleanly overwritten.

        Parameters
        ----------
        line1, line2 : str
            Text for the first and second rows.
        """
        l1 = self._pad(str(line1))
        l2 = self._pad(str(line2))
        logger.debug("LCD | %s | %s |", l1.rstrip(), l2.rstrip())

        if self._lcd is None:
            return   # log-only mode

        try:
            self._lcd.home()
            self._lcd.write_string(l1 + l2)
        except Exception as exc:
            logger.warning("LCD write error: %s", exc)

    def clear(self) -> None:
        """Blank both lines."""
        logger.debug("LCD cleared.")
        if self._lcd is not None:
            try:
                self._lcd.clear()
            except Exception as exc:
                logger.warning("LCD clear error: %s", exc)

    def show_message(self, key: str, **kwargs) -> None:
        """
        Display a predefined message from config by key.

        Supported keys: ready, scanning, recognized, unknown, duplicate, no_face

        Any keyword arguments are used to format the message strings
        (e.g. show_message("recognized", name="Salem")).

        Parameters
        ----------
        key : str
            Key into the _MESSAGES dict.
        **kwargs
            Format arguments substituted into the message strings.
        """
        template: Tuple[str, str] = _MESSAGES.get(key, ("", ""))
        line1 = template[0].format(**kwargs) if kwargs else template[0]
        line2 = template[1].format(**kwargs) if kwargs else template[1]
        self.show(line1, line2)

    def set_backlight(self, on: bool) -> None:
        """Turn the LCD backlight on or off."""
        if self._lcd is not None:
            try:
                self._lcd.backlight_enabled = on
            except Exception as exc:
                logger.warning("LCD backlight error: %s", exc)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LCD":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pad(self, text: str) -> str:
        """Truncate or space-pad text to exactly `cols` characters."""
        return text[: self._cols].ljust(self._cols)
