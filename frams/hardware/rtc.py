"""
T8 — RTC Driver
DS3231 Real-Time Clock via I2C (smbus2).

Provides:
  - get_datetime()  → datetime  (read current time from chip)
  - set_datetime(dt)            (write time to chip)
  - sync_system_clock()         (set Pi system clock from RTC)

On non-Pi machines (smbus2 unavailable or I2C bus absent) every method
raises RTCError so the caller can handle the absence gracefully.

Usage
-----
    from hardware.rtc import RTC, RTCError

    rtc = RTC()
    now = rtc.get_datetime()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
"""

import logging
import subprocess
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional smbus2 import
# ---------------------------------------------------------------------------
try:
    import smbus2
    _SMBUS2_AVAILABLE = True
except ImportError:
    _SMBUS2_AVAILABLE = False


def _bcd_to_dec(bcd: int) -> int:
    return (bcd >> 4) * 10 + (bcd & 0x0F)


def _dec_to_bcd(dec: int) -> int:
    return ((dec // 10) << 4) | (dec % 10)


class RTCError(Exception):
    """Raised when the RTC cannot be reached or a read/write fails."""


class RTC:
    """
    DS3231 driver.  Communicates over I2C using smbus2.

    The DS3231 register map used here:
        0x00 — seconds   (BCD)
        0x01 — minutes   (BCD)
        0x02 — hours     (BCD, 24-hour mode)
        0x04 — day       (BCD, 1–31)
        0x05 — month     (BCD, bit 7 = century flag)
        0x06 — year      (BCD, 00–99; century handled separately)
    """

    # First timekeeping register
    _REG_SECONDS = 0x00
    # Number of registers to read in one shot (sec, min, hr, dow, day, mon, yr)
    _READ_LEN = 7

    def __init__(
        self,
        i2c_address: int = config.RTC_I2C_ADDRESS,
        i2c_bus: int = config.RTC_I2C_BUS,
    ) -> None:
        self._address = i2c_address
        self._bus_id = i2c_bus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_datetime(self) -> datetime:
        """
        Read the current date and time from the DS3231.

        Returns
        -------
        datetime
            Timezone-naive datetime in local time.

        Raises
        ------
        RTCError
            If smbus2 is unavailable or the I2C read fails.
        """
        self._check_available()
        try:
            with smbus2.SMBus(self._bus_id) as bus:
                data = bus.read_i2c_block_data(
                    self._address, self._REG_SECONDS, self._READ_LEN
                )
        except Exception as exc:
            raise RTCError(f"I2C read failed: {exc}") from exc

        second = _bcd_to_dec(data[0] & 0x7F)
        minute = _bcd_to_dec(data[1] & 0x7F)
        hour   = _bcd_to_dec(data[2] & 0x3F)   # mask out 12/24 bit
        day    = _bcd_to_dec(data[4] & 0x3F)
        month  = _bcd_to_dec(data[5] & 0x1F)   # mask out century bit
        year   = _bcd_to_dec(data[6]) + 2000

        dt = datetime(year, month, day, hour, minute, second)
        logger.debug("RTC read: %s", dt.isoformat())
        return dt

    def set_datetime(self, dt: Optional[datetime] = None) -> None:
        """
        Write a datetime to the DS3231.  Defaults to the current system time
        if no argument is supplied.

        Parameters
        ----------
        dt : datetime, optional
            Time to write.  Uses datetime.now() if omitted.

        Raises
        ------
        RTCError
            If smbus2 is unavailable or the I2C write fails.
        """
        self._check_available()
        if dt is None:
            dt = datetime.now()

        data = [
            _dec_to_bcd(dt.second),
            _dec_to_bcd(dt.minute),
            _dec_to_bcd(dt.hour),
            0x01,                           # day-of-week (unused, set to 1)
            _dec_to_bcd(dt.day),
            _dec_to_bcd(dt.month),
            _dec_to_bcd(dt.year % 100),
        ]

        try:
            with smbus2.SMBus(self._bus_id) as bus:
                bus.write_i2c_block_data(self._address, self._REG_SECONDS, data)
        except Exception as exc:
            raise RTCError(f"I2C write failed: {exc}") from exc

        logger.info("RTC set to: %s", dt.isoformat())

    def sync_system_clock(self) -> None:
        """
        Read the RTC and update the Pi's system clock via `sudo date`.
        Only meaningful on the Raspberry Pi.

        Raises
        ------
        RTCError
            If the RTC read fails or the date command fails.
        """
        dt = self.get_datetime()
        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        try:
            subprocess.run(
                ["sudo", "date", "-s", date_str],
                check=True,
                capture_output=True,
            )
            logger.info("System clock synced to RTC: %s", date_str)
        except subprocess.CalledProcessError as exc:
            raise RTCError(
                f"Failed to set system clock: {exc.stderr.decode().strip()}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_available(self) -> None:
        if not _SMBUS2_AVAILABLE:
            raise RTCError(
                "smbus2 is not installed. "
                "Run: pip install smbus2  (or: pip install -r requirements.txt)"
            )
