import cv2
from pypylon import pylon, genicam
from datetime import datetime, timedelta
import time
import logging
import time

from sympy.core.benchmarks.bench_numbers import timeit_Integer_mul_Rational

wait_time = 30  # seconds between retries
log_interval = 1800  # seconds between logging the same error

logger = logging.getLogger(__name__)

class CameraBasler:
    def __init__(
            self,
            device_index: int = None,
            serial_number: str = None,
            exposure_us: int = -1,
            # manual exposure in microseconds (negative keeps current)
            log_exposure_changes: bool = True,
            # control info logging when exposure changes
    ):
        self.device_index = device_index
        self.serial_number = serial_number
        self.exposure_us = exposure_us
        self.log_exposure_changes = log_exposure_changes
        self.camera = None
        self.converter = None
        logger.info("Connecting to Basler camera:")
        if self.serial_number is not None:
            logger.info(f"  serial number: {self.serial_number}")
        if self.device_index is not None:
            logger.info(f"  device index: {self.device_index}")
        if self.exposure_us > 0:
            logger.info(f"  exposure: {self.exposure_us} µs")
        self._connect_camera()

    def _connect_camera(self):
        while True:
            try:
                factory = pylon.TlFactory.GetInstance()
                devices = factory.EnumerateDevices()
                if not devices:
                    if self._should_log_error('no_devices'):
                        logger.error(
                            f"No Basler cameras found. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                # Select device
                if self.serial_number:
                    selected = next((d for d in devices if
                                     d.GetSerialNumber() == self.serial_number),
                                    None)
                    if selected is None:
                        if self._should_log_error(
                                f'serial_{self.serial_number}'):
                            logger.error(
                                f"Camera with serial '{self.serial_number}' not found. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                elif self.device_index is not None:
                    if self.device_index >= len(devices):
                        if self._should_log_error(f'index_{self.device_index}'):
                            logger.error(
                                f"Camera index {self.device_index} out of range (found {len(devices)}). Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    selected = devices[self.device_index]
                else:
                    raise RuntimeError(
                        "❌ Either 'device_index' or 'serial_number' must be provided.")

                # Try to open camera
                self.camera = pylon.InstantCamera(
                    factory.CreateDevice(selected))
                info = self.camera.GetDeviceInfo()
                logger.info(
                    f"Connected to: {info.GetModelName()} ({info.GetSerialNumber()})")

                break

            except (pylon.RuntimeException, genicam.RuntimeException) as e:
                if self._should_log_error('runtime_error'):
                    logger.error(
                        f"Camera connection error: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                if self._should_log_error('unexpected_error'):
                    logger.error(
                        f"Unexpected error: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        # Exposure setup
        if self.exposure_us > 0:
            try:
                print('set:', self.exposure_us, flush=True)
                self.set_exposure(exposure_us=self.exposure_us)
                print('ok set:', self.exposure_us, flush=True)
            except Exception as e:
                logger.warning(f"Failed to set exposure: {str(e)}")

    def set_exposure(self, exposure_us: int = -1, timeout: int = 10):
        """Set exposure. exposure_us in microseconds."""
        #if exposure_us <= 0 or exposure_us == self.exposure_us:
        #    return
        timeout_time = time.monotonic() + timeout

        while time.monotonic() < timeout_time:
            try:
                if not self.camera:
                    self._connect_camera()

                self.camera.Open()

                node = self.camera.ExposureTimeRaw
                exp_min = node.Min
                exp_max = node.Max
                inc = getattr(node, "Inc", 1)

                desired = int(exposure_us)

                # clamp first
                desired = max(exp_min, min(exp_max, desired))

                # quantize to nearest legal value: exp = exp_min + round((desired-exp_min)/inc)*inc
                k = int(round((desired - exp_min) / inc))
                exp = exp_min + k * inc

                # ensure still within bounds after rounding
                if exp < exp_min:
                    exp = exp_min
                elif exp > exp_max:
                    exp = exp_max

                # write the value
                if genicam.IsWritable(node):
                    node.SetValue(int(exp))
                    if self.log_exposure_changes:
                        if exp != exposure_us:
                            logger.info(
                                f"Exposure snapped to grid: min={exp_min} inc={inc} → {exp} µs")
                        else:
                            logger.info(f"Exposure set to {exp} µs")
                    self.exposure_us = exp
                else:
                    logger.warning(
                        "ExposureTimeRaw not writable (check grabbing state / auto modes).")

                self.camera.Close()
                return

            except Exception as e:
                if self._should_log_error(f'set_exposure_{exposure_us}'):
                    logger.error(f"Failed to set exposure: {str(e)}")

        logging.warning(
            "Exposure set timeout expired. Exposure may not be set correctly.")

    def is_connected(self) -> bool:
        """Quick health check: is the target camera present and usable?"""
        try:
            if not self.camera:
                return False

            self.camera.Open()

            ok = self.camera.IsOpen()

            try:
                self.camera.Close()
            except Exception:
                ok = False
                pass

        except (pylon.RuntimeException, genicam.RuntimeException) as e:
            logger.debug(f"Camera health check failed (pylon/genicam): {e}")
            ok = False
        except Exception as e:
            logger.debug(f"Camera health check failed (unexpected): {e}")
            ok = False

        return bool(ok)

    def capture_image(self):
        """Grab a frame and return it as a NumPy array."""
        img = None
        try:
            t = time.time()
            if not self.camera.IsOpen():
                self.camera.Open()
            result = self.camera.GrabOne(10001)
            #self.camera.Close()
            #print(time.time() - t, flush=True)
            if result.GrabSucceeded():
                img = result.GetArray()
        except Exception as e:
            logger.error(f"Failed to capture image: {str(e)}")
        return img

    # Class variable to track last error log time
    _last_error_log = {}

    @classmethod
    def _should_log_error(cls, error_key: str) -> bool:
        """Check if enough time has passed since the last error log to not spam the log file."""
        now = datetime.now()
        if error_key not in cls._last_error_log:
            cls._last_error_log[error_key] = now
            return True

        if (now - cls._last_error_log[
            error_key]).total_seconds() >= log_interval:
            cls._last_error_log[error_key] = now
            return True
        return False