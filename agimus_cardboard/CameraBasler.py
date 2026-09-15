from typing import Optional
import cv2
from pypylon import pylon, genicam
from datetime import datetime
import logging
import time

wait_time = 30  # seconds between retries
log_interval = 1800  # seconds between logging the same error

class CameraBasler:
    # TODO: the camera should stay open after setting the parameters.
    # Otherwise, we observed in some cases, that the packet_size is reset
    # to wrong value.

    def __init__(
            self,
            device_index: Optional[int] = None,
            serial_number: Optional[str] = None,
            exposure_us: int = -1,
            packet_size: int = 1500,
            gain: int = 0,
            grayscale: bool = False,
            logger=None,
    ):
        """Connect to a Basler camera and configure it.

        A diagnostic frame is grabbed and the camera and frame parameters are
        logged.

        Either device_index or serial_number must be provided; serial_number
        takes precedence if both are given.

        Connection attempts retry until successful. The camera remains open.

        :param device_index: Zero-based index in the enumerated camera list.

        :param serial_number: Camera serial number,

        :param exposure_us: Manual exposure in microseconds; nonpositive value
            leave the current exposure settings unchanged.

        :param packet_size: GigE packet size in bytes; nonpositive value skip
                packet-size configuration; use this for USB cameras.

        :param gain: Raw manual gain; negative value sets continuous auto gain.

        :param grayscale: If True, capture_image converts frames through BGR8 to
                single-channel, 8-bit grayscale. Otherwise, all frames are
                converted to three-channel BGR8.

        :param logger: Logger to use, or None to use the default module logger.

        :raises RuntimeError: camera selector is not supplied, or the initial
                diagnostic frame cannot be captured.
        """
        self.logger = (
          logger if logger is not None else logging.getLogger(__name__)
        )

        self.device_index = device_index
        self.serial_number = serial_number
        self.exposure_us = exposure_us
        self.camera = None
        self.grayscale = grayscale
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.packet_size = packet_size
        self.gain = gain
        self.logger.info("Connecting to Basler camera:")
        if self.serial_number is not None:
            self.logger.info(f"  serial number: {self.serial_number}")
        if self.device_index is not None:
            self.logger.info(f"  device index: {self.device_index}")
        if self.exposure_us > 0:
            self.logger.info(f"  exposure: {self.exposure_us} µs")
        self._connect_camera()

        # Capture one image and log its properties

        result = self._grab_one()
        if result is None:
            raise RuntimeError("Cannot grab an image")
        img = result.GetArray()

        self.logger.info(f"PixelFormat:  {self.camera.PixelFormat.GetValue()}")
        self.logger.info(f"PixelType: {result.GetPixelType()}")
        self.logger.info(f"Array: {img.shape}, {img.dtype}")

    def _connect_camera(self):
        while True:
            try:
                if self.is_open():
                    self.camera.Close()

                factory = pylon.TlFactory.GetInstance()
                devices = factory.EnumerateDevices()
                if not devices:
                    if self._should_log_error('no_devices'):
                        self.logger.error(
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
                            self.logger.error(
                                f"Camera with serial '{self.serial_number}' not found. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                elif self.device_index is not None:
                    if self.device_index >= len(devices):
                        if self._should_log_error(f'index_{self.device_index}'):
                            self.logger.error(
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
                self.camera.Open()
                if self.packet_size > 0:
                    self.camera.GevSCPSPacketSize.SetValue(self.packet_size)

                if self.gain < 0:
                    self.camera.GainAuto.SetValue("Continuous")
                else:
                    self.camera.GainAuto.SetValue("Off")
                    self.camera.GainRaw.SetValue(int(self.gain))

                self.logger.info(
                    f"Connected to: {info.GetModelName()} ({info.GetSerialNumber()})")
                if self.packet_size > 0:
                    self.logger.info(
                        f"Packet size: {self.camera.GevSCPSPacketSize.GetValue()}")
                break

            except (pylon.RuntimeException, genicam.RuntimeException) as e:
                if self._should_log_error('runtime_error'):
                    self.logger.error(
                        f"Camera connection error: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        # Exposure setup
        if self.exposure_us > 0:
            try:
                self.logger.info(f"  exposure: {self.exposure_us}")
                self._set_exposure(self.exposure_us)
            except Exception as e:
                self.logger.warning(f"Failed to set exposure: {str(e)}")

    def is_open(self) -> bool:
        return self.camera is not None and self.camera.IsOpen()

    def set_exposure(self, exposure_us: int = -1, timeout: int = 10):
        if not self.is_open():
            self._connect_camera()

        self._set_exposure(exposure_us, timeout)

    def _set_exposure(self, exposure_us: int = -1, timeout: int = 10):
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
                    if exp != exposure_us:
                        self.logger.info(
                            f"Exposure snapped to grid: min={exp_min} inc={inc} → {exp} µs")
                    else:
                        self.logger.info(f"Exposure set to {exp} µs")
                    self.exposure_us = exp
                else:
                    self.logger.warning(
                        "ExposureTimeRaw not writable (check grabbing state / auto modes).")

                return

            except Exception as e:
                if self._should_log_error(f'set_exposure_{exposure_us}'):
                    self.logger.error(f"Failed to set exposure: {str(e)}")

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
            self.logger.debug(f"Camera health check failed (pylon/genicam): {e}")
            ok = False
        except Exception as e:
            self.logger.debug(f"Camera health check failed (unexpected): {e}")
            ok = False

        return bool(ok)

    def _grab_one(self):
        try:
            if not self.camera.IsOpen():
                self.camera.Open()
            result = self.camera.GrabOne(10001)
            if result.GrabSucceeded():
                return result
            self.logger.error(f"Failed to capture image (grab not succeeded)")

        except Exception as e:
            self.logger.error(f"Failed to capture image: {str(e)}")

        return None

    def capture_image(self):
        """Return three-channel BGR8, or single-channel 8-bit grayscale.

        All frames pass through BGR8 conversion. Return None if grabbing fails.
        """
        result = self._grab_one()
        if result is not None:
            img = self.converter.Convert(result).GetArray()
            if self.grayscale:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            return None

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
