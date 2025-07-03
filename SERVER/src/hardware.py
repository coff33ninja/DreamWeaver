import serial
import os
import logging
import socket

logger = logging.getLogger("dreamweaver_server")


class _ArduinoHardware: # Renamed to indicate lower-level implementation
    def __init__(self):
        self.serial_port = os.getenv(
            "ARDUINO_SERIAL_PORT", None
        )
        self.baud_rate = int(os.getenv("ARDUINO_BAUD_RATE", "9600"))
        self.arduino = None

        if self.serial_port:
            try:
                self.arduino = serial.Serial(
                    self.serial_port, self.baud_rate, timeout=1
                )
                logger.info(
                    f"Successfully connected to Arduino on {self.serial_port} at {self.baud_rate} baud."
                )
            except serial.SerialException as e:
                logger.warning(
                    f"Could not connect to Arduino on {self.serial_port}: {e}. Arduino hardware features will be disabled."
                )
            except Exception as e_other:
                logger.error(
                    f"An unexpected error occurred while trying to connect to Arduino on {self.serial_port}: {e_other}",
                    exc_info=True,
                )
        else:
            logger.info(
                "ARDUINO_SERIAL_PORT not set. Arduino hardware features disabled."
            )

    def send_command(self, command: bytes):
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(command)
                logger.debug(f"Sent command {command.strip()} to Arduino.")
                return True
            except serial.SerialException as e:
                logger.error(
                    f"Error writing command {command.strip()} to Arduino on {self.serial_port}: {e}",
                    exc_info=True,
                )
            except Exception as e_other:
                logger.error(
                    f"Unexpected error writing command {command.strip()} to Arduino: {e_other}", exc_info=True
                )
        elif self.arduino and not self.arduino.is_open:
            logger.warning(
                f"Attempted to send command {command.strip()}, but Arduino serial port is not open."
            )
        else: # self.arduino is None
            logger.debug( # Changed to debug as it's less critical if not configured
                f"Attempted to send command {command.strip()}, but Arduino is not configured (serial port not set)."
            )
        return False

    def __del__(self):
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.close()
                logger.info(f"Closed Arduino serial port {self.serial_port}.")
            except Exception as e:
                logger.error(
                    f"Error closing Arduino serial port {self.serial_port}: {e}",
                    exc_info=True,
                )

class HardwareManager:
    def __init__(self):
        self._arduino_hw = _ArduinoHardware()

    def update_story_leds(self, narration_text: str):
        """
        Updates LED status based on keywords in the narration text.
        Sends a command to the Arduino to change LED color:
        - Red for "danger".
        - Party mode for "party" or "celebrate".
        - Blue for normal narration.
        """
        command = b""
        narration_lower = narration_text.lower()

        if "danger" in narration_lower:
            command = b"R255G0B0\n"  # Red for danger
            logger.debug("HardwareManager: Determined RED LED for 'danger' in narration.")
        elif "party" in narration_lower or "celebrate" in narration_lower:
            command = b"P1\n"  # Assuming 'P1' is a command for party mode
            logger.debug("HardwareManager: Determined PARTY LED for 'party/celebrate' in narration.")
        else:
            command = b"B0G0B255\n"  # Blue for normal
            logger.debug("HardwareManager: Determined BLUE LED for normal narration.")

        if command:
            self._arduino_hw.send_command(command)
        else:
            logger.debug("HardwareManager: No specific LED command determined from narration.")

    # If other hardware interactions are needed, they would be added here,
    # calling methods on self._arduino_hw or other specific hardware controllers.

# get_adapter_ip_addresses has been moved to SERVER/src/network_utils.py

    def __del__(self):
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.close()
                logger.info(f"Closed Arduino serial port {self.serial_port}.")
            except Exception as e:
                logger.error(
                    f"Error closing Arduino serial port {self.serial_port}: {e}",
                    exc_info=True,
                )
