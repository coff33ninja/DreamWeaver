import serial
import os
import logging

logger = logging.getLogger("dreamweaver_server")

class Hardware:
    def __init__(self):
        self.serial_port = os.getenv("ARDUINO_SERIAL_PORT", None) # Default to None if not set
        self.baud_rate = int(os.getenv("ARDUINO_BAUD_RATE", "9600"))
        self.arduino = None

        if self.serial_port:
            try:
                self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
                logger.info(f"Successfully connected to Arduino on {self.serial_port} at {self.baud_rate} baud.")
            except serial.SerialException as e:
                logger.warning(f"Could not connect to Arduino on {self.serial_port}: {e}. Hardware features will be disabled.")
            except Exception as e_other: # Catch other potential errors like ValueError for baud_rate
                 logger.error(f"An unexpected error occurred while trying to connect to Arduino on {self.serial_port}: {e_other}", exc_info=True)

        else:
            logger.info("ARDUINO_SERIAL_PORT not set. Arduino hardware features disabled.")


    def update_leds(self, narration):
        if self.arduino and self.arduino.is_open:
            try:
                command = b""
                if "danger" in narration.lower():
                    command = b"R255G0B0\n" # Red for danger
                    logger.debug("Sending RED LED command to Arduino due to 'danger' in narration.")
                elif "party" in narration.lower() or "celebrate" in narration.lower():
                    command = b"P1\n" # Assuming 'P1' is a command for party mode
                    logger.debug("Sending PARTY LED command to Arduino.")
                else:
                    command = b"B0G0B255\n" # Blue for normal
                    logger.debug("Sending BLUE LED command to Arduino for normal narration.")

                if command:
                    self.arduino.write(command)
                    # logger.debug(f"Sent command {command.strip()} to Arduino.") # Can be too verbose
            except serial.SerialException as e:
                logger.error(f"Error writing to Arduino on {self.serial_port}: {e}", exc_info=True)
            except Exception as e_other:
                logger.error(f"Unexpected error writing to Arduino: {e_other}", exc_info=True)
        elif self.arduino and not self.arduino.is_open:
            logger.warning("Attempted to update LEDs, but Arduino serial port is not open.")
        # If self.arduino is None, no message is logged here as it's logged during init.

    def __del__(self):
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.close()
                logger.info(f"Closed Arduino serial port {self.serial_port}.")
            except Exception as e:
                logger.error(f"Error closing Arduino serial port {self.serial_port}: {e}", exc_info=True)
