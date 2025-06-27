import serial
import os

class Hardware:
    def __init__(self):
        self.serial_port = os.getenv("ARDUINO_SERIAL_PORT", "COM3")
        self.baud_rate = int(os.getenv("ARDUINO_BAUD_RATE", "9600"))
        try:
            self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            print(f"Connected to Arduino on {self.serial_port}")
        except serial.SerialException as e:
            print(f"Could not connect to Arduino on {self.serial_port}: {e}")
            self.arduino = None

    def update_leds(self, narration):
        if self.arduino:
            try:
                if "danger" in narration.lower():
                    self.arduino.write(b"R255G0B0\n") # Red for danger
                else:
                    self.arduino.write(b"B0G0B255\n") # Blue for normal
            except serial.SerialException as e:
                print(f"Error writing to Arduino: {e}")
