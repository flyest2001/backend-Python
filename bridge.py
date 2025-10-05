# This script acts as a bridge between the FastAPI server and the Arduino Mega.
# It polls the API and sends the sensor status to the Mega via USB Serial.

import serial
import requests
import time
import sys

# --- Configuration ---
# IMPORTANT: Replace with your Arduino Mega's COM port
# Find this in the Arduino IDE under Tools > Port (e.g., 'COM3' on Windows)
MEGA_PORT = 'COM12' 
BAUD_RATE = 115200
API_URL = 'http://127.0.0.1:8000/status'

def run_bridge():
    """
    Main function to run the bridge. Connects to the Arduino and polls the API.
    """
    print("Starting the API-to-Hardware bridge...")
    
    # Establish a serial connection with the Arduino Mega
    try:
        mega = serial.Serial(MEGA_PORT, BAUD_RATE, timeout=1)
        print(f"Successfully connected to Arduino Mega on {MEGA_PORT}")
        # Give the Mega a moment to reset after the connection is opened
        time.sleep(2) 
    except serial.SerialException as e:
        print(f"Error: Could not connect to the Arduino Mega on {MEGA_PORT}.")
        print(f"Details: {e}")
        print("Please check the port and ensure the Arduino IDE's Serial Monitor is closed.")
        sys.exit(1)

    # In your Python bridge script...

    # This new variable will track the previous running state.
    # In your Python bridge script...

    # This variable will store the last message sent to the Arduino.
    last_message_sent = ""

    while True:
        try:
            # 1. Get status from the FastAPI server
            response = requests.get(API_URL)
            response.raise_for_status() 
            data = response.json()

            # 2. Always format the current sensor state into a message string.
            if 'sensors' in data:
                sensor_states = ['1' if s['is_off'] else '0' for s in data['sensors']]
                current_message = ",".join(sensor_states) + "\n"

                # 3. Only send the message if it's different from the last one.
                if current_message != last_message_sent:
                    mega.write(current_message.encode('utf-8'))
                    last_message_sent = current_message # Update the tracker
                    print(f"State changed. Syncing LEDs: {current_message.strip()}")

            # Control the update speed
            time.sleep(0.1)

        except requests.exceptions.RequestException:
            print(f"Error: Could not connect to the server at {API_URL}. Retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    mega.close()
    print("Bridge stopped.")

if __name__ == '__main__':
    run_bridge()
