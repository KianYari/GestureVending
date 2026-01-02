# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER')
MQTT_PORT = int(os.getenv('MQTT_PORT', 8883))
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
MQTT_TOPIC_SELECTING = f"{MQTT_USERNAME}/vending/control"
MQTT_TOPIC_CONFIG = "vending/config/grid"
# Message structure: {"rows": int, "cols_double": int, "cols_single": int, "double_row_indices": list[int]}

# Validate required variables
if not all([MQTT_BROKER, MQTT_USERNAME, MQTT_PASSWORD]):
    raise ValueError("Missing required MQTT configuration in .env file")

# Network Configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 5000
MAX_UDP_PACKET_SIZE = 65536

# Interaction Configuration
CELL_SELECT_COOLDOWN = 0.3
CLICK_COOLDOWN = 0.5
CLICK_THRESHOLD = 40