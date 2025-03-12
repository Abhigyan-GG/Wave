import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
import asyncio
import websockets
import json
import threading
import logging
import time
from websockets.exceptions import ConnectionClosed
from concurrent.futures import ThreadPoolExecutor

# For system volume control (Windows)
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    VOLUME_CONTROL_AVAILABLE = True
except ImportError:
    VOLUME_CONTROL_AVAILABLE = False
    logging.warning("pycaw not available - volume control disabled")

##########################
# Configuration
##########################
CONFIG = {
    "ws_port": 8765,
    "screen_width": 1024,
    "screen_height": 600,
    "cam_width": 640,
    "cam_height": 480,
    "min_frame_interval": 0.1,
    "hand_detection_confidence": 0.6,
    "hand_tracking_confidence": 0.6,
    "jpeg_quality": 50,
    "default_volume": 50,
    "default_fan_speed": 100,
    "pinch_threshold": 0.05,
    "spread_threshold": 0.2,
    "volume_change_threshold": 2
}

##########################
# Logging Setup
##########################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gesture_control")

##########################
# MediaPipe Hands Setup
##########################
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HandDetector:
    """Context manager for MediaPipe Hands."""
    def __init__(self):
        self.hands = mp_hands.Hands(
            min_detection_confidence=CONFIG["hand_detection_confidence"],
            min_tracking_confidence=CONFIG["hand_tracking_confidence"],
            max_num_hands=2
        )

    def __enter__(self):
        return self.hands

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hands.close()

hand_detector = HandDetector()
hands = hand_detector.hands

##########################
# System Volume Control (Windows)
##########################
def initialize_volume_control():
    if not VOLUME_CONTROL_AVAILABLE:
        return None, 0, 0

    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
        min_vol, max_vol = volume_interface.GetVolumeRange()[:2]
        return volume_interface, min_vol, max_vol
    except Exception as e:
        logging.error(f"Failed to initialize volume control: {e}")
        return None, 0, 0

volume_interface, min_vol, max_vol = initialize_volume_control()

def set_system_volume(volume_percent):
    """Set system volume (0-100) using PyCAW (Windows-only)."""
    if not VOLUME_CONTROL_AVAILABLE or volume_interface is None:
        return
    try:
        volume_scalar = np.interp(volume_percent, [0, 100], [min_vol, max_vol])
        volume_interface.SetMasterVolumeLevel(volume_scalar, None)
    except Exception as e:
        logging.warning(f"Failed to set system volume: {e}")

##########################
# Pygame UI Setup
##########################
pygame.init()
screen_width, screen_height = CONFIG["screen_width"], CONFIG["screen_height"]
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Gesture Control System")

CAM_WIDTH, CAM_HEIGHT = CONFIG["cam_width"], CONFIG["cam_height"]
UI_WIDTH = screen_width - CAM_WIDTH
PANEL_PADDING = 20

# Colors
BG_COLOR = (30, 30, 40)
PRIMARY_COLOR = (76, 175, 80)
ACCENT_COLOR = (255, 193, 7)
TEXT_COLOR = (255, 255, 255)

# Fonts
TITLE_FONT = pygame.font.Font(None, 48)
LARGE_FONT = pygame.font.Font(None, 36)
MEDIUM_FONT = pygame.font.Font(None, 30)
SMALL_FONT = pygame.font.Font(None, 24)

# App State (globals)
meter_value = CONFIG["default_fan_speed"]  # 0-100
fan_state = False
last_intensity = CONFIG["default_fan_speed"]
volume_level = CONFIG["default_volume"]

# Gesture State
previous_fan_palm_state = None
previous_fan_pinch_state = None
previous_vol_pinch_state = None

# Frame Processing
remote_frame = None
processed_frame = None
last_frame_time = 0
frame_processing_lock = threading.Lock()

# Performance Metrics
fps_counter = 0
fps_timer = time.time()
current_fps = 0

# Thread Pool
frame_executor = ThreadPoolExecutor(max_workers=1)

##########################
# WebSocket Server Setup
##########################
connected_clients = set()
connected_clients_lock = threading.Lock()
ws_loop = None

##########################
# WebSocket Handlers
##########################
async def websocket_handler(websocket):
    """Handles new WebSocket connections (camera feed, commands, etc.)."""
    global remote_frame
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"

    with connected_clients_lock:
        connected_clients.add(websocket)
        logger.info(f"Client connected: {client_id}")

    # Send initial state
    await send_state_to_client(websocket)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                header = b"frame:"
                if message.startswith(header):
                    process_incoming_frame(message[len(header):])
                else:
                    logger.warning(f"Binary message without proper header from {client_id}")
            else:
                # Text message: could be commands or JSON
                try:
                    data = json.loads(message)
                    if "command" in data:
                        handle_command(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_id}: {message}")
                    await websocket.send("error: Invalid JSON format")

    except ConnectionClosed:
        logger.info(f"Connection closed by client: {client_id}")
    except Exception as e:
        logger.exception(f"Exception with {client_id}:")
    finally:
        with connected_clients_lock:
            connected_clients.discard(websocket)
            logger.info(f"Client disconnected: {client_id}")

def process_incoming_frame(jpeg_data):
    """Schedules frame decoding + gesture detection in a background thread."""
    global last_frame_time, fps_counter, current_fps, fps_timer

    current_time = time.time()
    # Throttle to ~10 FPS
    if current_time - last_frame_time < CONFIG["min_frame_interval"]:
        return

    last_frame_time = current_time
    fps_counter += 1

    if current_time - fps_timer >= 1.0:
        current_fps = fps_counter
        fps_counter = 0
        fps_timer = current_time

    frame_executor.submit(_process_frame, jpeg_data)

def _process_frame(jpeg_data):
    """Decodes and processes the frame, updates the UI image."""
    global remote_frame, processed_frame

    try:
        np_arr = np.frombuffer(jpeg_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Failed to decode JPEG data.")
            return

        frame = cv2.flip(frame, 1)  # Flip around y-axis

        with frame_processing_lock:
            remote_frame = frame.copy()

        processed = detect_gestures(frame)
        with frame_processing_lock:
            processed_frame = processed

    except Exception as e:
        logger.exception(f"Error processing frame: {e}")

def handle_command(data):
    """Handles JSON commands (set_fan, set_volume, etc.)."""
    global fan_state, meter_value, volume_level

    cmd = data.get("command")
    if cmd == "set_fan":
        fan_state = data.get("state", fan_state)
        if "value" in data:
            meter_value = max(0, min(100, data["value"]))
        send_state_update()

    elif cmd == "set_volume":
        if "value" in data:
            volume_level = max(0, min(100, data["value"]))
            set_system_volume(volume_level)
            send_state_update()

##########################
# Start Server in a Thread
##########################
async def start_server():
    global ws_loop
    ws_loop = asyncio.get_running_loop()
    server = await websockets.serve(websocket_handler, "0.0.0.0", CONFIG["ws_port"])
    ip = get_local_ip()
    logger.info(f"Server started on ws://{ip}:{CONFIG['ws_port']}")
    await asyncio.Future()

def run_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(start_server())
    except Exception as e:
        logger.exception("Exception starting WebSocket server:")
    loop.run_forever()

ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
ws_thread.start()

def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

##########################
# Messaging / Broadcasting
##########################
def send_state_update():
    """Broadcast the current fan/volume state to all connected clients."""
    state = {
        "type": "state",
        "fan_state": fan_state,
        "fan_speed": meter_value,
        "volume": volume_level
    }

    async def _send():
        with connected_clients_lock:
            for client in list(connected_clients):
                try:
                    await client.send(json.dumps(state))
                except Exception as e:
                    logger.error(f"Error sending state: {e}")
                    connected_clients.discard(client)

    if ws_loop is not None:
        asyncio.run_coroutine_threadsafe(_send(), ws_loop)

async def send_state_to_client(client):
    """Send the current state to a newly connected client."""
    state = {
        "type": "state",
        "fan_state": fan_state,
        "fan_speed": meter_value,
        "volume": volume_level
    }
    try:
        await client.send(json.dumps(state))
    except Exception as e:
        logger.error(f"Error sending state to client: {e}")

def send_gesture_detected(gesture_name):
    """Send a gesture event to all clients, so the front-end can highlight it."""
    msg = {"type": "gesture", "name": gesture_name}

    async def _send():
        with connected_clients_lock:
            for client in list(connected_clients):
                await client.send(json.dumps(msg))

    if ws_loop is not None:
        asyncio.run_coroutine_threadsafe(_send(), ws_loop)

##########################
# Gesture Detection Logic
##########################
def is_hand_open(hand_landmarks):
    """Check if at least 3 fingers are extended."""
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]
    open_fingers = sum(
        1 for tip, pip in fingers
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
    )
    return open_fingers >= 3

def detect_gestures(frame):
    """Detect gestures (palm open/close, pinch/spread) and update fan/volume."""
    global meter_value, fan_state, last_intensity, volume_level
    global previous_fan_palm_state, previous_fan_pinch_state, previous_vol_pinch_state

    output_frame = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return output_frame

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if idx >= len(results.multi_handedness):
            continue

        handedness = results.multi_handedness[idx].classification[0].label

        # Coordinates for pinch detection
        thumb_tip = np.array([
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        ])
        index_tip = np.array([
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        ])
        distance = np.linalg.norm(thumb_tip - index_tip)

        # Draw landmarks
        mp_drawing.draw_landmarks(
            output_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Label
        cv2.putText(
            output_frame,
            f"{handedness} Hand",
            (int(hand_landmarks.landmark[0].x * frame.shape[1]),
             int(hand_landmarks.landmark[0].y * frame.shape[0]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        # Right Hand => Fan
        if handedness == "Right":
            current_palm_open = is_hand_open(hand_landmarks)

            # Palm open/close => toggle fan
            if current_palm_open != previous_fan_palm_state:
                if current_palm_open:
                    # Palm Open => ON
                    fan_state = True
                    meter_value = last_intensity
                    logger.info("Fan toggled ON")
                    send_gesture_detected("palm-open")
                else:
                    # Palm Closed => OFF
                    fan_state = False
                    last_intensity = meter_value
                    meter_value = 0
                    logger.info("Fan toggled OFF")
                    send_gesture_detected("palm-closed")

                send_state_update()

            previous_fan_palm_state = current_palm_open

            # Pinch/spread => adjust intensity if fan ON
            if fan_state:
                if distance < CONFIG["pinch_threshold"]:
                    current_pinch = "pinch"
                elif distance > CONFIG["spread_threshold"]:
                    current_pinch = "spread"
                else:
                    current_pinch = None

                if current_pinch and current_pinch != previous_fan_pinch_state:
                    delta = 25 if current_pinch == "spread" else -25
                    new_meter = int(np.clip(meter_value + delta, 0, 100))
                    if new_meter != meter_value:
                        meter_value = new_meter
                        logger.info(f"Fan intensity => {meter_value}%")
                        send_gesture_detected(current_pinch)  # pinch or spread
                        send_state_update()

                    previous_fan_pinch_state = current_pinch
                elif not current_pinch:
                    previous_fan_pinch_state = None

        # Left Hand => Volume
        elif handedness == "Left" and VOLUME_CONTROL_AVAILABLE:
            new_volume = int(np.clip(np.interp(distance, [0.05, 0.3], [0, 100]), 0, 100))
            if abs(new_volume - volume_level) > CONFIG["volume_change_threshold"]:
                volume_level = new_volume
                set_system_volume(volume_level)
                logger.info(f"Volume => {volume_level}%")
                # You could also do send_gesture_detected("pinch/spread") if you want
                send_state_update()

            # Draw pinch line
            cv2.line(
                output_frame,
                (int(thumb_tip[0] * frame.shape[1]), int(thumb_tip[1] * frame.shape[0])),
                (int(index_tip[0] * frame.shape[1]), int(index_tip[1] * frame.shape[0])),
                (0, 255, 0),
                2
            )

    # FPS
    cv2.putText(
        output_frame,
        f"FPS: {current_fps}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return output_frame

##########################
# UI (Pygame) Drawing
##########################
def draw_meter(surface, x, y, width, height, value, active, label):
    pygame.draw.rect(surface, (50, 50, 60), (x, y, width, height), border_radius=10)
    fill_height = int((value / 100) * height)
    color = PRIMARY_COLOR if active else (100, 100, 100)
    pygame.draw.rect(
        surface,
        color,
        (x, y + height - fill_height, width, fill_height),
        border_radius=10
    )
    text = LARGE_FONT.render(f"{value}%", True, TEXT_COLOR)
    surface.blit(
        text,
        (x + width // 2 - text.get_width() // 2, y + height // 2 - 10)
    )
    label_text = SMALL_FONT.render(label, True, TEXT_COLOR)
    surface.blit(label_text, (x, y - 30))

def draw_camera_panel():
    frame = None
    with frame_processing_lock:
        if processed_frame is not None:
            frame = processed_frame.copy()

    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        camera_surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(camera_surf, (0, (screen_height - CAM_HEIGHT) // 2))

        fps_text = SMALL_FONT.render(f"FPS: {current_fps}", True, (255, 255, 0))
        screen.blit(fps_text, (10, (screen_height - CAM_HEIGHT) // 2 + 10))
    else:
        waiting_text = TITLE_FONT.render("Waiting for mobile feed...", True, TEXT_COLOR)
        text_rect = waiting_text.get_rect(center=(CAM_WIDTH // 2, screen_height // 2))
        temp_surface = pygame.Surface((CAM_WIDTH, CAM_HEIGHT))
        temp_surface.fill((0, 0, 0))
        temp_surface.blit(waiting_text, text_rect)
        screen.blit(temp_surface, (0, (screen_height - CAM_HEIGHT) // 2))

def draw_ui():
    draw_camera_panel()
    control_panel = pygame.Surface((UI_WIDTH, screen_height))
    control_panel.fill(BG_COLOR)

    title = TITLE_FONT.render("Gesture Control", True, TEXT_COLOR)
    control_panel.blit(title, (PANEL_PADDING, 10))

    # Fan meter
    draw_meter(
        control_panel,
        PANEL_PADDING,
        70,
        UI_WIDTH - 2 * PANEL_PADDING,
        150,
        meter_value,
        fan_state,
        "Fan Control (Right Hand)"
    )
    # Volume meter
    draw_meter(
        control_panel,
        PANEL_PADDING,
        270,
        UI_WIDTH - 2 * PANEL_PADDING,
        150,
        volume_level,
        True,
        "Volume Control (Left Hand)"
    )

    fan_status = MEDIUM_FONT.render(
        f"Fan: {'RUNNING' if fan_state else 'STOPPED'}",
        True,
        PRIMARY_COLOR if fan_state else (100, 100, 100)
    )
    vol_status = MEDIUM_FONT.render(
        f"System Volume: {volume_level}%",
        True,
        PRIMARY_COLOR
    )
    control_panel.blit(fan_status, (PANEL_PADDING, 230))
    control_panel.blit(vol_status, (PANEL_PADDING, 430))

    # Buttons
    if fan_state:
        draw_button(control_panel, "+", PANEL_PADDING, 470, 60, 40, (100, 100, 180))
        draw_button(control_panel, "-", PANEL_PADDING + 70, 470, 60, 40, (100, 100, 180))

    draw_button(
        control_panel,
        "TOGGLE FAN",
        PANEL_PADDING,
        520,
        UI_WIDTH - 2 * PANEL_PADDING,
        50,
        PRIMARY_COLOR if fan_state else (100, 100, 100)
    )

    # Connections
    with connected_clients_lock:
        ccount = len(connected_clients)
    conn_status = SMALL_FONT.render(f"Connections: {ccount}", True, ACCENT_COLOR)
    control_panel.blit(conn_status, (PANEL_PADDING, screen_height - 30))

    screen.blit(control_panel, (CAM_WIDTH, 0))

def draw_button(surface, text, x, y, width, height, color):
    mouse = pygame.mouse.get_pos()
    mouse_x = mouse[0] - CAM_WIDTH if mouse[0] > CAM_WIDTH else 0
    btn_rect = pygame.Rect(x, y, width, height)

    if btn_rect.collidepoint(mouse_x, mouse[1]):
        hover_color = tuple(min(c + 30, 255) for c in color)
        pygame.draw.rect(surface, hover_color, btn_rect, border_radius=5)
    else:
        pygame.draw.rect(surface, color, btn_rect, border_radius=5)

    text_surf = LARGE_FONT.render(text, True, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
    surface.blit(text_surf, text_rect)

##########################
# Main Pygame Loop
##########################
def main():
    global meter_value, fan_state, volume_level, last_intensity
    running = True
    clock = pygame.time.Clock()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    handle_mouse_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        fan_state = not fan_state
                        if fan_state:
                            meter_value = last_intensity
                        else:
                            last_intensity = meter_value
                            meter_value = 0
                        send_state_update()

            screen.fill(BG_COLOR)
            draw_ui()
            pygame.display.flip()
            clock.tick(60)

    except Exception as e:
        logger.exception("Exception in main loop:")
    finally:
        cv2.destroyAllWindows()
        pygame.quit()
        hands.close()
        frame_executor.shutdown()
        logger.info("Application shutdown complete")

def handle_mouse_click(pos):
    global fan_state, meter_value, last_intensity
    if pos[0] <= CAM_WIDTH:
        return
    btn_x = pos[0] - CAM_WIDTH

    # Fan intensity up
    if (PANEL_PADDING <= btn_x <= PANEL_PADDING + 60 and
        470 <= pos[1] <= 510 and fan_state):
        meter_value = min(meter_value + 25, 100)
        send_state_update()

    # Fan intensity down
    elif (PANEL_PADDING + 70 <= btn_x <= PANEL_PADDING + 130 and
          470 <= pos[1] <= 510 and fan_state):
        meter_value = max(meter_value - 25, 0)
        send_state_update()

    # Toggle fan
    elif (PANEL_PADDING <= btn_x <= UI_WIDTH - PANEL_PADDING and
          520 <= pos[1] <= 570):
        fan_state = not fan_state
        if fan_state:
            meter_value = last_intensity
        else:
            last_intensity = meter_value
            meter_value = 0
        send_state_update()

##########################
# Future IoT Integration
##########################
# For actual hardware control, you could add code to send
# MQTT messages or HTTP requests to an ESP32 or other microcontroller
# whenever fan_state / meter_value changes.

if __name__ == "__main__":
    main()
