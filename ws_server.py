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
from websockets.exceptions import ConnectionClosed

# For system volume control (Windows)
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# For Linux volume control (uncomment if needed)
# import alsaaudio

##########################
# Gesture Recognition Setup
##########################
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Using default detection & tracking confidences.
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def is_hand_open(hand_landmarks):
    """Return True if at least three fingers are extended."""
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]
    open_fingers = sum(1 for tip, pip in fingers
                       if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    return open_fingers >= 3


##########################
# System Volume Control Setup
##########################
# Windows Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol = volume_interface.GetVolumeRange()[:2]


# Linux Volume Control (uncomment if needed)
# mixer = alsaaudio.Mixer()

def set_system_volume(volume_percent):
    """Set system volume (0-100) based on OS."""
    try:
        # Windows
        volume_scalar = np.interp(volume_percent, [0, 100], [min_vol, max_vol])
        volume_interface.SetMasterVolumeLevel(volume_scalar, None)
    except Exception as e:
        # Linux (uncomment if needed)
        # mixer.setvolume(int(volume_percent))
        pass


##########################
# Pygame UI Setup
##########################
pygame.init()
screen_width, screen_height = 1024, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Gesture Control System")

# UI constants
CAM_WIDTH, CAM_HEIGHT = 640, 480
UI_WIDTH = screen_width - CAM_WIDTH
PANEL_PADDING = 20

# Colors
BG_COLOR = (30, 30, 40)
PRIMARY_COLOR = (76, 175, 80)
ACCENT_COLOR = (255, 193, 7)
TEXT_COLOR = (255, 255, 255)

# Application state
meter_value = 100  # Fan intensity (0-100)
fan_state = False  # Fan power state
last_intensity = 100

volume_level = 50  # System volume (0-100)

##########################
# Camera Setup
##########################
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Gesture control state variables
previous_fan_palm_state = None
previous_fan_pinch_state = None
previous_vol_pinch_state = None

##########################
# WebSocket Server Setup
##########################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("websocket_server")

connected_clients = set()
connected_clients_lock = threading.Lock()
ws_loop = None  # The asyncio event loop for the server


async def websocket_handler(websocket):
    """Handler for new WebSocket connections."""
    global connected_clients
    try:
        with connected_clients_lock:
            connected_clients.add(websocket)
            logger.info(f"Client connected: {websocket.remote_address} (path: {websocket.path})")
    except Exception as ex:
        logger.exception("Error adding client:")

    try:
        async for message in websocket:
            logger.debug(f"Received message from {websocket.remote_address}: {message}")
            # Echo back message for testing
            await websocket.send("echo: " + message)
    except ConnectionClosed:
        logger.warning(f"Connection closed by client: {websocket.remote_address}")
    except Exception as e:
        logger.exception(f"Exception in connection with {websocket.remote_address}:")
    finally:
        try:
            with connected_clients_lock:
                connected_clients.discard(websocket)
            logger.info(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            logger.exception("Error removing client:")


async def start_server():
    global ws_loop
    ws_loop = asyncio.get_running_loop()
    server = await websockets.serve(websocket_handler, "localhost", 8765)
    logger.info("Server started on ws://localhost:8765")
    await asyncio.Future()  # Run forever


def run_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(start_server())
    except Exception as e:
        logger.exception("Exception while starting the WebSocket server:")
    loop.run_forever()


ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
ws_thread.start()


##########################
# Messaging Functions
##########################
def send_state_update():
    """Send a JSON message with the current fan state and volume."""
    state = {
        "type": "state",
        "fan_state": fan_state,
        "meter_value": meter_value,
        "volume_level": volume_level
    }
    message = json.dumps(state)

    async def send_message():
        with connected_clients_lock:
            for client in list(connected_clients):
                try:
                    await client.send(message)
                except Exception as e:
                    logger.error(f"Error sending state to client {client.remote_address}: {e}")
                    connected_clients.discard(client)

    if ws_loop is not None:
        asyncio.run_coroutine_threadsafe(send_message(), ws_loop)


def send_camera_frame(frame):
    """Encode the frame as JPEG and send it to clients with a header."""
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        return

    header = b"frame:"  # Header indicating a camera frame
    payload = header + jpeg.tobytes()

    async def send_frame():
        with connected_clients_lock:
            for client in list(connected_clients):
                try:
                    await client.send(payload)
                except Exception as e:
                    logger.error(f"Error sending frame to client {client.remote_address}: {e}")
                    connected_clients.discard(client)

    if ws_loop is not None:
        asyncio.run_coroutine_threadsafe(send_frame(), ws_loop)


##########################
# Gesture Detection Functions
##########################
def detect_gestures(frame):
    """
    Process the frame and update states:
      - Right hand: Fan control via palm open (on/off) and thumb-index pinch (intensity).
      - Left hand: Volume control via mapping the pinch distance to system volume.
    """
    global meter_value, fan_state, last_intensity, volume_level
    global previous_fan_palm_state, previous_fan_pinch_state, previous_vol_pinch_state

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Use MediaPipe handedness output directly.
            handedness = results.multi_handedness[idx].classification[0].label

            if handedness == 'Right':  # Fan control
                current_palm_open = is_hand_open(hand_landmarks)
                # Toggle fan state if the palm open state changes.
                if previous_fan_palm_state is not None and current_palm_open != previous_fan_palm_state:
                    fan_state = current_palm_open
                    meter_value = 100 if current_palm_open else meter_value  # Do not force 0 if already lower
                    last_intensity = 100 if current_palm_open else last_intensity
                    logger.info(f"Fan toggled {'ON' if fan_state else 'OFF'}")
                    send_state_update()
                previous_fan_palm_state = current_palm_open

                # Adjust fan intensity with thumb-index pinch if fan is on.
                if fan_state:
                    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    distance = np.hypot(thumb.x - index_finger.x, thumb.y - index_finger.y)

                    current_pinch = None
                    if distance < 0.05:
                        current_pinch = 'pinch'
                    elif distance > 0.2:
                        current_pinch = 'spread'

                    # Only update if a valid gesture is detected and it changed from the previous state.
                    if current_pinch != previous_fan_pinch_state and current_pinch is not None:
                        # Use a delta of 25 for fan intensity.
                        delta = 25 if current_pinch == 'spread' else -25 if current_pinch == 'pinch' else 0
                        new_meter = int(np.clip(meter_value + delta, 0, 100))
                        if new_meter != meter_value:
                            meter_value = new_meter
                            logger.info(f"Fan intensity adjusted by {delta} to {meter_value}%")
                            send_state_update()
                        previous_fan_pinch_state = current_pinch
                    if current_pinch is None:
                        previous_fan_pinch_state = None

            elif handedness == 'Left':  # Volume control
                # Use thumb-index pinch to control system volume.
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distance = np.hypot(thumb.x - index_finger.x, thumb.y - index_finger.y)

                # Map the distance (0.05 to 0.3) to volume percentage (0-100)
                new_volume = int(np.interp(distance, [0.05, 0.3], [0, 100]))
                new_volume = int(np.clip(new_volume, 0, 100))

                # Only update volume if the difference is significant.
                if abs(new_volume - volume_level) > 2:
                    volume_level = new_volume
                    set_system_volume(volume_level)
                    logger.info(f"Volume set to {volume_level}%")
                    send_state_update()
                    previous_vol_pinch_state = new_volume

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame


##########################
# UI Drawing Functions
##########################
def draw_meter(surface, x, y, width, height, value, active, label):
    """Draw a meter with a label."""
    pygame.draw.rect(surface, (50, 50, 60), (x, y, width, height), border_radius=10)
    fill_height = int((value / 100) * height)
    color = PRIMARY_COLOR if active else (100, 100, 100)
    pygame.draw.rect(surface, color, (x, y + height - fill_height, width, fill_height), border_radius=10)
    text = pygame.font.Font(None, 36).render(f"{value}%", True, TEXT_COLOR)
    surface.blit(text, (x + width // 2 - text.get_width() // 2, y + height // 2 - 10))
    label_text = pygame.font.Font(None, 24).render(label, True, TEXT_COLOR)
    surface.blit(label_text, (x, y - 30))


def draw_camera_panel():
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)
        frame = detect_gestures(frame)
        send_camera_frame(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        camera_surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(camera_surf, (0, (screen_height - CAM_HEIGHT) // 2))


def draw_ui():
    draw_camera_panel()
    control_panel = pygame.Surface((UI_WIDTH, screen_height))
    control_panel.fill(BG_COLOR)
    draw_meter(control_panel, PANEL_PADDING, 50, UI_WIDTH - 2 * PANEL_PADDING, 150,
               meter_value, fan_state, "Fan Control (Right Hand)")
    draw_meter(control_panel, PANEL_PADDING, 250, UI_WIDTH - 2 * PANEL_PADDING, 150,
               volume_level, True, "Volume Control (Left Hand)")
    status_font = pygame.font.Font(None, 24)
    fan_status = status_font.render(f"Fan: {'RUNNING' if fan_state else 'STOPPED'}",
                                    True, PRIMARY_COLOR if fan_state else (100, 100, 100))
    vol_status = status_font.render(f"System Volume: {volume_level}%", True, PRIMARY_COLOR)
    control_panel.blit(fan_status, (PANEL_PADDING, 210))
    control_panel.blit(vol_status, (PANEL_PADDING, 410))
    screen.blit(control_panel, (CAM_WIDTH, 0))


def draw_button(surface, text, x, y, width, height, color):
    mouse = pygame.mouse.get_pos()
    btn_rect = pygame.Rect(CAM_WIDTH + x, y, width, height)
    if btn_rect.collidepoint(mouse):
        color = tuple(min(c + 30, 255) for c in color)
    pygame.draw.rect(surface, color, (x, y, width, height), border_radius=5)
    text_surf = pygame.font.Font(None, 36).render(text, True, TEXT_COLOR)
    surface.blit(text_surf, (x + (width - text_surf.get_width()) // 2,
                             y + (height - text_surf.get_height()) // 2))


##########################
# Main Loop
##########################
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if mouse_pos[0] > CAM_WIDTH:
                btn_x = mouse_pos[0] - CAM_WIDTH
                if PANEL_PADDING <= btn_x <= PANEL_PADDING + 120 and 450 <= mouse_pos[1] <= 500:
                    if fan_state:
                        meter_value = min(meter_value + 25, 100)
                        send_state_update()
                elif PANEL_PADDING * 2 + 120 <= btn_x <= PANEL_PADDING * 2 + 240 and 450 <= mouse_pos[1] <= 500:
                    if fan_state:
                        meter_value = max(meter_value - 25, 0)
                        send_state_update()
                elif PANEL_PADDING <= btn_x <= UI_WIDTH - PANEL_PADDING and 520 <= mouse_pos[1] <= 570:
                    fan_state = not fan_state
                    if fan_state:
                        meter_value = 100
                        last_intensity = 100
                    else:
                        last_intensity = meter_value
                        meter_value = 0
                    send_state_update()

    screen.fill(BG_COLOR)
    draw_ui()
    pygame.display.flip()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()
