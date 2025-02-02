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

##########################
# Gesture Recognition Setup
##########################
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

# Application state (for IoT commands)
meter_value = 0  # native integer (0-100)
fan_state = False
last_intensity = 100

##########################
# Camera Setup
##########################
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Gesture control states
previous_palm_state = None
previous_pinch_state = None

##########################
# WebSocket Server Setup
##########################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("websocket_server")

# Global variables (set later in the server event loop)
connected_clients = set()
# Use a threading.Lock since the shared data is accessed from both async callbacks and the main thread.
connected_clients_lock = threading.Lock()
ws_loop = None  # The asyncio event loop for the server

# Updated handler: Only one parameter is expected (the websocket).
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
            # For testing, echo back the message to keep the connection alive.
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
    """Starts the WebSocket server and sets the event loop."""
    global ws_loop
    ws_loop = asyncio.get_running_loop()
    # Note: The handler now accepts only one parameter.
    server = await websockets.serve(websocket_handler, "localhost", 8765)
    logger.info("Server started on ws://localhost:8765")
    # Run forever
    await asyncio.Future()  # This future will never complete

def run_websocket_server():
    """Run the WebSocket server in its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(start_server())
    except Exception as e:
        logger.exception("Exception while starting the WebSocket server:")
    loop.run_forever()

# Start the WebSocket server in a separate thread.
ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
ws_thread.start()

##########################
# Messaging Functions
##########################
def send_state_update():
    """Send a JSON message with the current state (e.g., commands to IoT)."""
    state = {
        "type": "state",
        "fan_state": fan_state,
        "meter_value": meter_value
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
    """Encode the frame as JPEG and send it to clients with a header to indicate a camera frame."""
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        return

    header = b"frame:"  # Header to identify the message as a camera frame
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
def is_hand_open(hand_landmarks):
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
    global meter_value, fan_state, previous_palm_state, previous_pinch_state, last_intensity
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            current_palm_open = is_hand_open(hand_landmarks)

            if previous_palm_state is not None and current_palm_open != previous_palm_state:
                if current_palm_open:
                    fan_state = True
                    meter_value = 100
                    last_intensity = 100
                else:
                    fan_state = False
                    last_intensity = meter_value
                    meter_value = 0
                send_state_update()
                previous_palm_state = current_palm_open
            else:
                previous_palm_state = current_palm_open

            if fan_state:
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distance = np.hypot(thumb.x - index.x, thumb.y - index.y)
                current_pinch = 'pinch' if distance < 0.05 else 'spread' if distance > 0.2 else None
                if current_pinch != previous_pinch_state:
                    delta = 25 if current_pinch == 'spread' else -25 if current_pinch == 'pinch' else 0
                    meter_value = int(np.clip(meter_value + delta, 0, 100))
                    send_state_update()
                    previous_pinch_state = current_pinch
    return frame

##########################
# UI Drawing Functions
##########################
def draw_ui():
    draw_camera_panel()
    control_panel = pygame.Surface((UI_WIDTH, screen_height))
    control_panel.fill(BG_COLOR)
    draw_meter(control_panel, PANEL_PADDING, 50, UI_WIDTH - 2 * PANEL_PADDING, 300)
    draw_fan_status(control_panel, PANEL_PADDING, 380)
    btn_width = (UI_WIDTH - 3 * PANEL_PADDING) // 2
    draw_button(control_panel, "+25", PANEL_PADDING, 450, btn_width, 50, PRIMARY_COLOR)
    draw_button(control_panel, "-25", PANEL_PADDING * 2 + btn_width, 450, btn_width, 50, (244, 67, 54))
    draw_button(control_panel, "TOGGLE POWER", PANEL_PADDING, 520, UI_WIDTH - 2 * PANEL_PADDING, 50, ACCENT_COLOR)
    screen.blit(control_panel, (CAM_WIDTH, 0))

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

def draw_meter(surface, x, y, width, height):
    pygame.draw.rect(surface, (50, 50, 60), (x, y, width, height), border_radius=10)
    fill_height = int((meter_value / 100) * height)
    pygame.draw.rect(surface, PRIMARY_COLOR if fan_state else (100, 100, 100),
                     (x, y + height - fill_height, width, fill_height), border_radius=10)
    text = pygame.font.Font(None, 36).render(f"{meter_value}%", True, TEXT_COLOR)
    surface.blit(text, (x + width // 2 - text.get_width() // 2, y + height // 2 - 10))

def draw_fan_status(surface, x, y):
    radius = 30
    color = ACCENT_COLOR if fan_state else (100, 100, 100)
    pygame.draw.circle(surface, color, (x + 50, y), radius, 3 if not fan_state else 0)
    if fan_state:
        for i in range(3):
            angle = i * 120 + (pygame.time.get_ticks() // 10 % 360)
            end_x = x + 50 + radius * 0.8 * np.cos(np.radians(angle))
            end_y = y + radius * 0.8 * np.sin(np.radians(angle))
            pygame.draw.line(surface, ACCENT_COLOR, (x + 50, y), (end_x, end_y), 3)
    status_text = pygame.font.Font(None, 36).render("RUNNING" if fan_state else "STOPPED", True, TEXT_COLOR)
    surface.blit(status_text, (x, y + 50))

def draw_button(surface, text, x, y, width, height, color):
    mouse = pygame.mouse.get_pos()
    btn_rect = pygame.Rect(CAM_WIDTH + x, y, width, height)
    if btn_rect.collidepoint(mouse):
        # Slightly brighten the color on hover.
        color = tuple(min(c + 30, 255) for c in color)
    pygame.draw.rect(surface, color, (x, y, width, height), border_radius=5)
    text_surf = pygame.font.Font(None, 36).render(text, True, TEXT_COLOR)
    surface.blit(text_surf, (x + (width - text_surf.get_width()) // 2, y + (height - text_surf.get_height()) // 2))

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
                # Check for +25 button
                if PANEL_PADDING <= btn_x <= PANEL_PADDING + 120 and 450 <= mouse_pos[1] <= 500:
                    if fan_state:
                        meter_value = min(meter_value + 25, 100)
                        send_state_update()
                # Check for -25 button
                elif PANEL_PADDING * 2 + 120 <= btn_x <= PANEL_PADDING * 2 + 240 and 450 <= mouse_pos[1] <= 500:
                    if fan_state:
                        meter_value = max(meter_value - 25, 0)
                        send_state_update()
                # Check for TOGGLE POWER button
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
