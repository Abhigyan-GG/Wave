# py -3.10 C:\Users\Hemant\Downloads\PYTHON\Wave\Gesture_control.py


import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Pygame
pygame.init()
screen_width, screen_height = 1024, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Gesture Control System")

# UI Constants
CAM_WIDTH, CAM_HEIGHT = 640, 480
UI_WIDTH = screen_width - CAM_WIDTH
PANEL_PADDING = 20

# Colors
BG_COLOR = (30, 30, 40)
PRIMARY_COLOR = (76, 175, 80)
ACCENT_COLOR = (255, 193, 7)
TEXT_COLOR = (255, 255, 255)

# Application State
meter_value = 0  # Start at 0% when off
fan_state = False  # Start switched off
last_intensity = 100  # Default to 100% when first turned on

# Initialize Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Gesture control states
previous_palm_state = None
previous_pinch_state = None


def draw_ui():
    # Camera panel
    draw_camera_panel()

    # Control panel
    control_panel = pygame.Surface((UI_WIDTH, screen_height))
    control_panel.fill(BG_COLOR)

    # Intensity Meter
    draw_meter(control_panel, PANEL_PADDING, 50, UI_WIDTH - 2 * PANEL_PADDING, 300)

    # Fan Status
    draw_fan_status(control_panel, PANEL_PADDING, 380)

    # Control Buttons
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        camera_surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(camera_surf, (0, (screen_height - CAM_HEIGHT) // 2))


def draw_meter(surface, x, y, width, height):
    pygame.draw.rect(surface, (50, 50, 60), (x, y, width, height), border_radius=10)
    fill_height = int((meter_value / 100) * height)
    pygame.draw.rect(surface, PRIMARY_COLOR if fan_state else (100, 100, 100),
                     (x, y + height - fill_height, width, fill_height),
                     border_radius=10)
    text = pygame.font.Font(None, 36).render(f"{meter_value}%", True, TEXT_COLOR)
    surface.blit(text, (x + width // 2 - text.get_width() // 2, y + height // 2 - 10))


def draw_fan_status(surface, x, y):
    radius = 30
    color = ACCENT_COLOR if fan_state else (100, 100, 100)
    pygame.draw.circle(surface, color, (x + 50, y), radius, 3 if not fan_state else 0)

    if fan_state:
        for i in range(3):
            angle = i * 120 + pygame.time.get_ticks() // 10 % 360
            end_x = x + 50 + radius * 0.8 * np.cos(np.radians(angle))
            end_y = y + radius * 0.8 * np.sin(np.radians(angle))
            pygame.draw.line(surface, ACCENT_COLOR, (x + 50, y), (end_x, end_y), 3)

    status_text = pygame.font.Font(None, 36).render("RUNNING" if fan_state else "STOPPED", True, TEXT_COLOR)
    surface.blit(status_text, (x, y + 50))


def draw_button(surface, text, x, y, width, height, color):
    mouse = pygame.mouse.get_pos()
    btn_rect = pygame.Rect(CAM_WIDTH + x, y, width, height)
    if btn_rect.collidepoint(mouse):
        color = tuple(min(c + 30, 255) for c in color)

    pygame.draw.rect(surface, color, (x, y, width, height), border_radius=5)
    text_surf = pygame.font.Font(None, 36).render(text, True, TEXT_COLOR)
    surface.blit(text_surf, (x + (width - text_surf.get_width()) // 2, y + (height - text_surf.get_height()) // 2))


def is_hand_open(hand_landmarks):
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]

    open_fingers = sum(1 for tip, pip in fingers
                       if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    return open_fingers >= 3


def detect_gestures(frame):
    global meter_value, fan_state, previous_palm_state, previous_pinch_state, last_intensity

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            current_palm_open = is_hand_open(hand_landmarks)
            if previous_palm_state is not None and current_palm_open != previous_palm_state:
                if current_palm_open:  # Fan turned ON
                    fan_state = True
                    meter_value = 100  # Always start at 100% when turned on
                    last_intensity = 100
                else:  # Fan turned OFF
                    fan_state = False
                    last_intensity = meter_value
                    meter_value = 0
                previous_palm_state = current_palm_open
            previous_palm_state = current_palm_open

            if fan_state:
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distance = np.hypot(thumb.x - index.x, thumb.y - index.y)

                current_pinch = 'pinch' if distance < 0.05 else 'spread' if distance > 0.2 else None
                if current_pinch != previous_pinch_state:
                    meter_value += 25 if current_pinch == 'spread' else -25 if current_pinch == 'pinch' else 0
                    meter_value = np.clip(meter_value, 0, 100)
                    previous_pinch_state = current_pinch

    return frame


running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if mouse_pos[0] > CAM_WIDTH:
                btn_x = mouse_pos[0] - CAM_WIDTH
                # +25 Button
                if PANEL_PADDING <= btn_x <= PANEL_PADDING + 120 and 450 <= mouse_pos[1] <= 500:
                    if fan_state:
                        meter_value = min(meter_value + 25, 100)
                # -25 Button
                elif PANEL_PADDING * 2 + 120 <= btn_x <= PANEL_PADDING * 2 + 240 and 450 <= mouse_pos[1] <= 500:
                    if fan_state:
                        meter_value = max(meter_value - 25, 0)
                # Toggle Button
                elif PANEL_PADDING <= btn_x <= UI_WIDTH - PANEL_PADDING and 520 <= mouse_pos[1] <= 570:
                    fan_state = not fan_state
                    if fan_state:
                        meter_value = 100  # Reset to 100% when turned on
                        last_intensity = 100
                    else:
                        last_intensity = meter_value
                        meter_value = 0

    screen.fill(BG_COLOR)
    draw_ui()
    pygame.display.flip()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()