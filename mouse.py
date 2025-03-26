import cv2
import numpy as np
from collections import deque
import pyautogui
import mediapipe as mp

# Triangle tracking variables
tracked_triangle = None
tracked_center = None
prev_center = None
smoothing_factor = 0.2

# Mediapipe finger-only tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def preprocess_image(frame):
    """Preprocess image: grayscale, blur, threshold, and clean."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

def detect_initial_triangle(frame):
    """Detects the first triangle and returns its contour and center."""
    preprocessed = preprocess_image(frame)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame.shape[0] * frame.shape[1]
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 3:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)

        if area < 1000 or w < 50 or h < 50 or area > frame_area * 0.6:
            continue

        M = cv2.moments(approx)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return approx, (cx, cy)

    return None, None

def update_mouse_position(center):
    global prev_center
    if prev_center is not None:
        dx = center[0] - prev_center[0]
        dy = center[1] - prev_center[1]

        move_x = int(dx * smoothing_factor) *2
        move_y = int(dy * smoothing_factor) * -2

        screen_width, screen_height = pyautogui.size()
        current_mouse_x, current_mouse_y = pyautogui.position()

        new_mouse_x = min(max(current_mouse_x + move_x, 0), screen_width - 1)
        new_mouse_y = min(max(current_mouse_y + move_y, 0), screen_height - 1)

        pyautogui.moveTo(new_mouse_x, new_mouse_y)

    prev_center = center

def detect_finger_tip(frame):
    """Returns index fingertip position if detected."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        h, w, _ = frame.shape
        return int(index_tip.x * w), int(index_tip.y * h)

    return None

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    key = cv2.waitKey(1) & 0xFF

    # R = Reset tracking
    if key == ord('r'):
        tracked_triangle = None
        tracked_center = None
        prev_center = None
        print("Tracking reset.")

    # Q = Quit
    if key == ord('q'):
        break

    if tracked_triangle is None:
        triangle, center = detect_initial_triangle(frame)
        if triangle is not None:
            tracked_triangle = triangle
            tracked_center = center
            print("Triangle locked.")
    else:
        # Draw the tracked triangle
        cv2.drawContours(display_frame, [tracked_triangle], -1, (255, 255, 0), 3)
        cx, cy = tracked_center
        cv2.circle(display_frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(display_frame, "TRIANGLE", (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        update_mouse_position(tracked_center)

        finger = detect_finger_tip(frame)
        if finger:
            fx, fy = finger
            cv2.circle(display_frame, (fx, fy), 5, (255, 0, 255), -1)
            distance = np.linalg.norm([cx - fx, cy - fy])
            if distance < 40:
                pyautogui.click()
                cv2.putText(display_frame, "CLICK", (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Finger-Controlled Triangle Tracker", display_frame)

cap.release()
cv2.destroyAllWindows()
