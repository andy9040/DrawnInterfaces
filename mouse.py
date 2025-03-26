import cv2
import numpy as np
from collections import deque
import pyautogui
import mediapipe as mp

# Initialize shape history and previous center point
shape_history = {}
history_length = 10
prev_center = None
smoothing_factor = 0.2  # Adjusts cursor movement sensitivity

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def preprocess_image(frame):
    """Preprocess image: grayscale, blur, threshold, and clean."""
    cv2.imshow('Round 0', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('Round 1', blurred)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    cv2.imshow('Round 2', thresh)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow('Round 3', cleaned)

    return cleaned

def merge_contours(contours):
    """Merges small overlapping contours to avoid duplicates."""
    merged_contours = []
    contour_boxes = [cv2.boundingRect(c) for c in contours]
    used = set()

    for i, rect1 in enumerate(contour_boxes):
        if i in used:
            continue

        x1, y1, w1, h1 = rect1
        merged = contours[i]

        for j, rect2 in enumerate(contour_boxes):
            if i != j and j not in used:
                x2, y2, w2, h2 = rect2
                if (x1 < x2 + w2 and x1 + w1 > x2 and
                    y1 < y2 + h2 and y1 + h1 > y2 and max(w1, h1, w2, h2) < 300):
                    merged = np.vstack((merged, contours[j]))
                    used.add(j)

        used.add(i)
        merged_contours.append(cv2.convexHull(merged))

    return merged_contours

def detect_shapes(frame, hand_landmarks):
    global shape_history, prev_center

    preprocessed = preprocess_image(frame)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = merge_contours(contours)

    frame_area = frame.shape[0] * frame.shape[1]

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        num_sides = len(approx)

        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)

        if area < 1000 or w < 50 or h < 50:
            continue
        if area > frame_area * 0.6:
            continue

        if num_sides == 3:
            shape = "TRIANGLE"
            color = (255, 255, 0)

            # Calculate center of triangle
            M = cv2.moments(approx)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Draw triangle and center
            cv2.drawContours(frame, [approx], -1, color, 3)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Mouse movement
            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]

                move_x = int(dx * smoothing_factor) * -1
                move_y = int(dy * smoothing_factor) * -1

                screen_width, screen_height = pyautogui.size()
                current_mouse_x, current_mouse_y = pyautogui.position()

                new_mouse_x = min(max(current_mouse_x + move_x, 0), screen_width - 1)
                new_mouse_y = min(max(current_mouse_y + move_y, 0), screen_height - 1)

                pyautogui.moveTo(new_mouse_x, new_mouse_y)

            prev_center = (cx, cy)

            # Click detection: Check if finger tip is near triangle center
            if hand_landmarks:
                finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                fx, fy = int(finger_tip.x * w), int(finger_tip.y * h)

                cv2.circle(frame, (fx, fy), 5, (255, 0, 255), -1)
                distance = np.linalg.norm([cx - fx, cy - fy])

                if distance < 40:  # Trigger click if close enough
                    pyautogui.click()
                    cv2.putText(frame, "CLICK", (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            continue

    return frame

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Flip for natural webcam feel
    frame = cv2.flip(frame, 1)

    # Run hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_landmarks = None
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            hand_landmarks = handLms
            break  # Use only the first hand

    processed_frame = detect_shapes(frame, hand_landmarks)
    cv2.imshow('Improved Shape Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
