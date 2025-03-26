import cv2
import numpy as np
import pyautogui
from collections import deque

# State
tracked_triangle = None
tracked_center = None
prev_center = None
dot_present_last_frame = True
near_edge_last_frame = False
smoothing_factor = 0.4  # Increased for faster movement
missed_frames = 0
MISS_FRAME_THRESHOLD = 5

FRAME_EDGE_MARGIN = 60
MAX_HISTORY = 8
shape_history = deque(maxlen=MAX_HISTORY)

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

def merge_contours(contours):
    merged = []
    used = set()
    boxes = [cv2.boundingRect(c) for c in contours]

    for i, rect1 in enumerate(boxes):
        if i in used:
            continue
        x1, y1, w1, h1 = rect1
        merged_contour = contours[i]
        for j, rect2 in enumerate(boxes):
            if i != j and j not in used:
                x2, y2, w2, h2 = rect2
                if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
                    if max(w1, h1, w2, h2) < 300:
                        merged_contour = np.vstack((merged_contour, contours[j]))
                        used.add(j)
        used.add(i)
        merged.append(cv2.convexHull(merged_contour))
    return merged

def find_best_triangle(frame, prev_triangle, prev_center):
    preprocessed = preprocess_image(frame)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = merge_contours(contours)

    best_match = None
    best_center = None
    best_score = float('inf')
    frame_area = frame.shape[0] * frame.shape[1]

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 3:
            continue

        area = cv2.contourArea(approx)
        if area < 1000 or area > frame_area * 0.6:
            continue

        M = cv2.moments(approx)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        if prev_triangle is not None and prev_center is not None:
            shape_score = cv2.matchShapes(prev_triangle, approx, 1, 0.0)
            if shape_score > 0.3:
                continue  # Too different in shape

            dist_score = np.linalg.norm(np.array(prev_center) - np.array([cx, cy]))
            if dist_score > 100:
                continue  # Too far from last known center

            size_diff = abs(area - cv2.contourArea(prev_triangle)) / frame_area
            score = shape_score + (dist_score / 100) + size_diff
        else:
            score = area  # fallback when starting

        if score < best_score:
            best_score = score
            best_match = approx
            best_center = (cx, cy)

    return best_match, best_center

def update_mouse(center):
    global prev_center
    if prev_center is not None:
        dx = center[0] - prev_center[0]
        dy = center[1] - prev_center[1]

        move_x = int(dx * smoothing_factor * 4)
        move_y = int(dy * smoothing_factor * -4)

        screen_width, screen_height = pyautogui.size()
        current_mouse_x, current_mouse_y = pyautogui.position()

        new_mouse_x = min(max(current_mouse_x + move_x, 0), screen_width - 1)
        new_mouse_y = min(max(current_mouse_y + move_y, 0), screen_height - 1)

        pyautogui.moveTo(new_mouse_x, new_mouse_y)

    prev_center = center

def is_near_frame_edge(center, frame_shape):
    h, w = frame_shape[:2]
    x, y = center
    return (
        x < FRAME_EDGE_MARGIN or x > (w - FRAME_EDGE_MARGIN) or
        y < FRAME_EDGE_MARGIN or y > (h - FRAME_EDGE_MARGIN)
    )

# Main loop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        tracked_triangle = None
        tracked_center = None
        prev_center = None
        dot_present_last_frame = True
        near_edge_last_frame = False
        shape_history.clear()
        missed_frames = 0
        print("Reset tracking.")

    if key == ord('q'):
        break

    triangle, center = find_best_triangle(frame, tracked_triangle, tracked_center)

    if triangle is not None:
        missed_frames = 0
        tracked_triangle = triangle
        tracked_center = center
        shape_history.append(triangle)
        update_mouse(center)

        cv2.drawContours(display_frame, [tracked_triangle], -1, (255, 255, 0), 3)
        cv2.circle(display_frame, tracked_center, 5, (0, 255, 255), -1)
        cv2.putText(display_frame, "TRIANGLE", (center[0], center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        missed_frames += 1
        if missed_frames >= MISS_FRAME_THRESHOLD and tracked_triangle is not None:
            pyautogui.click()
            tracked_triangle = None
            tracked_center = None
            prev_center = None

    cv2.imshow("Stable Triangle Tracker", display_frame)

cap.release()
cv2.destroyAllWindows()
