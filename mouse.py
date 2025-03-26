import cv2
import numpy as np
import pyautogui
from collections import deque
import time

# Setup
tracked_triangle = None
tracked_center = None
prev_center = None
locked_area = None
locked_ratio = None
dot_present_last_frame = True
near_edge_last_frame = False
smoothing_factor = 0.2

FRAME_EDGE_MARGIN = 60
MAX_HISTORY = 8
shape_history = deque(maxlen=MAX_HISTORY)

SENSITIVITY_INCREMENT = 0.05
MIN_SENSITIVITY = 0.05
MAX_SENSITIVITY = 1.0

cv2.setUseOptimized(True)

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

def merge_contours(contours):
    if len(contours) < 10:
        return contours

    filtered = [c for c in contours if cv2.contourArea(c) > 500]
    merged = []
    used = set()
    boxes = [cv2.boundingRect(c) for c in filtered]

    for i, rect1 in enumerate(boxes):
        if i in used:
            continue
        x1, y1, w1, h1 = rect1
        merged_contour = filtered[i]
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            x2, y2, w2, h2 = boxes[j]
            if (
                x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2 and
                max(w1, h1, w2, h2) < 300
            ):
                merged_contour = np.vstack((merged_contour, filtered[j]))
                used.add(j)
                break
        used.add(i)
        merged.append(cv2.convexHull(merged_contour))
    return merged

def find_best_triangle(frame, prev_triangle, prev_center, locked_area=None, locked_ratio=None):
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

        # Shape locking check
        if locked_area is not None and locked_ratio is not None:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / h if h != 0 else 0
            area_ratio = abs(area - locked_area) / locked_area
            ratio_diff = abs(ratio - locked_ratio)
            if area_ratio > 0.25 or ratio_diff > 0.2:
                continue  # Skip dissimilar triangles

        score = 0
        if prev_triangle is not None and prev_center is not None:
            shape_score = cv2.matchShapes(prev_triangle, approx, 1, 0.0)
            dist_score = np.linalg.norm(np.array(prev_center) - np.array([cx, cy])) / 100
            size_diff = abs(area - cv2.contourArea(prev_triangle)) / frame_area
            score = shape_score + dist_score + size_diff
        else:
            score = area

        if score < best_score:
            best_score = score
            best_match = approx
            best_center = (cx, cy)
            current_area = area
            x, y, w, h = cv2.boundingRect(approx)
            current_ratio = w / h if h != 0 else 0

    if best_match is not None:
        return best_match, best_center, current_area, current_ratio
    return None, None, None, None

def update_mouse(center):
    global prev_center
    if prev_center is not None:
        dx = center[0] - prev_center[0]
        dy = center[1] - prev_center[1]

        move_x = int(dx * smoothing_factor) * 2
        move_y = int(dy * smoothing_factor) * -2

        screen_width, screen_height = pyautogui.size()
        current_mouse_x, current_mouse_y = pyautogui.position()

        new_mouse_x = min(max(current_mouse_x + move_x, 0), screen_width - 1)
        new_mouse_y = min(max(current_mouse_y + move_y, 0), screen_height - 1)

        pyautogui.moveTo(new_mouse_x, new_mouse_y)

    prev_center = center

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

    # Sensitivity adjustment
    if key == ord('+') or key == ord('='):
        smoothing_factor = min(smoothing_factor + SENSITIVITY_INCREMENT, MAX_SENSITIVITY)
        print(f"Sensitivity increased: {smoothing_factor:.2f}")
    elif key == ord('-') or key == ord('_'):
        smoothing_factor = max(smoothing_factor - SENSITIVITY_INCREMENT, MIN_SENSITIVITY)
        print(f"Sensitivity decreased: {smoothing_factor:.2f}")

    # Reset
    if key == ord('r'):
        tracked_triangle = None
        tracked_center = None
        prev_center = None
        locked_area = None
        locked_ratio = None
        shape_history.clear()
        print("Reset tracking.")

    if key == ord('q'):
        break

    triangle, center, area, ratio = find_best_triangle(
        frame, tracked_triangle, tracked_center, locked_area, locked_ratio
    )

    if triangle is not None:
        shape_history.append(triangle)
        tracked_triangle = triangle
        tracked_center = center
        locked_area = area
        locked_ratio = ratio
        update_mouse(center)

        cv2.drawContours(display_frame, [tracked_triangle], -1, (255, 255, 0), 3)
        cv2.circle(display_frame, tracked_center, 5, (0, 255, 255), -1)
        cv2.putText(display_frame, "TRIANGLE", (center[0], center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        # Don't replace triangle unless similar shape is found (handled in find_best_triangle)
        if tracked_triangle is not None:
            pyautogui.click()
        tracked_triangle = None
        tracked_center = None

    cv2.imshow("Triangle Mouse Tracker", display_frame)
    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
