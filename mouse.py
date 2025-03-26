import cv2
import numpy as np
import pyautogui
import mediapipe as mp

# Setup
tracked_triangle = None
tracked_center = None
prev_center = None
dot_present_last_frame = True
near_edge_last_frame = False
smoothing_factor = 0.2

FRAME_EDGE_MARGIN = 60  # pixels to define "near edge"

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

def find_best_triangle(frame, prev_triangle):
    preprocessed = preprocess_image(frame)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame.shape[0] * frame.shape[1]
    best_match = None
    best_center = None
    best_score = float('inf')

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
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Match by shape similarity and distance
        if prev_triangle is not None:
            score = cv2.matchShapes(prev_triangle, approx, 1, 0.0)
            distance = np.linalg.norm(np.array([cx, cy]) - np.array(tracked_center))
            score += distance / 100  # combine shape + spatial similarity
        else:
            score = 0

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

        move_x = int(dx * smoothing_factor) * 2
        move_y = int(dy * smoothing_factor) * -2

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

def detect_dot_inside_triangle(frame, triangle_contour, debug_frame=None):
    """
    Detects a dark dot inside the triangle using intensity thresholding.
    Optionally shows the detection mask on debug_frame.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [triangle_contour], -1, 255, -1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    triangle_region = cv2.bitwise_and(gray, gray, mask=mask)

    # Use adaptive or fixed threshold to detect darker regions
    _, dot_thresh = cv2.threshold(triangle_region, 60, 255, cv2.THRESH_BINARY_INV)
    dot_thresh = cv2.GaussianBlur(dot_thresh, (3, 3), 0)
    cv2.imshow("Dot Detection Mask", dot_thresh)

    contours, _ = cv2.findContours(dot_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_dot = False
    for c in contours:
        area = cv2.contourArea(c)
        if 30 < area < 1000:  # tuned for ~10–30px radius dots
            (x, y, w, h) = cv2.boundingRect(c)
            if debug_frame is not None:
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(debug_frame, "DOT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            found_dot = True

    if debug_frame is not None:
        cv2.imshow("Dot Detection Mask", dot_thresh)

    return found_dot

# Start video
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

    # Reset with R key
    if key == ord('r'):
        tracked_triangle = None
        tracked_center = None
        prev_center = None
        dot_present_last_frame = True
        near_edge_last_frame = False
        print("Reset tracking.")

    if key == ord('q'):
        break

    # Update triangle each frame
    tracked_triangle, tracked_center = find_best_triangle(frame, tracked_triangle)

    if tracked_triangle is not None:
        cx, cy = tracked_center
        update_mouse(tracked_center)
        cv2.drawContours(display_frame, [tracked_triangle], -1, (255, 255, 0), 3)
        cv2.circle(display_frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(display_frame, "TRIANGLE", (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Detect dot inside triangle
        dot_present = detect_dot_inside_triangle(frame, tracked_triangle, display_frame)
        near_edge = is_near_frame_edge(tracked_center, frame.shape)

        # If dot disappeared, and triangle is stable → click
        if not dot_present and (not near_edge and not near_edge_last_frame):
            pyautogui.click()
            cv2.putText(display_frame, "CLICK", (cx, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        dot_present_last_frame = dot_present
        near_edge_last_frame = near_edge

    cv2.imshow("Dot Occlusion Click Tracker", display_frame)

cap.release()
cv2.destroyAllWindows()
