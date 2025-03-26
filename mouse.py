import cv2
import numpy as np
import pyautogui

# Tracking state
tracked_shape = None
tracked_center = None
prev_center = None
last_known_center = None
last_seen_frame = 0
frame_count = 0

# Responsiveness & controls
smoothing_factor = 0.4
EDGE_MARGIN = 60
DISAPPEAR_TIMEOUT = 5
CLICK_COOLDOWN = 10
last_click_frame = -100

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned

def find_main_shape(frame, prev_shape, prev_center):
    preprocessed = preprocess_image(frame)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    best_match = None
    best_center = None
    best_score = float('inf')

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000 or area > (h * w * 0.6):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.6:  # Less circular = not oval
            continue

        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if prev_shape is not None:
            shape_score = cv2.matchShapes(prev_shape, approx, 1, 0.0)
            dist_score = np.linalg.norm(np.array([cx, cy]) - np.array(prev_center))
            score = shape_score + (dist_score / 100)
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

        move_x = int(dx * smoothing_factor)
        move_y = int(dy * smoothing_factor)

        screen_width, screen_height = pyautogui.size()
        current_x, current_y = pyautogui.position()

        new_x = min(max(current_x + move_x, 0), screen_width - 1)
        new_y = min(max(current_y + move_y, 0), screen_height - 1)

        pyautogui.moveTo(new_x, new_y)

    prev_center = center

def is_near_edge(center, frame_shape):
    x, y = center
    h, w = frame_shape[:2]
    return (
        x < EDGE_MARGIN or x > w - EDGE_MARGIN or
        y < EDGE_MARGIN or y > h - EDGE_MARGIN
    )

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
    frame_count += 1

    # Reset
    if key == ord('r'):
        tracked_shape = None
        tracked_center = None
        prev_center = None
        last_known_center = None
        last_seen_frame = 0
        print("Reset tracking.")

    if key == ord('q'):
        break

    # Detect oval
    shape, center = find_main_shape(frame, tracked_shape, tracked_center)

    if shape is not None:
        tracked_shape = shape
        tracked_center = center
        last_known_center = center
        last_seen_frame = frame_count

        update_mouse(center)
        cv2.drawContours(display_frame, [shape], -1, (0, 255, 255), 3)
        cv2.circle(display_frame, center, 5, (255, 0, 255), -1)
        cv2.putText(display_frame, "OVAL", (center[0], center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    else:
        frames_missing = frame_count - last_seen_frame
        if (
            last_known_center is not None
            and not is_near_edge(last_known_center, frame.shape)
            and frames_missing == DISAPPEAR_TIMEOUT
            and (frame_count - last_click_frame > CLICK_COOLDOWN)
        ):
            pyautogui.click()
            last_click_frame = frame_count
            print("CLICK!")
            cv2.putText(display_frame, "CLICK!", (last_known_center[0], last_known_center[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    cv2.imshow("Oval Click Tracker", display_frame)

cap.release()
cv2.destroyAllWindows()
