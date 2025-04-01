import cv2
import numpy as np
import pyautogui
from collections import deque

# State
tracked_square = None
tracked_center = None
prev_center = None
smoothing_factor = 0.4
missed_frames = 0
MISS_FRAME_THRESHOLD = 5

FRAME_EDGE_MARGIN = 60
MAX_HISTORY = 8
shape_history = deque(maxlen=MAX_HISTORY)

WARP_WIDTH = 400
WARP_HEIGHT = 400

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

def order_points(pts):
    """ Orders contour points in top-left, top-right, bottom-right, bottom-left order """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def get_perspective_warp(frame, square):
    src_pts = order_points(square.reshape(4, 2).astype(np.float32))
    dst_pts = np.array([
        [0, 0],
        [WARP_WIDTH - 1, 0],
        [WARP_WIDTH - 1, WARP_HEIGHT - 1],
        [0, WARP_HEIGHT - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (WARP_WIDTH, WARP_HEIGHT))
    return warped, M

def find_best_square(frame):
    preprocessed = preprocess_image(frame)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_square = None
    best_center = None
    max_area = 0
    frame_area = frame.shape[0] * frame.shape[1]

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        if area < 1000 or area > frame_area * 0.8:
            continue

        if not cv2.isContourConvex(approx):
            continue

        # Optional: check for squareness (right angles)
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        # if aspect_ratio < 0.75 or aspect_ratio > 1.25:
        #     continue

        if area > max_area:
            max_area = area
            best_square = approx

            M = cv2.moments(approx)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_center = (cx, cy)

    return best_square, best_center

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
        tracked_square = None
        tracked_center = None
        prev_center = None
        shape_history.clear()
        missed_frames = 0
        print("Reset tracking.")

    if key == ord('q'):
        break

    square, center = find_best_square(frame)

    if square is not None:
        missed_frames = 0
        tracked_square = square
        tracked_center = center
        shape_history.append(square)
        update_mouse(center)

        warped, _ = get_perspective_warp(frame, square)

        # Show both original and warped views
        cv2.drawContours(display_frame, [square], -1, (0, 255, 255), 3)
        cv2.circle(display_frame, tracked_center, 5, (255, 0, 0), -1)
        cv2.putText(display_frame, "SQUARE", (center[0], center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Warped Top-Down View", warped)
    else:
        missed_frames += 1
        if missed_frames >= MISS_FRAME_THRESHOLD and tracked_square is not None:
            pyautogui.click()
            tracked_square = None
            tracked_center = None
            prev_center = None

    cv2.imshow("Square Tracker", display_frame)

cap.release()
cv2.destroyAllWindows()
