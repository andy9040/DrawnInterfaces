import cv2
import numpy as np
import pyautogui
import math
import time

# === VideoCapture setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found")
    exit()

prev_center = None
JITTER_THRESHOLD = 5
SMOOTHING_ALPHA = 0.25
MAX_DISTANCE = 50
MIN_DURATION = 0.01
MAX_DURATION = 0.15

# Click detection
prev_visible = True
tap_ready = True
last_disappear_time = 0
click_cooldown = 0.5

# Motion accumulators
dx_accumulator = 0.0
dy_accumulator = 0.0

def find_green_square_corners(frame):
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.medianBlur(mask, 5)
    cv2.imshow("Green Mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        hull = cv2.convexHull(cnt)
        epsilon = 0.04 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) >= 4:
            M = cv2.moments(approx)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)

            points = approx.reshape(-1, 2)
            distances = np.linalg.norm(points - np.array(center), axis=1)
            top4_idx = np.argsort(distances)[-4:]
            top4 = points[top4_idx]
            return top4, center
    return None, None

def draw_tracking_info(frame, corners, center):
    for pt in corners:
        cv2.circle(frame, tuple(pt.astype(int)), 5, (255, 0, 0), -1)
    cv2.circle(frame, center, 8, (0, 255, 0), -1)
    cv2.putText(frame, f"Green center: {center}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

def map_distance_to_duration(dist):
    ratio = min(dist / MAX_DISTANCE, 1.0)
    return MAX_DURATION - (MAX_DURATION - MIN_DURATION) * ratio

print("Tracking started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    display_frame = frame.copy()
    corners, current_center = find_green_square_corners(frame)

    if corners is not None and current_center is not None:
        dx = dy = 0

        if not prev_visible:
            tap_ready = True
        prev_visible = True

        if prev_center is None:
            prev_center = current_center
        else:
            new_x, new_y = current_center
            prev_x, prev_y = prev_center

            if abs(new_x - prev_x) < JITTER_THRESHOLD:
                new_x = prev_x
            if abs(new_y - prev_y) < JITTER_THRESHOLD:
                new_y = prev_y

            dx = -(new_x - prev_x) * 3
            dy = -(new_y - prev_y) * 3
            prev_center = (new_x, new_y)

            dx_accumulator += dx * SMOOTHING_ALPHA
            dy_accumulator += dy * SMOOTHING_ALPHA

            dx_int = int(dx_accumulator)
            dy_int = int(dy_accumulator)

            dx_accumulator -= dx_int
            dy_accumulator -= dy_int

            distance = math.hypot(dx, dy)
            duration = map_distance_to_duration(distance)

            if dx_int != 0 or dy_int != 0:
                pyautogui.moveRel(dx_int, dy_int, duration=duration)
                print(f"Green center: {current_center} | Move by ({dx_int}, {dy_int}) | Duration: {round(duration, 3)}s")
            else:
                print(f"Green center: {current_center} | Move by (0, 0)")

        draw_tracking_info(display_frame, corners, current_center)

    else:
        if prev_visible and tap_ready:
            print("Click! Square disappeared.")
            pyautogui.click()
            tap_ready = False
        prev_visible = False

    cv2.imshow("Tracking", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()