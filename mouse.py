import cv2
import numpy as np
from collections import deque
import pyautogui
import math

# --- Globals for calibration ---
adaptive_thresh_C = 4
blur_ksize = 5
use_adaptive = True

# --- Shape history and state ---
shape_history = {}
history_length = 10
last_mouse_pos = None
mouse_down = False

# --- Triangle tracking ---
tracked_triangles = []

def is_equilateral_triangle(pts, tolerance=0.4):
    if len(pts) != 3:
        return False
    sides = []
    for i in range(3):
        pt1 = pts[i][0]
        pt2 = pts[(i+1)%3][0]
        sides.append(np.linalg.norm(pt1 - pt2))
    avg = sum(sides) / 3
    return all(abs(s - avg)/avg < tolerance for s in sides)


def preprocess_image(frame):
    global adaptive_thresh_C, blur_ksize, use_adaptive

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_size = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    if use_adaptive:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, adaptive_thresh_C)
    else:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imshow("Calibrated View", cleaned)
    return cleaned

def detect_triangles(frame):
    global shape_history

    processed = preprocess_image(frame)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    triangles = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.08 * cv2.arcLength(contour, True), True)


        if len(approx) == 3 and is_equilateral_triangle(approx):
            x, y, w, h = cv2.boundingRect(approx)
            area = cv2.contourArea(approx)

            if area > 500 and w > 20 and h > 20:
                triangles.append((approx, x, y))

    # Sort left to right by x-coordinate
    triangles.sort(key=lambda t: t[1])

    return triangles[:2]  # Keep only first 2

def get_triangle_center(triangle):
    M = cv2.moments(triangle)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def control_mouse(triangles):
    global last_mouse_pos, mouse_down

    if len(triangles) == 2:
        center = get_triangle_center(triangles[0][0])  # Leftmost triangle
        if center:
            if last_mouse_pos:
                dx = center[0] - last_mouse_pos[0]
                dy = center[1] - last_mouse_pos[1]
                pyautogui.moveRel(dx, dy)
            last_mouse_pos = center
            if mouse_down:
                pyautogui.mouseUp()
                mouse_down = False

    elif len(triangles) == 1:
        center = get_triangle_center(triangles[0][0])
        if center:
            if last_mouse_pos:
                dx = center[0] - last_mouse_pos[0]
                dy = center[1] - last_mouse_pos[1]
                pyautogui.moveRel(dx, dy)
            last_mouse_pos = center
            if not mouse_down:
                pyautogui.mouseDown()
                mouse_down = True

    else:
        last_mouse_pos = None
        if mouse_down:
            pyautogui.mouseUp()
            mouse_down = False

def draw_triangles(frame, triangles):
    for tri, x, y in triangles:
        cv2.drawContours(frame, [tri], -1, (0, 255, 255), 3)
        cv2.putText(frame, "TRIANGLE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

def handle_key_press(key):
    global adaptive_thresh_C, blur_ksize, use_adaptive

    if key == ord('1'):
        adaptive_thresh_C = max(adaptive_thresh_C - 1, 0)
    elif key == ord('2'):
        adaptive_thresh_C += 1
    elif key == ord('3'):
        blur_ksize = max(3, blur_ksize - 2)
    elif key == ord('4'):
        blur_ksize += 2
    elif key == ord('5'):
        use_adaptive = not use_adaptive

    print(f"Threshold C: {adaptive_thresh_C}, Blur: {blur_ksize}, Adaptive: {use_adaptive}")

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not read")
            break

        triangles = detect_triangles(frame)
        draw_triangles(frame, triangles)
        control_mouse(triangles)

        cv2.imshow("Final Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        handle_key_press(key)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
