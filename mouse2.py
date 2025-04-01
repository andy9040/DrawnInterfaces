import cv2
import numpy as np
import pyautogui

# --- Mouse control state ---
last_mouse_pos = None
mouse_down = False

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

def get_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 4)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

def detect_triangles(frame):
    preprocessed = preprocess_image(frame)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    triangles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 3 and is_equilateral_triangle(approx):
            area = cv2.contourArea(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if area > 500 and w > 30 and h > 30:
                triangles.append((approx, x, y))
    
    # Sort by x (left to right)
    triangles.sort(key=lambda t: t[1])
    return triangles[:2]

def control_mouse(triangles):
    global last_mouse_pos, mouse_down

    if len(triangles) == 2:
        center = get_center(triangles[0][0])
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
        center = get_center(triangles[0][0])
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
        cv2.circle(frame, get_center(tri), 5, (0, 255, 0), -1)

# OpenCV camera capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    triangles = detect_triangles(frame)
    control_mouse(triangles)
    draw_triangles(frame, triangles)

    cv2.imshow("Triangle Mouse Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
