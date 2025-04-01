import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import time
import pyautogui 
import subprocess

def toggle_mute():
    # AppleScript to check if the volume is muted and toggle it
    apple_script = '''
    set currentVolume to output muted of (get volume settings)
    if currentVolume is true then
        set volume output muted false
    else
        set volume output muted true
    end if
    '''
    subprocess.run(['osascript', '-e', apple_script])

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
align = rs.align(rs.stream.color)
pipeline.start(config)
time.sleep(2)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Mode control
mode = "shape_detection"  # Start with shape detection
static_shapes = []  # Store detected shapes
hand_tracking_enabled = False  # Ensure hand tracking only starts after pressing 's'

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 4)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

def detect_shapes(frame):
    preprocessed = preprocess_image(frame)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_shapes = []
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        num_sides = len(approx)
        area = cv2.contourArea(contour)
        
        if area < 1000:
            continue
        
        shape = "UNKNOWN"
        color = (255, 255, 255)
        
        if num_sides == 3:
            shape = "TRIANGLE"
            color = (0, 255, 0)
        elif num_sides == 4:
            shape = "SQUARE"
            color = (255, 0, 0)
        elif num_sides > 6:
            shape = "CIRCLE"
            color = (0, 0, 255)
        else:
            continue
        
        detected_shapes.append((shape, contour, color))
        cv2.drawContours(frame, [contour], -1, color, 3)
        
    return frame, detected_shapes

def get_depth_at_pixel(depth_frame, x, y):
    depth_image = np.asanyarray(depth_frame.get_data())
    if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
        return depth_image[y, x]
    return None

def handle_shape_detection(shape):
    """Simulate key presses based on the shape detected."""
    if shape == "SQUARE":
        pyautogui.press('space')  # pause/play
        time.sleep(0.2)
        print("Pressed space for square")
    elif shape == "TRIANGLE":
        pyautogui.hotkey('command', 'right')
        time.sleep(0.3)
        print("Pressed cmd + right for triangle")
    elif shape == "CIRCLE":
        toggle_mute()
        time.sleep(0.3)
        print("Pressed mute/unmute for circle")

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        continue
    
    color_image = np.asanyarray(color_frame.get_data())
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    if mode == "shape_detection" and not hand_tracking_enabled:
        # Only detect shapes if hand tracking is not enabled
        processed_frame, detected_shapes = detect_shapes(color_image)
        static_shapes = detected_shapes  # Store shapes for future frames
    else:
        # Once hand tracking is enabled, do not detect shapes again
        processed_frame = color_image.copy()
    
    # Draw the stored shapes even after hand tracking is enabled
    for shape, contour, color in static_shapes:
        cv2.drawContours(processed_frame, [contour], -1, color, 3)
    
    if hand_tracking_enabled:
        result = hands.process(rgb_image)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, _ = color_image.shape
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                
                for shape, contour, color in static_shapes:
                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                        print(f"Finger touched {shape}!")
                        handle_shape_detection(shape)  # Simulate key press
                        break
                
                mp_draw.draw_landmarks(processed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Shape & Hand Detection", processed_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        hand_tracking_enabled = True  # Start hand tracking only after 's' is pressed
        mode = "hand_tracking"  # Switch to hand tracking mode
        print("Hand tracking enabled.")
    elif key == ord("q"):
        break

pipeline.stop()
cv2.destroyAllWindows()
