import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import time
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load sounds
cymbal = pygame.mixer.Sound('cymbal.wav') 
piano = pygame.mixer.Sound('piano.wav') 
drum = pygame.mixer.Sound('drum.wav') 

sound_map = {'PIANO KEY': piano, 'CYMBAL': cymbal, 'DRUM': drum}

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
            shape = "CYMBAL"
            color = (0, 255, 0)
        elif num_sides == 4:
            shape = "PIANO KEY"
            color = (255, 0, 0)
        elif num_sides > 6:
            shape = "DRUM"
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

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        continue
    
    color_image = np.asanyarray(color_frame.get_data())
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    if mode == "shape_detection":
        processed_frame, detected_shapes = detect_shapes(color_image)
        static_shapes = detected_shapes  # Store shapes
    else:
        processed_frame = color_image.copy()
        for shape, contour, color in static_shapes:
            cv2.drawContours(processed_frame, [contour], -1, color, 3)
    
    result = hands.process(rgb_image)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = color_image.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            
            for shape, contour, color in static_shapes:
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    print(f"Finger touched {shape}!")
                    sound_map[shape].play()
                    break
            
            mp_draw.draw_landmarks(processed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Shape & Hand Detection", processed_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        mode = "hand_tracking"
    elif key == ord("q"):
        break

pipeline.stop()
cv2.destroyAllWindows()
