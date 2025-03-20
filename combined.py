import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import time
from collections import deque
import pygame


pygame.mixer.init()

cymbal = pygame.mixer.Sound('cymbal.wav') 
piano = pygame.mixer.Sound('piano.wav') 
drum = pygame.mixer.Sound('drum.wav') 

sound_map = {'PIANO KEY': piano, 'CYMBAL': cymbal, 'DRUM': drum}



# Track previous shape detections for stabilization
shape_history = {}  
history_length = 10  

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # Lower frame rate
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # Lower depth frame rate

# Align depth to color stream
align = rs.align(rs.stream.color)

# Start pipeline
pipeline.start(config)
time.sleep(2)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def get_depth_at_pixel(depth_frame, x, y):
    """Retrieve the depth value (mm) at the given (x, y) pixel location."""
    depth_image = np.asanyarray(depth_frame.get_data())
    if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
        depth_value = depth_image[y, x]
        return depth_value
    return None

def preprocess_image(frame):
    """Preprocess image: convert to grayscale, apply thresholding & edge smoothing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 4)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned

def merge_contours(contours):
    """Merges small overlapping contours instead of merging everything."""
    merged_contours = []
    contour_boxes = [cv2.boundingRect(c) for c in contours]
    used = set()

    for i, rect1 in enumerate(contour_boxes):
        if i in used:
            continue

        x1, y1, w1, h1 = rect1
        merged = contours[i]

        for j, rect2 in enumerate(contour_boxes):
            if i != j and j not in used:
                x2, y2, w2, h2 = rect2
                if (x1 < x2 + w2 and x1 + w1 > x2 and
                    y1 < y2 + h2 and y1 + h1 > y2 and max(w1, h1, w2, h2) < 300):
                    merged = np.vstack((merged, contours[j]))
                    used.add(j)

        used.add(i)
        merged_contours.append(cv2.convexHull(merged))

    return merged_contours

def detect_shapes(frame):
    global shape_history

    preprocessed = preprocess_image(frame)
    
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = merge_contours(contours)

    detected_shapes = []

    frame_area = frame.shape[0] * frame.shape[1]  # Total pixels in the frame

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        num_sides = len(approx)

        # Get bounding box and area
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)

        if area < 1000 or w < 50 or h < 50:
            continue

        if area > frame_area * 0.6:
            continue

        # Shape classification
        if num_sides == 3:
            shape = "CYMBAL"
            color = (0, 255, 0)  # Green
        elif num_sides == 4:
            shape = "PIANO KEY"
            color = (255, 0, 0)  # Blue
        elif num_sides > 6:
            shape = "DRUM"
            color = (0, 0, 255)  # Red
        else:
            continue

        # Stabilize detection with shape history
        shape_id = f"{x}_{y}"
        if shape_id not in shape_history:
            shape_history[shape_id] = deque(maxlen=history_length)
        shape_history[shape_id].append(shape)
        most_common_shape = max(set(shape_history[shape_id]), key=shape_history[shape_id].count)

        cv2.drawContours(frame, [approx], -1, color, 3)
        cv2.putText(frame, most_common_shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        detected_shapes.append((most_common_shape, (x, y, w, h)))  # Return shape and bounding box

    return frame, detected_shapes

def restart_pipeline():
    global pipeline
    print("Restarting RealSense pipeline...")
    pipeline.stop()
    time.sleep(2)
    pipeline.start(config)

while True:
    # Capture frames
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    if not frames:
        print("ERROR: No frames received. Restarting pipeline...")
        restart_pipeline()
        continue

    # Align depth and color frames
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("No valid frame received. Retrying...")
        continue

    # Convert to NumPy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Convert color image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Process hand tracking
    result = hands.process(rgb_image)
    finger_pos = None
    finger_depth = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get index finger tip coordinates (landmark 8)
            h, w, _ = color_image.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Get depth at the finger's pixel location
            finger_depth = get_depth_at_pixel(depth_frame, x, y)
            finger_pos = (x, y)

            # Draw the hand landmarks
            mp_draw.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw the detected finger tip
            cv2.circle(color_image, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(color_image, f"Depth: {finger_depth}mm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Detect and draw shapes
    processed_frame, detected_shapes = detect_shapes(color_image)

    # Draw detected shapes and check for interactions
    for shape, (sx, sy, sw, sh) in detected_shapes:
        color = (255, 0, 0)  # Default Blue
        # If the finger is detected and touching the shape, highlight in green
        if finger_pos and (sx < finger_pos[0] < sx + sw and sy < finger_pos[1] < sy + sh):
            color = (0, 255, 0)  # Change to Green
            print(f"Finger touched {shape}!")

            sound_map[shape].play()

        # Instead of drawing a rectangle, just label the shape
        cv2.putText(processed_frame, shape, (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the output
    cv2.imshow("Hand & Shape Detection", processed_frame)

    # Exit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Exiting program.")
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
