import cv2
import numpy as np
from collections import deque

# Track previous shape detections for stabilization
shape_history = {}  
history_length = 10  

def preprocess_image(frame):
    """Preprocess image: convert to grayscale, apply thresholding & edge smoothing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to reduce noise
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 4)

    # Morphological closing to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return cleaned

def merge_contours(contours):
    """Merges overlapping contours to prevent duplicate detections."""
    merged_contours = []
    contour_boxes = [cv2.boundingRect(c) for c in contours]
    used = set()  # Track merged contours

    for i, rect1 in enumerate(contour_boxes):
        if i in used:  # Skip if already merged
            continue

        x1, y1, w1, h1 = rect1
        merged = contours[i]

        for j, rect2 in enumerate(contour_boxes):
            if i != j and j not in used:
                x2, y2, w2, h2 = rect2

                # Check if bounding boxes overlap significantly
                if (x1 < x2 + w2 and x1 + w1 > x2 and
                    y1 < y2 + h2 and y1 + h1 > y2):
                    merged = np.vstack((merged, contours[j]))  # Merge contours
                    used.add(j)

        used.add(i)
        merged_contours.append(cv2.convexHull(merged))  # Store final merged contour

    return merged_contours


def detect_shapes(frame):
    global shape_history

    preprocessed = preprocess_image(frame)
    
    # Find contours
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Merge overlapping contours
    contours = merge_contours(contours)

    detected_shapes = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        num_sides = len(approx)

        # Get bounding box and area
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)

        # Ignore small noisy objects
        if area < 1000 or w < 50 or h < 50:
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
        detected_shapes.append(most_common_shape)

    return frame

# Start camera capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open RealSense camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    processed_frame = detect_shapes(frame)
    cv2.imshow('Improved Shape Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
