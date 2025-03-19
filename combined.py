import cv2
import numpy as np
from collections import deque
import pyrealsense2 as rs


# Track previous shape detections for stabilization
shape_history = {}  
history_length = 10  

def preprocess_image(frame):
    """Preprocess image: convert to grayscale, apply thresholding & edge smoothing."""
    cv2.imshow('Round 0', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('Round 1', blurred)

    # Adaptive thresholding to reduce noise
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 4)
    cv2.imshow('Round 2', thresh)

    # Morphological closing to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow('Round 3', cleaned)


    return cleaned

# def preprocess_image(frame):
#     """Preprocess the image to remove noise and enhance shape detection."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur to smooth out noise before thresholding
#     blurred = cv2.GaussianBlur(gray, (7, 7), 0)

#     # Use Otsu's thresholding instead of adaptive thresholding
#     _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     cv2.imshow('Round 1', thresh)

#     # Morphological operations to remove small noise
#     kernel = np.ones((3, 3), np.uint8)
#     thresh = cv2.erode(thresh, kernel, iterations=2)  # Remove tiny dots

#     cv2.imshow('Round 2', thresh)

#     thresh = cv2.dilate(thresh, kernel, iterations=2)  # Restore main shapes

#     cv2.imshow('Round 3', thresh)


#     return thresh



def merge_contours_old(contours):
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


def merge_contours(contours):
    """Merges only small overlapping contours instead of merging everything."""
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

                # Only merge if both contours are relatively small
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
    
    # Find contours
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Merge overlapping contours
    contours = merge_contours(contours)

    detected_shapes = []

    frame_area = frame.shape[0] * frame.shape[1]  # Total pixels in the frame


    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        num_sides = len(approx)

        # Get bounding box and area
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(contour)

        # Ignore small noisy objects
        if area < 1000 or w < 50 or h < 50:
            continue

        if area > frame_area * 0.6:  # Ignore if it's covering more than 60% of the frame
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

    cv2.imshow('Before', frame)

    processed_frame = detect_shapes(frame)
    cv2.imshow('Improved Shape Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("QUIT")
        break
    elif cv2.waitKey(1) & 0xFF == ord('d'):
        print("DEPTH MODE ACTIVATED")
        pipeline = rs.pipeline()
        config = rs.config()

        # Enable both color and depth streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Align depth map to color stream
        align = rs.align(rs.stream.color)

        # Start pipeline
        pipeline.start(config)

        def get_closest_point(depth_frame):
            """Find the closest point in the depth frame (potentially a finger)."""
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Ignore 0-depth areas (invalid readings)
            depth_image[depth_image == 0] = np.max(depth_image)

            # Get the minimum depth point (closest object)
            min_val = np.min(depth_image)
            min_loc = np.unravel_index(np.argmin(depth_image), depth_image.shape)

            return min_loc[::-1], min_val  # Return (X, Y), Depth

        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            # Get depth and color frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Get closest hand point (finger tip)
            finger_pos, finger_depth = get_closest_point(depth_frame)
            
            # Draw finger point on color image
            cv2.circle(color_image, finger_pos, 10, (0, 0, 255), -1)
            cv2.putText(color_image, f"Depth: {finger_depth}mm", (finger_pos[0], finger_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Check if finger is touching a detected shape
            for shape, (x, y, w, h) in detected_shapes.items():
                if x < finger_pos[0] < x + w and y < finger_pos[1] < y + h:
                    print(f"Finger touched {shape}!")
                    # sounds[shape].play()  # Play corresponding sound
            
            # Display depth and color frames
            cv2.imshow("Depth View", depth_image)
            cv2.imshow("Shape Interaction", color_image)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()
