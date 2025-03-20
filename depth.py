import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import simpleaudio as sa  # For sound playback
import time

# Load sounds for shape interactions
# sounds = {
#     "CYMBAL": sa.WaveObject.from_wave_file("cymbal.wav"),
#     "PIANO KEY": sa.WaveObject.from_wave_file("piano.wav"),
#     "DRUM": sa.WaveObject.from_wave_file("drum.wav"),
# }

# Simulated detected shapes (x, y, width, height)
detected_shapes = {
    "CYMBAL": (100, 150, 80, 80),
    "PIANO KEY": (300, 200, 100, 50),
    "DRUM": (500, 100, 120, 120),
}

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

# Function to restart the pipeline
def restart_pipeline():
    global pipeline
    print("Restarting RealSense pipeline...")
    pipeline.stop()
    time.sleep(2)
    pipeline.start(config)

def get_depth_at_pixel(depth_frame, x, y):
    """Retrieve the depth value (mm) at the given (x, y) pixel location."""
    depth_image = np.asanyarray(depth_frame.get_data())
    if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
        depth_value = depth_image[y, x]
        return depth_value
    return None

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

    # Draw detected shapes and check for interactions
    for shape, (sx, sy, sw, sh) in detected_shapes.items():
        color = (255, 0, 0)  # Default Blue

        # If the finger is detected and touching the shape, highlight in green
        if finger_pos and (sx < finger_pos[0] < sx + sw and sy < finger_pos[1] < sy + sh):
            color = (0, 255, 0)  # Change to Green
            print(f"Finger touched {shape}!")
            # sounds[shape].play()  # Play sound

        # Draw shape
        cv2.rectangle(color_image, (sx, sy), (sx + sw, sy + sh), color, 3)
        cv2.putText(color_image, shape, (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the output
    cv2.imshow("Hand & Depth Tracking", color_image)

    # Exit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Exiting program.")
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
