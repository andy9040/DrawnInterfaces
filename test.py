import cv2

for i in range(3):  # Loop through available cameras
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Opening Camera {i}...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {i} failed to capture.")
                break
            
            cv2.putText(frame, f"Camera {i}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f"Camera {i}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to close the camera window
                break

        cap.release()
        cv2.destroyAllWindows()
