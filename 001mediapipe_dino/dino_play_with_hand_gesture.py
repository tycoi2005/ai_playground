# This code is a simple Dino game controller using hand gestures.
# It uses OpenCV for video capture and Mediapipe for hand tracking.
# The user can jump in the game by raising their index finger.

import cv2
import mediapipe as mp
import pyautogui

# Configurations
# Index finger tip landmark ID
INDEX_TIP_ID = 8

# Threshold for jump detection
# This threshold determines how much the index finger must move up to trigger a jump.
# A smaller value means a more sensitive jump detection.
# A larger value means a less sensitive jump detection.
JUMP_THRESHOLD = 0.05

# Initialization opencv2 camera
def init_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam.")
    print("✅ Webcam initialized.")
    return cap

# Initialize Mediapipe Hands
def init_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

# Get the y-coordinate of the index finger tip
def get_index_finger_y(hand_landmarks):
    return hand_landmarks.landmark[INDEX_TIP_ID].y

# Detect jump based on the y-coordinate of the index finger tip
# If the finger moves up by more than JUMP_THRESHOLD, trigger a jump
# Update the previous y-coordinate for the next frame.
def detect_jump(prev_y, current_y):
    if prev_y is not None:
        delta = current_y - prev_y
        if delta < -JUMP_THRESHOLD:
            print("⬆️ Jump")
            pyautogui.press("space")
    return current_y

# Handles the webcam frame processing: flips the frame, converts it to RGB, and processes it with Mediapipe.
# Detects hand landmarks and checks for jump gestures.
# The function returns the processed frame and the updated previous y-coordinate.
def process_frame(frame, hands_detector, prev_y):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )
            current_y = get_index_finger_y(hand_landmarks)
            prev_y = detect_jump(prev_y, current_y)

    return frame, prev_y


# Run the Dino Jump Controller
def run_dino_control():
    print("🚀 Starting Dino Jump Controller...")
    cap = init_camera()

    with init_hands() as hands_detector:
        prev_y = None

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame, prev_y = process_frame(frame, hands_detector, prev_y)
            cv2.imshow("🦖 Dino Control", frame)

            if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Exited Dino Controller.")

# Main function to run the Dino Jump Controller
if __name__ == "__main__":
    run_dino_control()