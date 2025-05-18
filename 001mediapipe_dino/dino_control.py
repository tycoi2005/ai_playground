# dino_control.py

import cv2
import mediapipe as mp
import pyautogui

# === Configuration ===
INDEX_TIP_ID = 8
LEFT_SHOULDER_ID = 11
JUMP_THRESHOLD = 0.05  # Lower = more sensitive

# === Mediapipe Initializers ===
def init_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam.")
    print("✅ Webcam initialized.")
    return cap

def init_detector(mode="hand"):
    if mode == "hand":
        return init_hands()
    elif mode == "pose":
        return init_pose()
    else:
        raise ValueError("Invalid mode. Choose 'hand' or 'pose'.")

def init_hands():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

def init_pose():
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

# === Landmark Utilities ===
def get_landmark_y(landmarks, landmark_id):
    return landmarks.landmark[landmark_id].y

# === Jump Detection ===
def detect_jump(prev_y, current_y):
    if prev_y is not None and (current_y - prev_y) < -JUMP_THRESHOLD:
        print("⬆️ Jump")
        pyautogui.press("space")
    return current_y

# === Frame Preprocessing ===
def preprocess_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb

def process_frame(mode, frame, rgb_frame, detector, prev_y):
    if mode == "hand":
        return process_hand_frame(frame, rgb_frame, detector, prev_y)
    elif mode == "pose":
        return process_pose_frame(frame, rgb_frame, detector, prev_y)
    else:
        raise ValueError("Invalid mode. Choose 'hand' or 'pose'.")

# === Frame Processing: Hand Mode ===
def process_hand_frame(frame, rgb_frame, hands_detector, prev_y):
    results = hands_detector.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS
            )
            y = get_landmark_y(hand, INDEX_TIP_ID)
            prev_y = detect_jump(prev_y, y)
    return frame, prev_y

# === Frame Processing: Pose Mode ===
def process_pose_frame(frame, rgb_frame, pose_detector, prev_y):
    results = pose_detector.process(rgb_frame)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )
        y = get_landmark_y(results.pose_landmarks, LEFT_SHOULDER_ID)
        prev_y = detect_jump(prev_y, y)
    return frame, prev_y

# === Main Control Loop ===
def run_dino_controller(mode="hand"):
    print(f"🚀 Starting Dino Controller in '{mode}' mode...")
    cap = init_camera()
    prev_y = None

    with init_detector(mode) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame, rgb = preprocess_frame(frame)
            frame, prev_y = process_frame(mode, frame, rgb, detector, prev_y)
            cv2.imshow("🦖 Dino Hand Control", frame)
            if cv2.waitKey(5) & 0xFF == 27: break
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Exited Dino Controller.")
