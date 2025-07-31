import cv2
from deepface import DeepFace

# --- Main Program ---

# Use OpenCV to capture video from the webcam.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Starting emotion detection. Press 'q' or 'Esc' to exit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # We'll work with a copy of the frame
    result_frame = frame.copy()

    try:
        # Use DeepFace to analyze the frame for emotions.
        # We specify 'opencv' as the detector backend, which is very reliable.
        # 'enforce_detection=False' allows the program to continue if no face is found.
        analysis_results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv' # Switched to opencv for reliability
        )

        # DeepFace.analyze returns a list of dictionaries, one for each detected face.
        for result in analysis_results:
            # Get the bounding box of the face
            x = result['region']['x']
            y = result['region']['y']
            w = result['region']['w']
            h = result['region']['h']

            # Get the dominant emotion
            dominant_emotion = result['dominant_emotion']

            # Define text and box color (green)
            text_color = (0, 255, 0)
            box_color = (0, 255, 0)

            # Draw a rectangle around the detected face
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), box_color, 2)

            # Prepare the text to display (emotion)
            text = f"Emotion: {dominant_emotion.capitalize()}"

            # Put the text on the frame
            # We place it slightly above the bounding box
            text_position = (x, y - 10)
            cv2.putText(result_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    except Exception as e:
        # This can happen if no face is detected with enforce_detection=True
        # With it set to False, we'll just see other potential errors.
        pass # We can pass silently to keep the video stream smooth

    # Display the resulting frame
    cv2.imshow('Facial Emotion Recognition', result_frame)

    # --- THIS IS THE MODIFIED SECTION ---
    # Check for user input to exit
    key_pressed = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' or the 'Esc' key is pressed
    if key_pressed == ord('q') or key_pressed == 27: # 27 is the ASCII code for the Escape key
        print("Exit key pressed. Closing program.")
        break
    # ------------------------------------

# --- Cleanup ---
# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()