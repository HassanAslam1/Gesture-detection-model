import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Menu for selecting functionality
print("Select Mode:")
print("1: Gesture Detection")
print("2: Keyboard and Mouse Control")
mode = input("Enter 1 or 2: ")

# Initialize variables
cap = cv2.VideoCapture(0)
prev_x = None
prev_y = None
gesture_name = None  # Variable to store detected gesture

if mode == "1":
    print("Gesture Detection Mode Activated.")
elif mode == "2":
    print("Keyboard and Mouse Control Mode Activated.")
else:
    print("Invalid input. Exiting...")
    cap.release()
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Mirror the image
    frame = cv2.flip(frame, 1)

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Reset gesture name
    gesture_name = None

    if results.multi_hand_landmarks:
        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            # Detect handedness safely
            try:
                handedness = results.multi_handedness[hand_idx].classification[0].label
            except (IndexError, AttributeError):
                handedness = "Unknown"

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Safely access landmarks
            try:
                thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mid = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_mid = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            except (IndexError, AttributeError):
                continue  # Skip this hand if landmarks are missing

            if mode == "1":  # Gesture Detection Only
                # Thumbs Up: Thumb tip above IP joint, index finger below IP joint
                if thumb_tip.y < thumb_ip.y and index_tip.y > thumb_ip.y:
                    gesture_name = "Thumbs Up"

                # Fist: All fingers curled (thumb across)
                elif thumb_tip.x < index_tip.x and thumb_tip.x < middle_tip.x and thumb_tip.x < ring_tip.x and thumb_tip.x < pinky_tip.x:
                    gesture_name = "Fist"

                # Open Hand: All fingers extended outward
                elif index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y and ring_tip.y < pinky_tip.y:
                    # Check if the fingers are spread out (index, middle, ring, pinky are above their respective PIPs)
                    if (index_tip.y < index_mid.y and middle_tip.y < middle_mid.y and ring_tip.y < pinky_tip.y):
                        gesture_name = "Open Hand"

            elif mode == "2":  # Keyboard and Mouse Control Only
                if handedness == "Left":
                    mcp_x = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
                    mcp_y = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y

                    cursor_x = int(mcp_x * screen_width)
                    cursor_y = int(mcp_y * screen_height)

                    pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

                    # Check for click gesture
                    if index_tip.y >= index_mid.y:
                        pyautogui.click()

                elif handedness == "Right":
                    x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)

                    if prev_x is not None and prev_y is not None:
                        dx = x - prev_x
                        dy = y - prev_y

                        if abs(dx) > abs(dy):  # Horizontal swipe
                            if dx > 50:
                                pyautogui.press('right')
                            elif dx < -50:
                                pyautogui.press('left')
                        else:  # Vertical swipe
                            if dy > 50:
                                pyautogui.press('down')
                            elif dy < -50:
                                pyautogui.press('up')

                    prev_x = x
                    prev_y = y

    # Display gesture name in Gesture Detection mode
    if mode == "1" and gesture_name:
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Gesture Recognition", frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
