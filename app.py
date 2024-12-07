import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define interactive elements (buttons and scrollbars)
elements = [
    {"type": "button", "x": 100, "y": 100, "width": 200, "height": 50, "label": "Button 1"},
    {"type": "button", "x": 400, "y": 100, "width": 200, "height": 50, "label": "Button 2"},
    {"type": "scrollbar", "x": 100, "y": 400, "width": 400, "height": 20, "label": "Horizontal Scroll"},
    {"type": "scrollbar", "x": 600, "y": 100, "width": 20, "height": 400, "label": "Vertical Scroll"},
]

# Gesture detection: Pinching
def is_pinching(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    return distance < 0.05  # Adjust threshold as necessary

# Gesture detection: Thumbs up
def is_thumbs_up(hand_landmarks):
    # Get landmarks for thumb
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    
    # Get landmarks for other fingers (tip and middle joints)
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    # Check if thumb is extended upward
    thumb_extended = thumb_tip.y < thumb_ip.y

    # Check if other fingers are curled (tip should be below PIP joint)
    index_curled = index_tip.y > index_pip.y
    middle_curled = middle_tip.y > middle_pip.y
    ring_curled = ring_tip.y > ring_pip.y
    pinky_curled = pinky_tip.y > pinky_pip.y

    # All conditions must be met for a proper thumbs up
    return (thumb_extended and 
            index_curled and 
            middle_curled and 
            ring_curled and 
            pinky_curled)

# Cursor position calculation
def get_cursor_position(hand_landmarks, frame_width, frame_height):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    cursor_x = int(index_tip.x * frame_width)
    cursor_y = int(index_tip.y * frame_height)
    return cursor_x, cursor_y

# Check if cursor is hovering over an element
def is_hovering(cursor_x, cursor_y, element):
    return (
        element["x"] <= cursor_x <= element["x"] + element["width"]
        and element["y"] <= cursor_y <= element["y"] + element["height"]
    )

# Open webcam feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame_height, frame_width, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Cursor position
            cursor_x, cursor_y = get_cursor_position(hand_landmarks, frame_width, frame_height)

            # Gesture detection
            if is_pinching(hand_landmarks):
                print("Pinching Gesture Detected: Interaction Triggered")
            elif is_thumbs_up(hand_landmarks):
                print("Thumbs Up Gesture Detected")

            # Hover detection
            for element in elements:
                color = (255, 0, 0)  # Default color
                if is_hovering(cursor_x, cursor_y, element):
                    color = (0, 255, 0)  # Hover color
                    print(f"Hovering over: {element['label']}")

                # Draw elements (buttons or scrollbars)
                cv2.rectangle(frame, (element["x"], element["y"]),
                              (element["x"] + element["width"], element["y"] + element["height"]),
                              color, -1)
                cv2.putText(frame, element["label"], (element["x"] + 10, element["y"] + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw cursor
            cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), -1)

    cv2.imshow("Gesture-Based Interaction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
