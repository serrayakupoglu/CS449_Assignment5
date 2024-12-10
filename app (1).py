import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

elements = [
    {"type": "menu", "x": 0, "y": 0, "width": 1280, "height": 50, "label": "Menu Bar"},
    {"type": "button", "x": 50, "y": 100, "width": 200, "height": 50, "label": "Button A"},
    {"type": "button", "x": 50, "y": 200, "width": 200, "height": 50, "label": "Button B"},
    {"type": "button", "x": 50, "y": 300, "width": 200, "height": 50, "label": "Help"},
    {"type": "button", "x": 300, "y": 100, "width": 400, "height": 20, "label": "Horizontal Scroll"},
    {"type": "button", "x": 750, "y": 100, "width": 20, "height": 400, "label": "Vertical Scroll"},
]

activation_time = 2.5
hovered_element_label = None
hover_start_time = None
element_activated = False

thumbs_up_start_time = None
thumbs_up_activated = False
thumbs_up_detected = False
thumbs_up_hold_time = 0

pinch_frames = 0
pinch_frames_threshold = 5

def is_pinching(hand_landmarks):
    if not hand_landmarks:
        return False
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    return distance < 0.035

def is_thumbs_up(hand_landmarks):
    if not hand_landmarks:
        return False

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    thumb_extended = thumb_tip.y < thumb_ip.y
    index_curled = index_tip.y > index_pip.y
    middle_curled = middle_tip.y > middle_pip.y
    ring_curled = ring_tip.y > ring_pip.y
    pinky_curled = pinky_tip.y > pinky_pip.y

    return thumb_extended and index_curled and middle_curled and ring_curled and pinky_curled

def get_cursor_position(hand_landmarks, frame_width, frame_height):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    cursor_x = int(index_tip.x * frame_width)
    cursor_y = int(index_tip.y * frame_height)
    return cursor_x, cursor_y

def is_hovering(cursor_x, cursor_y, element):
    return (
        element["x"] <= cursor_x <= element["x"] + element["width"]
        and element["y"] <= cursor_y <= element["y"] + element["height"]
    )

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

help_active = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame_height, frame_width, _ = frame.shape

    hovered_now = None
    current_time = time.time()

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cursor_x, cursor_y = get_cursor_position(hand_landmarks, frame_width, frame_height)

            if is_thumbs_up(hand_landmarks):
                if thumbs_up_start_time is None:
                    thumbs_up_start_time = current_time
                    thumbs_up_hold_time = 0
                else:
                    thumbs_up_hold_time = current_time - thumbs_up_start_time
                    
                    if not thumbs_up_detected:
                        progress_text = f"Hold Thumbs Up: {thumbs_up_hold_time:.1f}s / 2.5s"
                        cv2.putText(frame, progress_text, (frame_width//2 - 150, frame_height - 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    if thumbs_up_hold_time >= activation_time and not thumbs_up_detected:
                        thumbs_up_detected = True
                        print("Thumbs Up Gesture Detected!")
            else:
                thumbs_up_start_time = None
                thumbs_up_hold_time = 0
                thumbs_up_detected = False

            if is_pinching(hand_landmarks):
                pinch_frames += 1
            else:
                pinch_frames = 0

            if pinch_frames > pinch_frames_threshold:
                print("Pinching Gesture Detected: Interaction Triggered")
                pinch_frames = 0

            for element in elements:
                if is_hovering(cursor_x, cursor_y, element):
                    hovered_now = element['label']
                    break

        if hovered_now:
            if hovered_now != hovered_element_label:
                hovered_element_label = hovered_now
                hover_start_time = current_time
                element_activated = False
            else:
                elapsed = current_time - hover_start_time
                if elapsed >= activation_time and not element_activated:
                    print(f"Activated {hovered_element_label}")
                    element_activated = True
                    if hovered_element_label == "Help":
                        help_active = not help_active
        else:
            hovered_element_label = None
            hover_start_time = None
            element_activated = False

    else:
        hovered_element_label = None
        hover_start_time = None
        element_activated = False
        pinch_frames = 0
        thumbs_up_start_time = None
        thumbs_up_hold_time = 0
        thumbs_up_detected = False

    # Draw UI elements with activation messages
    for element in elements:
        if element['type'] == 'menu':
            cv2.rectangle(frame, (element["x"], element["y"]),
                          (element["x"] + element["width"], element["y"] + element["height"]),
                          (50, 50, 50), -1)
            cv2.putText(frame, element["label"], (element["x"] + 10, element["y"] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if element['label'] == hovered_element_label and element_activated:
                cv2.putText(frame, f"{element['label']} Activated!", (frame_width//2 - 100, frame_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            continue

        color = (255, 0, 0)  # Default blue
        if element['label'] == hovered_element_label:
            if element_activated:
                color = (0, 255, 0)  # Green when activated
                cv2.putText(frame, f"{element['label']} Activated!", (frame_width//2 - 100, frame_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                color = (0, 0, 255)  # Red when hovering

        cv2.rectangle(frame, (element["x"], element["y"]),
                      (element["x"] + element["width"], element["y"] + element["height"]),
                      color, -1)
        cv2.putText(frame, element["label"], (element["x"] + 10, element["y"] + element["height"] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display thumbs up detection
    if thumbs_up_detected:
        cv2.putText(frame, "Thumbs Up Detected!", (frame_width//2 - 100, frame_height//2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Help sidebar
    if help_active:
        sidebar_x = 600
        sidebar_y = 200
        sidebar_w = 600
        sidebar_h = 300
        cv2.rectangle(frame, (sidebar_x, sidebar_y),
                      (sidebar_x + sidebar_w, sidebar_y + sidebar_h),
                      (30, 30, 30), -1)
        cv2.putText(frame, "Guidelines:", (sidebar_x + 10, sidebar_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "- Hover over any element for 2.5s to activate it", (sidebar_x + 10, sidebar_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "- Pinch gesture triggers an action after stable detection", (sidebar_x + 10, sidebar_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "- Hold Thumbs Up for 2.5s to trigger action", (sidebar_x + 10, sidebar_y + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "- All elements turn green when activated", (sidebar_x + 10, sidebar_y + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "- Press 'q' to quit", (sidebar_x + 10, sidebar_y + 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Draw cursor
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), -1)

    # Info text at bottom
    info_text = "Hover 2.5s to Activate | Pinch to Interact | Hold Thumbs Up for 2.5s"
    cv2.putText(frame, info_text, (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Gesture-Based Interaction with Menu & Guidelines", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()