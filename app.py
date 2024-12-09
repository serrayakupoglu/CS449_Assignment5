import cv2
import mediapipe as mp
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Enhanced UI Elements with more design and functionality
elements = [
    {"type": "menu", "x": 0, "y": 0, "width": 1280, "height": 50, "label": "Gesture Control Interface", "color": (30, 40, 60)},
    {"type": "button", "x": 50, "y": 100, "width": 250, "height": 70, "label": "Start Session", "color": (70, 100, 150)},
    {"type": "button", "x": 50, "y": 200, "width": 250, "height": 70, "label": "Settings", "color": (70, 100, 150)},
    {"type": "button", "x": 50, "y": 300, "width": 250, "height": 70, "label": "Help", "color": (70, 100, 150)},
    {"type": "scrollbar", "x": 350, "y": 100, "width": 20, "height": 400, "label": "Vertical Scroll", "color": (50, 70, 100)},
    {"type": "scrollbar", "x": 400, "y": 500, "width": 400, "height": 20, "label": "Horizontal Scroll", "color": (50, 70, 100)},
]

# Configuration variables
activation_time = 2.5
hover_sensitivity = 0.035
interaction_timeout = 3.0

# State tracking variables
hovered_element_label = None
hover_start_time = None
element_activated = False
help_active = False
current_mode = "Default"

def is_pinching(hand_landmarks):
    if not hand_landmarks:
        return False
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    return distance < hover_sensitivity

def is_thumbs_up(hand_landmarks):
    if not hand_landmarks:
        return False

    # More robust thumbs up detection
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    
    def is_finger_curled(tip, pip):
        return tip.y > pip.y

    return (
        thumb_tip.y < thumb_ip.y and
        all(is_finger_curled(hand_landmarks.landmark[tip], hand_landmarks.landmark[pip]) 
            for tip, pip in [
                (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
            ])
    )

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

def draw_gradient_background(frame):
    height, width = frame.shape[:2]
    for y in range(height):
        # Create a gradient from dark blue to lighter blue
        r = int(20 + (y / height) * 50)
        g = int(30 + (y / height) * 70)
        b = int(50 + (y / height) * 100)
        frame[y, :] = [b, g, r]
    return frame

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    global current_mode, help_active, hovered_element_label, hover_start_time, element_activated

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply gradient background
        frame = draw_gradient_background(frame)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_height, frame_width, _ = frame.shape

        current_time = time.time()
        hovered_now = None
        cursor_x, cursor_y = None, None

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cursor_x, cursor_y = get_cursor_position(hand_landmarks, frame_width, frame_height)

                # Interaction detection
                for element in elements:
                    if is_hovering(cursor_x, cursor_y, element):
                        hovered_now = element['label']
                        break

                # Gestures handling
                if is_thumbs_up(hand_landmarks):
                    current_mode = "Thumbs Up Mode"
                
                if is_pinching(hand_landmarks):
                    current_mode = "Pinch Mode"

        # Hover and activation logic
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

        # Draw UI elements with enhanced design
        for element in elements:
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (element["x"], element["y"]),
                          (element["x"] + element["width"], element["y"] + element["height"]),
                          element.get('color', (50, 50, 50)), -1)
            
            # Blending for depth
            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Text rendering
            color = (255, 255, 255)
            if element['label'] == hovered_element_label:
                color = (0, 255, 255)  # Highlight when hovered
                if element_activated:
                    color = (0, 255, 0)  # Green when activated

            cv2.putText(frame, element["label"], 
                        (element["x"] + 10, element["y"] + element["height"] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Help sidebar with better design
        if help_active:
            sidebar_x, sidebar_y = 600, 100
            sidebar_w, sidebar_h = 600, 400
            
            # Semi-transparent dark overlay
            help_overlay = frame.copy()
            cv2.rectangle(help_overlay, (sidebar_x, sidebar_y),
                          (sidebar_x + sidebar_w, sidebar_y + sidebar_h),
                          (20, 20, 40), -1)
            frame = cv2.addWeighted(help_overlay, 0.7, frame, 0.3, 0)

            # Guidelines text
            guidelines = [
                "Interaction Guidelines:",
                "• Hover over elements for 2.5s to activate",
                "• Pinch gesture triggers contextual actions",
                "• Thumbs Up changes interaction mode",
                "• Green indicates element activation",
                "• Press 'q' to quit application"
            ]

            for i, line in enumerate(guidelines):
                cv2.putText(frame, line, 
                            (sidebar_x + 20, sidebar_y + 50 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Cursor visualization
        if cursor_x is not None and cursor_y is not None:
            cv2.circle(frame, (cursor_x, cursor_y), 15, (0, 255, 255), -1)

        # Mode and info display
        info_text = f"Mode: {current_mode} | Hover 2.5s to Activate"
        cv2.putText(frame, info_text, (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Gesture Control Interface", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()