import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Quiz questions
questions = [
    {
        "question": "Is Mount Everest the highest mountain in the world?",
        "correctAnswer": True,
        "explanation": "Yes, Mount Everest stands at 8,848 meters above sea level."
    },
    {
        "question": "Is the Sun a planet?",
        "correctAnswer": False,
        "explanation": "No, the Sun is a star, not a planet."
    },
    {
        "question": "Does water boil at 100 degrees Celsius at sea level?",
        "correctAnswer": True,
        "explanation": "Yes, water boils at 100°C (212°F) at sea level."
    },
    {
        "question": "Is the Atlantic Ocean the largest ocean on Earth?",
        "correctAnswer": False,
        "explanation": "No, the Pacific Ocean is the largest ocean."
    },
    {
        "question": "Is the speed of light faster than the speed of sound?",
        "correctAnswer": True,
        "explanation": "Yes, light travels much faster than sound."
    }
]

# Game state variables
current_question = 0
score = 0
game_active = True
answer_shown = False
answer_time = None
ANSWER_DISPLAY_DURATION = 3.0
showing_final_screen = False
final_screen_start_time = None
CELEBRATION_ANIMATION_DURATION = 5.0
help_hover_start_time = None
HELP_ACTIVATION_TIME = 2.5

# Help system variables
help_active = False
help_start_time = None
HELP_DISPLAY_DURATION = 15.0
help_button = {
    "x": 50,
    "y": 650,
    "width": 150,
    "height": 50
}
cursor_x, cursor_y = 0, 0
last_click_time = 0

# Scroll variables
prev_y_pos = None
prev_x_pos = None
scroll_offset = 0
MAX_SCROLL = 200
instruction_height = 600
visible_height = 400
can_scroll_next = False

def is_thumbs_up(hand_landmarks):
    if not hand_landmarks:
        return False

    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]

    index_pip = hand_landmarks.landmark[6]
    middle_pip = hand_landmarks.landmark[10]
    ring_pip = hand_landmarks.landmark[14]
    pinky_pip = hand_landmarks.landmark[18]
    
    thumb_up = (
        thumb_tip.y < thumb_ip.y and
        thumb_ip.y < thumb_mcp.y and
        abs(thumb_tip.x - thumb_ip.x) < 0.1
    )
    
    other_fingers_curled = all([
        hand_landmarks.landmark[8].y > index_pip.y,
        hand_landmarks.landmark[12].y > middle_pip.y,
        hand_landmarks.landmark[16].y > ring_pip.y,
        hand_landmarks.landmark[20].y > pinky_pip.y
    ])

    return thumb_up and other_fingers_curled

def is_thumbs_down(hand_landmarks):
    if not hand_landmarks:
        return False

    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]

    index_dip = hand_landmarks.landmark[7]
    middle_dip = hand_landmarks.landmark[11]
    ring_dip = hand_landmarks.landmark[15]
    pinky_dip = hand_landmarks.landmark[19]

    index_mcp = hand_landmarks.landmark[5]
    middle_mcp = hand_landmarks.landmark[9]
    ring_mcp = hand_landmarks.landmark[13]
    pinky_mcp = hand_landmarks.landmark[17]

    thumb_pointing_down = (
        thumb_tip.y > thumb_ip.y > thumb_mcp.y and
        abs(thumb_tip.x - thumb_mcp.x) < 0.1
    )

    other_fingers_curled = all([
        index_dip.y > index_mcp.y,
        middle_dip.y > middle_mcp.y,
        ring_dip.y > ring_mcp.y,
        pinky_dip.y > pinky_mcp.y,
        abs(index_dip.x - middle_dip.x) < 0.05,
        abs(middle_dip.x - ring_dip.x) < 0.05,
        abs(ring_dip.x - pinky_dip.x) < 0.05
    ])

    return thumb_pointing_down and other_fingers_curled

def is_vertical_scrolling(hand_landmarks, prev_y):
    if not hand_landmarks:
        return False, 0
    
    index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
    current_y = index_tip.y
    
    if prev_y is not None:
        # Yukarıdan aşağıya hareket
        if current_y - prev_y > 0.01:  # Threshold for scroll detection
            return True, current_y
    
    return False, current_y

def is_horizontal_scrolling(hand_landmarks, prev_x):
    if not hand_landmarks:
        return False, 0
    
    index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
    current_x = index_tip.x
    
    if prev_x is not None:
        # Soldan sağa hareket
        if current_x - prev_x > 0.05:  # Threshold for scroll detection
            return True, current_x
    
    return False, current_x

def get_cursor_position(hand_landmarks, frame_width, frame_height):
    index_tip = hand_landmarks.landmark[8]
    cursor_x = int(index_tip.x * frame_width)
    cursor_y = int(index_tip.y * frame_height)
    return cursor_x, cursor_y

def is_cursor_on_help(cx, cy):
    return (help_button["x"] <= cx <= help_button["x"] + help_button["width"] and
            help_button["y"] <= cy <= help_button["y"] + help_button["height"])
def draw_help_menu(frame, scroll_offset=0):
    height, width = frame.shape[:2]
    
    # Semi-transparent dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Create a centered menu box
    menu_width = int(width * 0.6)
    menu_height = int(height * 0.6)
    menu_x = (width - menu_width) // 2
    menu_y = (height - menu_height) // 2
    
    # Main content area with clipping
    cv2.rectangle(frame, (menu_x, menu_y), 
                 (menu_x + menu_width, menu_y + menu_height),
                 (50, 50, 50), -1)
    
    # Draw scroll bar
    scroll_bar_width = 20
    scroll_bar_x = menu_x + menu_width - scroll_bar_width
    
    # Draw scroll track
    cv2.rectangle(frame, 
                 (scroll_bar_x, menu_y),
                 (scroll_bar_x + scroll_bar_width, menu_y + menu_height),
                 (100, 100, 100), -1)
    
    # Calculate scroll thumb position and size
    visible_ratio = menu_height / instruction_height
    thumb_height = int(menu_height * visible_ratio)
    thumb_position = int(menu_y + (scroll_offset / instruction_height) * (menu_height - thumb_height))
    
    # Draw scroll thumb
    cv2.rectangle(frame,
                 (scroll_bar_x, thumb_position),
                 (scroll_bar_x + scroll_bar_width, thumb_position + thumb_height),
                 (200, 200, 200), -1)

    instructions = [
        "Welcome to the Quiz Game!",
        "",
        "How to Play:",
        "1. Read question carefully",
        "2. Use hand gestures to answer:",
        "   - Thumbs Up = YES",
        "   - Thumbs Down = NO",
        "3. Hold gesture for 1 second",
        "",
        "Navigation:",
        "- Scroll vertically to see more",
        "- Swipe right for next question",
        "",
        "Scoring:",
        "- Get 4+ correct for celebration",
        "- Less than 4? Keep practicing!",
        "",
        "Additional Controls:",
        "- Hover over HELP for 2.5s",
        "- 'R' key to restart game",
        "- 'Q' key to quit application",
        "",
        "Tips:",
        "- Keep your hand steady",
        "- Make clear gestures",
        "- Take your time to answer",
        "",
        "Good luck!"
    ]

    y_offset = 70 - scroll_offset
    for line in instructions:
        if menu_y <= y_offset <= menu_y + menu_height - 30:
            cv2.putText(frame, line,
                       (menu_x + 30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

    # Time remaining
    time_remaining = int(HELP_DISPLAY_DURATION - (time.time() - help_start_time))
    cv2.putText(frame, f"Menu closes in: {time_remaining}s",
                (menu_x + 30, menu_y + menu_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def draw_celebration(frame, elapsed_time):
    height, width = frame.shape[:2]
    n_particles = 20
    for i in range(n_particles):
        x = int(width/2 + np.sin(elapsed_time * 3 + i) * 200)
        y = int(height/2 + np.cos(elapsed_time * 3 + i) * 100)
        color = (0, 255, 255) if i % 2 == 0 else (255, 255, 0)
        cv2.circle(frame, (x, y), 10, color, -1)

    cv2.putText(frame, "CONGRATULATIONS!", (width//2 - 200, height//2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    cv2.putText(frame, f"You got {score}/5 correct!", (width//2 - 150, height//2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

def draw_disappointment(frame):
    height, width = frame.shape[:2]
    center = (width//2, height//2 - 50)
    cv2.circle(frame, center, 100, (0, 0, 255), 3)
    cv2.circle(frame, (center[0] - 40, center[1] - 20), 15, (0, 0, 255), -1)
    cv2.circle(frame, (center[0] + 40, center[1] - 20), 15, (0, 0, 255), -1)
    cv2.ellipse(frame, (center[0], center[1] + 40), (60, 40), 0, 0, 180, (0, 0, 255), 3)

    cv2.putText(frame, f"Score: {score}/5", (width//2 - 80, height//2 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, "Keep practicing!", (width//2 - 130, height//2 + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

gesture_start_time = None
gesture_detected = False
current_gesture = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    height, width = frame.shape[:2]

    cursor_x, cursor_y = 0, 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cursor_x, cursor_y = get_cursor_position(hand_landmarks, width, height)
            cv2.circle(frame, (cursor_x, cursor_y), 5, (0, 255, 0), -1)

    button_color = (0, 150, 255) if help_active else (0, 100, 200)
    if is_cursor_on_help(cursor_x, cursor_y):
        button_color = (0, 200, 255)

    cv2.rectangle(frame,
                 (help_button["x"], help_button["y"]),
                 (help_button["x"] + help_button["width"], 
                  help_button["y"] + help_button["height"]),
                 button_color, -1)
    cv2.putText(frame, "HELP",
                (help_button["x"] + 45, help_button["y"] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    current_time = time.time()

    if is_cursor_on_help(cursor_x, cursor_y) and not help_active:
        if help_hover_start_time is None:
            help_hover_start_time = current_time
        else:
            if current_time - help_hover_start_time >= HELP_ACTIVATION_TIME:
                help_active = True
                help_start_time = current_time
    else:
        help_hover_start_time = None

    if game_active and not showing_final_screen:
        cv2.rectangle(frame, (50, 50), (width-50, 200), (0, 0, 0), -1)
        question_text = questions[current_question]["question"]
        cv2.putText(frame, question_text, (70, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Score: {score}/{current_question}", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if results.multi_hand_landmarks and not help_active:
            for hand_landmarks in results.multi_hand_landmarks:
                # Handle horizontal scroll for next question
                if answer_shown and can_scroll_next:
                    is_scrolling, current_x = is_horizontal_scrolling(hand_landmarks, prev_x_pos)
                    if is_scrolling:
                        current_question += 1
                        answer_shown = False
                        gesture_detected = False
                        gesture_start_time = None
                        can_scroll_next = False
                        
                        if current_question >= len(questions):
                            game_active = False
                            showing_final_screen = True
                            final_screen_start_time = current_time
                    prev_x_pos = current_x
                else:
                    prev_x_pos = None

                if not answer_shown:
                    if is_thumbs_up(hand_landmarks):
                        current_gesture = "YES"
                    elif is_thumbs_down(hand_landmarks):
                        current_gesture = "NO"
                    else:
                        current_gesture = None
                        gesture_start_time = None
                        gesture_detected = False

                    if current_gesture and not gesture_detected:
                        if gesture_start_time is None:
                            gesture_start_time = current_time
                        
                        hold_time = current_time - gesture_start_time
                        cv2.putText(frame, f"Detected: {current_gesture}", 
                                    (500, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        if hold_time >= 1.0:
                            gesture_detected = True
                            answer_shown = True
                            answer_time = current_time
                            user_answer = (current_gesture == "YES")
                            if user_answer == questions[current_question]["correctAnswer"]:
                                score += 1
                                can_scroll_next = True

        if answer_shown:
            correct = (current_gesture == "YES") == questions[current_question]["correctAnswer"]
            color = (0, 255, 0) if correct else (0, 0, 255)
            result_text = "CORRECT!" if correct else "WRONG!"
            cv2.putText(frame, result_text, (500, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
            cv2.putText(frame, questions[current_question]["explanation"], (70, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if correct:
                cv2.putText(frame, "Swipe right for next question →", 
                            (width//2 - 200, height - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, "Thumbs Up=Yes | Thumbs Down=No", 
                    (width//2 - 200, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if score >= 4:
            draw_celebration(frame, time.time() - final_screen_start_time)
        else:
            draw_disappointment(frame)
        
        cv2.putText(frame, "Press 'R' to restart or 'Q' to quit", 
                    (width//2 - 200, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if help_active:
        if time.time() - help_start_time > HELP_DISPLAY_DURATION:
            help_active = False
            help_start_time = None
            scroll_offset = 0
        else:
            # Handle vertical scrolling in help menu
            if results.multi_hand_landmarks:
                is_scrolling, current_y = is_vertical_scrolling(hand_landmarks, prev_y_pos)
                if is_scrolling:
                    scroll_offset = min(max(0, scroll_offset + 10), MAX_SCROLL)
                prev_y_pos = current_y
            else:
                prev_y_pos = None
            
            draw_help_menu(frame, scroll_offset)

    cv2.imshow('Quiz Game', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and showing_final_screen:
        current_question = 0
        score = 0
        game_active = True
        answer_shown = False
        showing_final_screen = False
        gesture_detected = False
        help_active = False
        help_hover_start_time = None
        scroll_offset = 0
        can_scroll_next = False

cap.release()
cv2.destroyAllWindows()