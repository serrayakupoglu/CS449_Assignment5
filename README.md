**CS449-ASSIGNMENT 5**

---

# **Gesture-Based Quiz Game**

This project is a gesture-controlled True/False quiz game built using **MediaPipe Hands** and **OpenCV**. It leverages hand gesture recognition to create an interactive and intuitive user experience. The system supports gestures like "Thumbs Up," "Thumbs Down," and "Pointing" to control quiz navigation and interact with the help menu.

---

## **Features**
- **Gesture Recognition**:
  - **Thumbs Up**: Indicates a "YES" response.
  - **Thumbs Down**: Indicates a "NO" response.
  - **Pointing Gesture**: Simulates a virtual cursor for on-screen interactions.
- **Quiz Game**:
  - Displays a series of True/False questions.
  - Tracks user answers and updates the score dynamically.
  - Provides feedback with animations based on the user's performance.
- **Scrolling Functionality**:
  - **Vertical Scrolling**: Navigate the help menu using swiping gestures.
  - **Horizontal Scrolling**: Move between questions with smooth transitions.
- **Help Menu**:
  - Accessible via a 2.5-second hover over the "HELP" button.
  - Displays instructions for gameplay, gestures, and controls.

---

## **System Requirements**
- Python 3.7 or later
- OpenCV (`cv2`)
- MediaPipe
- NumPy

---

## **Setup and Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/serrayakupoglu/CS449_Assignment5.git
   cd CS449_Assignment5
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python gesture_quiz_game.py
   ```

---

## **Usage**
1. **Start the Game**:
   - Launch the script and position your hand in front of the webcam.
   - The first question will be displayed on the screen.
2. **Answer Questions**:
   - Hold a **Thumbs Up** gesture to select "YES."
   - Hold a **Thumbs Down** gesture to select "NO."
   - Gestures must be held for approximately 1 second to register.
3. **Navigate**:
   - Swipe right to move to the next question.
   - Hover over the "HELP" button for instructions.
   - Use vertical scrolling gestures to navigate the help menu.
4. **Complete the Quiz**:
   - After answering all questions, the system will display your score:
     - **Celebration Animation** for scores of 4 or higher.
     - **Disappointment Screen** for lower scores.

---

## **Contributors**
| Name                | Role                                                 |
|---------------------|------------------------------------------------------|
| **Serra Yakupoğlu** | Developed gesture detection and hover interactions.  |
| **Yiğit Kaan Tonkaz** | Implemented quiz logic and horizontal scrolling.    |
| **Melis Pehlivan**  | Designed vertical scrolling for the help menu.       |
| **Yağmur Dolunay**  | Integrated gesture logic with quiz management.       |

---



---

## **Video Demonstration**
[Watch the Demo Video](https://youtu.be/FQ4HprACDko)

---

## **Acknowledgments**
This project was developed as part of **CS 449: Assignment 5** at Sabanci University. It showcases the integration of gesture-based interaction systems using **MediaPipe Hands** for a user-friendly and immersive experience.

---
