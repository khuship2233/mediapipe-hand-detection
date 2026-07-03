import cv2
import mediapipe as mp

# initialize mediapipe hands module
mp_hands = mp.solutions.hands   # load the mediapipe hands module
hands = mp_hands.Hands()        # create the hand detector object

# Draw utility for drawing the hand landmarks
mp_drawing = mp.solutions.drawing_utils   # draw the hand landmarks

# Open default camera
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    # Initialize hand count
    hand_count = 0

    # If hands are detected
    if result.multi_hand_landmarks:

        hand_count = len(result.multi_hand_landmarks)

        for i in result.multi_hand_landmarks:

            # Draw the hand landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                i,
                mp_hands.HAND_CONNECTIONS
            )

        # Display "Hand Detected"
        cv2.putText(frame,
                    "Hand Detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    # Display number of hands detected
    cv2.putText(frame,
                "Hands: " + str(hand_count),
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    # Display the webcam feed
    cv2.imshow("Hand Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()