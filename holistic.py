import mediapipe as mp
import cv2
import time

DRAW_FACE = True
DRAW_POSE = True
DRAW_HANDS = True

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

previousTime = 0
currentTime = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while True:
        ret, frame = cap.read()

        # Recolor the feed for mp input
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make detections
        results = holistic.process(imgRGB)

        # Draw face landmarks
        if DRAW_FACE:
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(80, 110, 10), thickness=1, circle_radius=1
                ),
                mp_drawing.DrawingSpec(
                    color=(80, 256, 121), thickness=1, circle_radius=1
                ),
            )
        # Draw pose landmarks
        if DRAW_POSE:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )
        # Draw hands
        if DRAW_HANDS:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(
            frame,
            f"{int(fps)} FPS",
            (10, 50),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Holistic Model Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
