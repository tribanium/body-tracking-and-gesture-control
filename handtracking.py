import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):

        self.mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        mpHands = mp.solutions.hands
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils


def handLabels(id):
    if id == 4:
        cv2.putText(
            img, "POUCE", (cx + 10, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1,
        )
    if id == 8:
        cv2.putText(
            img, "INDEX", (cx + 10, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1,
        )
    if id == 12:
        cv2.putText(
            img, "MAJEUR", (cx + 10, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1,
        )
    if id == 16:
        cv2.putText(
            img, "ANNULAIRE", (cx + 10, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1,
        )
    if id == 20:
        cv2.putText(
            img, "AURICULAIRE", (cx + 10, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1,
        )


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    ret, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # for loop over the hands if a hand is detected
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            # for id, landmark in enumerate(handLandmarks.landmark):
            #     height, width, channels = frame.shape
            #     cx, cy = int(landmark.x * width), int(landmark.y * height)
            #     if id in [4, 8, 12, 16, 20]:
            #         cv2.circle(frame, (cx, cy), 10, (0, 0, 0), cv2.FILLED)
            #     handLabels(id)

            mpDraw.draw_landmarks(frame, handLandmarks, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(
        frame, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    cv2.imshow("Hands tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

