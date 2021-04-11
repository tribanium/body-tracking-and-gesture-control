import cv2
import mediapipe as mp


class handDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.static_image_mode,
            self.max_num_hands,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )

    def find_hands(self, frame, draw=True):
        """This function detects hands and draw landmarks on them."""

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if draw:
            if self.results.multi_hand_landmarks:
                for handLandmarks in self.results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, handLandmarks, self.mp_hands.HAND_CONNECTIONS
                    )

    def landmarks_position(self, frame, hand_number=0):
        """This function converts the landmarks position from decimal coordinates to integer coordinates.
        This will allow us to draw relatively to landmarks positions without the need of using built-in functions."""

        landmarks_list = []
        if self.results.multi_hand_landmarks:
            for id, landmark in enumerate(
                self.results.multi_hand_landmarks[hand_number].landmark
            ):
                height, width, channels = frame.shape
                cx = int(landmark.x * width)
                cy = int(landmark.y * height)
                landmarks_list.append([id, cx, cy])
        return landmarks_list
