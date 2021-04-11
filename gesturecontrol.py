import cv2
import time
import numpy as np
import handtracking
from math import sqrt

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
cam_width, cam_height = 640, 480
cap.set(3, cam_width)
cap.set(4, cam_height)

previous_time = 0

detector = handtracking.handDetector(min_detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()  # (min = -63.5, max = 0.0, ? = 0.5)
min_volume = volume_range[0]
max_volume = volume_range[1]
vol = 0

volume_bar = 400
volume_percentage = 0

while True:
    ret, frame = cap.read()
    detector.find_hands(frame, draw=True)
    landmarks_list = detector.landmarks_position(frame, hand_number=0)

    if landmarks_list:
        thumb_coord = (landmarks_list[4][1], landmarks_list[4][2])
        index_coord = (landmarks_list[8][1], landmarks_list[8][2])
        center_coord = (
            (thumb_coord[0] + index_coord[0]) // 2,
            (thumb_coord[1] + index_coord[1]) // 2,
        )
        cv2.circle(frame, thumb_coord, 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, index_coord, 10, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, thumb_coord, index_coord, (255, 0, 0), 3)
        cv2.circle(frame, center_coord, 10, (255, 0, 255), cv2.FILLED)

        line_length = sqrt(
            (thumb_coord[0] - index_coord[0]) ** 2
            + (thumb_coord[1] - index_coord[1]) ** 2
        )

        # Mapping between line_length range and volume range
        vol = np.interp(line_length, [30, 400], [max_volume, min_volume])
        volume_bar = np.interp(line_length, [30, 400], [150, 400])
        volume_percentage = np.interp(line_length, [30, 400], [100, 0])
        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(frame, (50, int(volume_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(
        frame,
        f"{int(volume_percentage)} %",
        (40, 450),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 255, 0),
        3,
    )

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(
        frame, f"{int(fps)} FPS", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2,
    )
    cv2.imshow("Video Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

