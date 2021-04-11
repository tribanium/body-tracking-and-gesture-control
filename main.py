import handtracking
import cv2
import time


def main():
    previous_time = 0
    current_time = 0

    cap = cv2.VideoCapture(0)
    detector = handtracking.handDetector()

    while True:
        ret, frame = cap.read()
        detector.find_hands(frame)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(
            frame,
            f"{int(fps)} FPS",
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Hands tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
