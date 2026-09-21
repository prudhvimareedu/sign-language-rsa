import cv2
import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from backend.vision.hand_detector import HandDetector
from backend.ai.asl_inference import ASLInference


def main():

    print("========================================")
    print("ASL RSA LIVE WEBCAM TEST")
    print("========================================")

    detector = HandDetector()
    inference = ASLInference()

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print()
    print("Webcam started.")
    print("Show one ASL sign.")
    print("Hold it steady for 2-3 seconds.")
    print("Press Q to quit.")
    print()

    last_prediction = None
    frame_count = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        frame_count += 1

        results = detector.detect(frame)

        prediction = "No hand"
        confidence = 0.0

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            try:

                output = inference.predict(
                    hand_landmarks
                )

                prediction = output["prediction"]
                confidence = output["confidence"]

            except Exception as e:

                prediction = "Error"

                if last_prediction != "Error":
                    print("Prediction error:", e)

        if prediction != last_prediction:

            print(
                f"Prediction changed -> "
                f"{prediction} | "
                f"Score: {confidence:.3f}"
            )

            last_prediction = prediction

        frame = detector.draw_landmarks(
            frame,
            results
        )

        cv2.rectangle(
            frame,
            (10, 10),
            (460, 110),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Score: {confidence:.2f}",
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ASL RSA Live Test",
            frame
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print()
    print("========================================")
    print("LIVE TEST FINISHED")
    print("========================================")
    print("Frames processed:", frame_count)
    print("Last prediction:", last_prediction)


if __name__ == "__main__":
    main()