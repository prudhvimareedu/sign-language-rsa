import cv2

from hand_detector import HandDetector
from feature_extractor import FeatureExtractor


detector = HandDetector()
extractor = FeatureExtractor()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("========================================")
print("HAND FEATURE EXTRACTION TEST")
print("========================================")
print("Show one hand clearly to the camera.")
print("Press Q to quit.")
print("========================================")


features_printed = False


while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    results = detector.detect(frame)

    frame = detector.draw_landmarks(
        frame,
        results
    )

    if results.multi_hand_landmarks:

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        if not features_printed:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            features = extractor.extract(
                hand_landmarks
            )

            print()
            print("==============================")
            print("HAND DETECTED")
            print("Number of features:", len(features))
            print("Features:")
            print(features)
            print("==============================")
            print()

            features_printed = True

    else:

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Sign Language - Feature Extraction",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()