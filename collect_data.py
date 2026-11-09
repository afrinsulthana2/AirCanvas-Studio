
import cv2
import mediapipe as mp
import numpy as np
import os

DATA_PATH = "dataset/gestures"

labels = {
    ord('1'): "draw",
    ord('2'): "stop",
    ord('3'): "erase"
}

# Create folders automatically
for label in labels.values():
    os.makedirs(os.path.join(DATA_PATH, label), exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened!")
    exit()

selected = None
counts = {}

for label in labels.values():
    folder = os.path.join(DATA_PATH, label)
    counts[label] = len([
        f for f in os.listdir(folder) if f.endswith(".npy")
    ])

print("AIR CANVAS DATA COLLECTION")
print("1 = DRAW | 2 = STOP | 3 = ERASE")
print("C = Capture sample | Q = Quit")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera frame not received")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        landmarks = None

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand.landmark
            ], dtype=np.float32)

            # Normalize coordinates
            landmarks -= landmarks[0]

            scale = np.max(np.abs(landmarks))

            if scale > 0:
                landmarks /= scale

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

        # Display instructions
        cv2.putText(
            frame,
            "1:DRAW  2:STOP  3:ERASE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        current = selected if selected else "None"

        cv2.putText(
            frame,
            f"Selected: {current}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {counts}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Press C to capture | Q to quit",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow("AirCanvas - Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in labels:
            selected = labels[key]
            print("Selected:", selected.upper())

        elif key == ord('c'):

            if selected is None:
                print("Select a gesture first!")
            elif landmarks is None:
                print("No hand detected!")
            else:
                filename = os.path.join(
                    DATA_PATH,
                    selected,
                    f"{counts[selected]}.npy"
                )

                np.save(filename, landmarks)
                counts[selected] += 1

                print(
                    f"{selected.upper()}: "
                    f"{counts[selected]} samples collected"
                )

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Dataset collection completed!")
