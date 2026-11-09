
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from collections import deque, Counter

# --------------------------------
# 1. LOAD TRAINED CNN MODEL
# --------------------------------

model = tf.keras.models.load_model(
    "models/gesture_model.keras"
)

labels = ["DRAW", "STOP", "ERASE"]

# Prediction settings
prediction_history = deque(maxlen=9)
CONFIDENCE_THRESHOLD = 0.80
MIN_VOTES = 6

# --------------------------------
# 2. INITIALIZE WEBCAM
# --------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    exit()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# --------------------------------
# 3. HAND TRACKING
# --------------------------------

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Camera frame not received!")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        prediction_text = "NO HAND DETECTED"
        confidence = 0.0

        # --------------------------------
        # 4. PROCESS HAND LANDMARKS
        # --------------------------------

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            points = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand.landmark
            ], dtype=np.float32)

            # Normalize landmarks
            points -= points[0]

            scale = np.max(np.abs(points))

            if scale > 0:
                points /= scale

                # --------------------------------
                # 5. CNN PREDICTION
                # --------------------------------

                prediction = model.predict(
                    np.expand_dims(points, axis=0),
                    verbose=0
                )[0]

                index = int(np.argmax(prediction))
                confidence = float(prediction[index])

                # --------------------------------
                # 6. CONFIDENCE FILTERING
                # --------------------------------

                if confidence >= CONFIDENCE_THRESHOLD:

                    prediction_history.append(
                        labels[index]
                    )

                else:
                    prediction_history.clear()

                # --------------------------------
                # 7. PREDICTION SMOOTHING
                # --------------------------------

                if prediction_history:

                    most_common = Counter(
                        prediction_history
                    ).most_common(1)[0]

                    gesture, votes = most_common

                    if votes >= MIN_VOTES:
                        prediction_text = gesture
                    else:
                        prediction_text = "STABILIZING"

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

            else:
                prediction_history.clear()

        else:
            prediction_history.clear()

        # --------------------------------
        # 8. DISPLAY RESULTS
        # --------------------------------

        cv2.putText(
            frame,
            f"Gesture: {prediction_text}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2%}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Stable votes: {len(prediction_history)}/{MIN_VOTES}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q - QUIT",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "AirCanvas - Stable CNN Gesture Recognition",
            frame
        )

        # --------------------------------
        # 9. EXIT
        # --------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("AirCanvas gesture testing closed.")
