
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "dataset/gestures"

labels = ["draw", "stop", "erase"]

X = []
y = []

# Load collected gesture data
for index, label in enumerate(labels):

    folder = os.path.join(DATA_PATH, label)

    for file in os.listdir(folder):

        if file.endswith(".npy"):

            data = np.load(os.path.join(folder, file))

            X.append(data)
            y.append(index)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Total samples:", len(X))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# CNN architecture
model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(21, 3)),

    tf.keras.layers.Conv1D(
        32, 3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(
        64, 3,
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train CNN
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Classification report
predictions = np.argmax(
    model.predict(X_test),
    axis=1
)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2],
    target_names=labels,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

# Save trained model
os.makedirs("models", exist_ok=True)

model.save("models/gesture_model.keras")

print("\nCNN model saved successfully!")
