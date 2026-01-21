import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------- Reproducibilidad ----------------
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- Dataset ----------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalización y dtype consistentes
x_train = (x_train.astype(np.float32) / 255.0)
x_test  = (x_test.astype(np.float32) / 255.0)

w, h = x_train.shape[1], x_train.shape[2]
n_classes = 10

# ---------------- Modelo (MLP) ----------------
model_mlp = Sequential([
    Flatten(input_shape=(w, h)),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(n_classes, activation="softmax"),
])

model_mlp.summary()

# ---------------- Optimizer ----------------
lr = 1e-3
op = Adam(learning_rate=lr)
# Alternativa si preferís SGD:
# op = SGD(learning_rate=0.05, momentum=0.9, nesterov=True)

model_mlp.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=op,
    metrics=["accuracy"],
)

# ---------------- Callbacks ----------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )
]

# ---------------- Entrenamiento ----------------
history = model_mlp.fit(
    x_train,
    y_train,
    epochs=32,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=2
)

# ---------------- Evaluación ----------------
test_loss, test_acc = model_mlp.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

# ---------------- Curvas ----------------
plt.figure(figsize=(10, 3))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy During Training")
plt.legend()
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss During Training")
plt.legend()
plt.show()

# ---------------- Inferencia individual ----------------
index = 12
plt.figure(figsize=(2, 2))
plt.imshow(x_test[index], cmap="gray")
plt.title("Image for Inference")
plt.axis("off")
plt.show()

x_one = x_test[index][None, ...]  # shape: (1, 28, 28)
probs = model_mlp.predict(x_one, verbose=0)
pred = int(np.argmax(probs, axis=1)[0])
print(f"Predicted Class: {pred} - True Label: {y_test[index]}")

# ---------------- Inferencia total + métricas ----------------
y_pred_probs = model_mlp.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix – MNIST (MLP)")
plt.show()
