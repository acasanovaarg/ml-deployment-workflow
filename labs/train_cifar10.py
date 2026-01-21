import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    RandomFlip, RandomTranslation, RandomRotation,
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10

# ---------------- Reproducibilidad ----------------
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# (Opcional) Mejora determinismo; puede afectar performance
# tf.config.experimental.enable_op_determinism()

# ---------------- Output dirs ----------------
run_id = time.strftime("%Y%m%d-%H%M%S")
out_dir = os.path.join("runs", f"cifar10_cnn_{run_id}")
os.makedirs(out_dir, exist_ok=True)

ckpt_path = os.path.join(out_dir, "best_model.keras")
final_model_path = os.path.join(out_dir, "final_model.keras")
csv_log_path = os.path.join(out_dir, "training_log.csv")
tb_log_dir = os.path.join(out_dir, "tb")

# ---------------- Dataset ----------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Aplano labels: (N, 1) -> (N,)
y_train = y_train.squeeze().astype(np.int64)
y_test = y_test.squeeze().astype(np.int64)

# Normalizo + dtype consistente
x_train = (x_train.astype(np.float32) / 255.0)
x_test  = (x_test.astype(np.float32) / 255.0)

print("x_train:", x_train.shape, x_train.dtype, "y_train:", y_train.shape, y_train.dtype)
print("x_test :", x_test.shape, x_test.dtype, "y_test :", y_test.shape, y_test.dtype)

# ---------------- Modelo CNN (mejorado) ----------------
l2 = keras.regularizers.l2(1e-4)

model = Sequential([
    keras.Input(shape=(32, 32, 3)),

    RandomFlip("horizontal"),
    RandomTranslation(0.1, 0.1),
    RandomRotation(0.05),

    Conv2D(32, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.10),

    Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.15),

    Conv2D(128, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2),
    BatchNormalization(),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.20),

    GlobalAveragePooling2D(),
    Dense(128, activation="relu", kernel_regularizer=l2),
    Dropout(0.40),

    Dense(10, activation="softmax"),
])

model.summary()

# ---------------- Compile ----------------
op = Adam(learning_rate=1e-3)

model.compile(
    optimizer=op,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---------------- Callbacks ----------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    keras.callbacks.CSVLogger(csv_log_path),
    keras.callbacks.TensorBoard(log_dir=tb_log_dir),
    keras.callbacks.TerminateOnNaN(),
]

# ---------------- Train ----------------
history = model.fit(
    x_train,
    y_train,
    epochs=64,             # más alto porque EarlyStopping corta solo
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
)

# ---------------- Evaluate ----------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# Guardado final (además del best checkpoint)
model.save(final_model_path)

# ---------------- Plots ----------------
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

print("\nArtifacts saved in:", out_dir)
print("Best checkpoint:", ckpt_path)
print("Final model   :", final_model_path)
print("CSV log       :", csv_log_path)
print("TensorBoard   :", tb_log_dir)
