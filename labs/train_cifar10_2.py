import os, time, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Data
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.squeeze().astype(np.int64)
y_test  = y_test.squeeze().astype(np.int64)
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

# Output
run_id = time.strftime("%Y%m%d-%H%M%S")
out_dir = os.path.join("runs", f"cifar10_cnn_{run_id}")
os.makedirs(out_dir, exist_ok=True)

# Data augmentation inside model
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomRotation(0.05),
], name="data_aug")

l2 = keras.regularizers.l2(1e-4)

inputs = keras.Input(shape=(32, 32, 3))
x = data_aug(inputs)

# Block 1
x = layers.Conv2D(32, 3, padding="same", use_bias=False, kernel_regularizer=l2)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.1)(x)

# Block 2
x = layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_regularizer=l2)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.15)(x)

# Block 3
x = layers.Conv2D(128, 3, padding="same", use_bias=False, kernel_regularizer=l2)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.2)(x)

# Head (key change)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=l2)(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()

# LR schedule (CosineDecay) - reemplaza ReduceLROnPlateau
steps_per_epoch = int(np.ceil(len(x_train) * 0.8 / 64))  # ~por validation_split 0.2
total_steps = steps_per_epoch * 80

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=total_steps,
    alpha=1e-2  # LR final = 1% del inicial
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(out_dir, "best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.CSVLogger(os.path.join(out_dir, "log.csv")),
    keras.callbacks.TensorBoard(log_dir=os.path.join(out_dir, "tb")),
    keras.callbacks.TerminateOnNaN(),
]

history = model.fit(
    x_train, y_train,
    epochs=80,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test acc: {test_acc:.4f} | loss: {test_loss:.4f}")
