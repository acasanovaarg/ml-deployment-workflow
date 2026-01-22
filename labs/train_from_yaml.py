#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import argparse
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

# Requiere: pip install pyyaml
import yaml


# ---------------- Utilities ----------------

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def deep_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def make_out_dir(cfg: Dict[str, Any]) -> str:
    out_root = deep_get(cfg, "run.out_root", "runs")
    prefix = deep_get(cfg, "run.name_prefix", "run")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out_root, f"{prefix}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# Map strings in YAML -> actual Keras layer classes
LAYER_REGISTRY = {
    "RandomFlip": layers.RandomFlip,
    "RandomTranslation": layers.RandomTranslation,
    "RandomRotation": layers.RandomRotation,
    "ZeroPadding2D": layers.ZeroPadding2D,
    "RandomCrop": layers.RandomCrop,

    "Conv2D": layers.Conv2D,
    "BatchNormalization": layers.BatchNormalization,
    "Activation": layers.Activation,
    "MaxPooling2D": layers.MaxPooling2D,
    "Dropout": layers.Dropout,
    "Flatten": layers.Flatten,
    "GlobalAveragePooling2D": layers.GlobalAveragePooling2D,
    "Dense": layers.Dense,
}

OPTIM_REGISTRY = {
    "Adam": keras.optimizers.Adam,
    "SGD": keras.optimizers.SGD,
    "AdamW": getattr(keras.optimizers, "AdamW", None),  # puede ser None según TF
}

CALLBACK_REGISTRY = {
    "EarlyStopping": keras.callbacks.EarlyStopping,
    "ReduceLROnPlateau": keras.callbacks.ReduceLROnPlateau,
    "ModelCheckpoint": keras.callbacks.ModelCheckpoint,
    "CSVLogger": keras.callbacks.CSVLogger,
    "TensorBoard": keras.callbacks.TensorBoard,
    "TerminateOnNaN": keras.callbacks.TerminateOnNaN,
}


def build_layer(layer_cfg: Dict[str, Any], l2_value: float | None) -> keras.layers.Layer:
    layer_type = layer_cfg["type"]
    cls = LAYER_REGISTRY.get(layer_type)
    if cls is None:
        raise ValueError(f"Unsupported layer type in YAML: {layer_type}")

    args = layer_cfg.get("args", [])
    kwargs = layer_cfg.get("kwargs", {}).copy()

    # Auto-apply L2 to Conv2D/Dense if configured and not explicitly set
    if l2_value is not None and layer_type in ("Conv2D", "Dense"):
        kwargs.setdefault("kernel_regularizer", keras.regularizers.l2(l2_value))

    return cls(*args, **kwargs)


def build_model(cfg: Dict[str, Any]) -> keras.Model:
    input_shape = tuple(deep_get(cfg, "model.input_shape"))
    l2_value = deep_get(cfg, "regularization.l2", None)

    seq = keras.Sequential([keras.Input(shape=input_shape)])

    for layer_cfg in deep_get(cfg, "model.layers", []):
        seq.add(build_layer(layer_cfg, l2_value))

    return seq


def build_optimizer(cfg: Dict[str, Any]) -> keras.optimizers.Optimizer:
    opt_cfg = deep_get(cfg, "compile.optimizer")
    opt_type = opt_cfg["type"]
    opt_kwargs = opt_cfg.get("kwargs", {})

    cls = OPTIM_REGISTRY.get(opt_type)
    if cls is None:
        raise ValueError(f"Unsupported/Unavailable optimizer: {opt_type}")

    return cls(**opt_kwargs)


def build_callbacks(cfg: Dict[str, Any], out_dir: str) -> list:
    cbs = []
    for cb_cfg in deep_get(cfg, "callbacks", []):
        cb_type = cb_cfg["type"]
        cls = CALLBACK_REGISTRY.get(cb_type)
        if cls is None:
            raise ValueError(f"Unsupported callback type: {cb_type}")

        kwargs = cb_cfg.get("kwargs", {}).copy()

        # Fill paths automatically
        if cb_type == "ModelCheckpoint":
            kwargs.setdefault("filepath", os.path.join(out_dir, "best_model.keras"))
        elif cb_type == "CSVLogger":
            kwargs.setdefault("filename", os.path.join(out_dir, "training_log.csv"))
        elif cb_type == "TensorBoard":
            kwargs.setdefault("log_dir", os.path.join(out_dir, "tb"))

        cbs.append(cls(**kwargs))
    return cbs


def load_data(cfg: Dict[str, Any]):
    dataset = deep_get(cfg, "data.dataset", "cifar10")

    if dataset != "cifar10":
        raise ValueError(f"Only cifar10 implemented in this example, got: {dataset}")

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    y_train = y_train.squeeze().astype(np.int64)
    y_test = y_test.squeeze().astype(np.int64)

    if deep_get(cfg, "data.normalize", True):
        dtype = deep_get(cfg, "data.dtype", "float32")
        np_dtype = np.float32 if dtype == "float32" else np.float64
        x_train = (x_train.astype(np_dtype) / 255.0)
        x_test = (x_test.astype(np_dtype) / 255.0)

    return (x_train, y_train), (x_test, y_test)


def plot_history(history: keras.callbacks.History) -> None:
    plt.figure(figsize=(10, 3))
    plt.plot(history.history.get("accuracy", []), label="Train Accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy During Training")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.plot(history.history.get("loss", []), label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss During Training")
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(deep_get(cfg, "run.seed", 123))
    set_seed(seed)

    out_dir = make_out_dir(cfg)
    print("Output dir:", out_dir)

    (x_train, y_train), (x_test, y_test) = load_data(cfg)

    model = build_model(cfg)
    model.summary()

    optimizer = build_optimizer(cfg)

    model.compile(
        optimizer=optimizer,
        loss=deep_get(cfg, "compile.loss"),
        metrics=deep_get(cfg, "compile.metrics", []),
    )

    callbacks = build_callbacks(cfg, out_dir)

    epochs = int(deep_get(cfg, "train.epochs", 64))
    batch_size = int(deep_get(cfg, "train.batch_size", 64))
    val_split = float(deep_get(cfg, "data.validation_split", 0.2))
    verbose = int(deep_get(cfg, "train.verbose", 1))

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=verbose
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Save final model
    final_model_path = os.path.join(out_dir, "final_model.keras")
    model.save(final_model_path)

    plot_history(history)

    print("\nArtifacts saved in:", out_dir)
    print("Final model:", final_model_path)


if __name__ == "__main__":
    main()
