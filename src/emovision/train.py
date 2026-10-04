import os
import json
import argparse
from typing import List, Tuple

import numpy as np

from .config import IMG_SIZE, DATA_DIR, MODEL_PATH, LABELS_SUFFIX
from .utils.dataset import find_image_paths
from .utils.preprocess import load_and_preprocess


def load_dataset(root: str, classes: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    paths, labels, classes = find_image_paths(root, classes)
    if len(paths) == 0:
        raise SystemExit(f"No images found under '{root}'. Run the collector first.")
    X = np.stack([load_and_preprocess(p, IMG_SIZE) for p in paths])
    return X, labels, classes


def build_model(num_classes: int, img_size: int = IMG_SIZE):
    from tensorflow import keras
    from tensorflow.keras import layers

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        return layers.Dropout(0.25)(x)

    inputs = keras.Input(shape=(img_size, img_size, 1))
    # Augmentation runs only during training (Keras disables these layers at inference)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomContrast(0.2)(x)

    for filters in (32, 64, 128):
        x = conv_block(x, filters)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="emovision_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_labels(model_path: str, classes: List[str]) -> str:
    path = model_path + LABELS_SUFFIX
    with open(path, "w") as f:
        json.dump(classes, f, indent=2)
    return path


def plot_history(history, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(history.history[metric], label="train")
        if f"val_{metric}" in history.history:
            ax.plot(history.history[f"val_{metric}"], label="val")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the emotion classifier.")
    parser.add_argument("--data", default=DATA_DIR, help="Dataset root (one subfolder per class)")
    parser.add_argument("--classes", nargs="+", default=None, help="Class names (default: all subfolders, sorted)")
    parser.add_argument("--model", default=MODEL_PATH, help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, confusion_matrix

    keras.utils.set_random_seed(args.seed)

    X, y, classes = load_dataset(args.data, args.classes)
    counts = np.bincount(y, minlength=len(classes))
    print(f"Loaded {len(X)} images: " + ", ".join(f"{c}={n}" for c, n in zip(classes, counts)))
    present = counts > 0
    if present.sum() < 2:
        raise SystemExit("Need images for at least 2 classes to train.")
    if not present.all():
        print("Warning: no images for " + ", ".join(c for c, p in zip(classes, present) if not p))

    stratify = y if counts[present].min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed, stratify=stratify
    )

    # Balance the loss when some emotions have fewer samples than others
    train_classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=train_classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(train_classes, weights)}

    model = build_model(len(classes))
    model.summary()

    os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(args.model, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(len(X_train), seed=args.seed)
                .batch(args.batch_size)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # EarlyStopping restored the best weights; save them so the file matches the report below
    model.save(args.model)
    labels_path = save_labels(args.model, classes)
    plot_path = os.path.splitext(args.model)[0] + "_history.png"
    plot_history(history, plot_path)

    y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    labels = list(range(len(classes)))
    print("\nValidation report:")
    print(classification_report(y_val, y_pred, labels=labels, target_names=classes, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_val, y_pred, labels=labels))

    print(f"\nSaved model to {args.model}")
    print(f"Saved labels to {labels_path}")
    print(f"Saved training curves to {plot_path}")


if __name__ == "__main__":
    main()
