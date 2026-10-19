
from pathlib import Path
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

# -----------------------------
# SETTINGS
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "preprocessing" / "output"
SAVE_DIR = ROOT / "results" / "sota" / "mobilenetv2"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled_in_scale", "scratches"
]

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# -----------------------------
# LOAD DATASETS
# -----------------------------
def load_data(split):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        class_names=CLASSES,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=(split == "train"),
        seed=SEED
    )
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = load_data("train")
val_ds = load_data("valid")
test_ds = load_data("test")

# -----------------------------
# BUILD MOBILENETV2
# -----------------------------
base = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet",
    alpha=1.0
)

base.trainable = False

inputs = tf.keras.Input(shape=(128, 128, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(6, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

print("Total parameters:", model.count_params())

# -----------------------------
# TRAINING HELPERS
# -----------------------------
class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.start)

def callbacks(stage):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            SAVE_DIR / f"{stage}_best.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

# -----------------------------
# STAGE 1: FEATURE EXTRACTION
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

timer1 = EpochTimer()
start = time.perf_counter()

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks("stage1") + [timer1]
)

# -----------------------------
# STAGE 2: FINE-TUNING
# -----------------------------
base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization statistics fixed
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

timer2 = EpochTimer()

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks("stage2") + [timer2]
)

training_time = time.perf_counter() - start

# -----------------------------
# SELECT BEST STAGE CHECKPOINT
# -----------------------------
stage1_model = tf.keras.models.load_model(
    SAVE_DIR / "stage1_best.keras"
)
stage2_model = tf.keras.models.load_model(
    SAVE_DIR / "stage2_best.keras"
)

val1 = stage1_model.evaluate(val_ds, verbose=0)[0]
val2 = stage2_model.evaluate(val_ds, verbose=0)[0]

best_model = stage1_model if val1 <= val2 else stage2_model
best_model.save(SAVE_DIR / "best_model.keras")

print("Selected:", "Stage 1" if val1 <= val2 else "Stage 2")

# -----------------------------
# TEST EVALUATION
# -----------------------------
test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

y_prob = best_model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

report = classification_report(
    y_true, y_pred,
    target_names=CLASSES,
    digits=4,
    zero_division=0
)

(SAVE_DIR / "classification_report.txt").write_text(
    report, encoding="utf-8"
)

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay(
    cm, display_labels=CLASSES
).plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.tight_layout()
plt.savefig(SAVE_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# -----------------------------
# SAVE TRAINING CURVES
# -----------------------------
history = {}
for key in ["loss", "accuracy", "val_loss", "val_accuracy"]:
    history[key] = history1.history[key] + history2.history[key]

epochs = np.arange(1, len(history["loss"]) + 1)
stage_boundary = len(history1.history["loss"])

for metric in ["accuracy", "loss"]:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history[metric], label=f"Training {metric}")
    ax.plot(epochs, history[f"val_{metric}"],
            label=f"Validation {metric}")
    ax.axvline(stage_boundary + 0.5,
               linestyle="--", label="Fine-tuning begins")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.title())
    ax.set_title(f"MobileNetV2: {metric.title()} vs Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"{metric}_vs_epochs.png", dpi=300)
    plt.close()

# -----------------------------
# SAVE SUMMARY
# -----------------------------
model_size_mb = (
    (SAVE_DIR / "best_model.keras").stat().st_size / (1024 ** 2)
)

with open(SAVE_DIR / "summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Model", "Parameters", "Model Size (MB)",
        "Test Accuracy (%)", "Macro F1", "Training Time (s)"
    ])
    writer.writerow([
        "MobileNetV2",
        best_model.count_params(),
        round(model_size_mb, 3),
        round(test_acc * 100, 2),
        round(f1_score(y_true, y_pred, average="macro"), 4),
        round(training_time, 2)
    ])

print("\nMOBILENETV2 RESULTS")
print("Test accuracy:", round(test_acc * 100, 2), "%")
print("Total parameters:", best_model.count_params())
print("Model size:", round(model_size_mb, 3), "MB")
print("Macro F1:", round(f1_score(y_true, y_pred, average="macro"), 4))
print("Training time:", round(training_time, 2), "seconds")
print("\nSaved results to:", SAVE_DIR)
