import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)

IMAGE_SIZE = 64
BATCH_SIZE = 32

TEST_DIR = Path("preprocessing/output/test")
MODEL_PATH = Path("results/final_custom/model_a/best_model_a.keras")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

test_ds = test_ds.map(
    lambda images, labels:
    (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

print("\nLoading Model A...")

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")
print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# Accuracy
# ============================================================

accuracy = np.mean(y_true == y_pred)


# ============================================================
# Precision and Recall
# ============================================================

precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("MODEL A TEST EVALUATION")
print("=" * 60)

print(f"\nTest accuracy   : {accuracy:.4f}")
print(f"Test accuracy   : {accuracy * 100:.2f}%")
print(f"Macro precision : {precision_macro:.4f}")
print(f"Macro recall    : {recall_macro:.4f}")


# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report:")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(CLASS_NAMES))
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Save results
# ============================================================

OUTPUT_DIR = Path("results/model_a")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

np.savetxt(
    OUTPUT_DIR / "confusion_matrix.txt",
    cm,
    fmt="%d"
)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as f:

    f.write("MODEL A TEST EVALUATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Test accuracy   : {accuracy:.4f}\n"
    )

    f.write(
        f"Test accuracy   : "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Macro precision : "
        f"{precision_macro:.4f}\n"
    )

    f.write(
        f"Macro recall    : "
        f"{recall_macro:.4f}\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))


print("\nResults saved to:")
print(
    OUTPUT_DIR /
    "confusion_matrix.txt"
)

print(
    OUTPUT_DIR /
    "classification_report.txt"
)

print("\nEvaluation complete.")

