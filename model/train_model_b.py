import tensorflow as tf
from pathlib import Path
from model_b import build_model_b


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
EPOCHS = 30

IMAGE_SIZE = 64

OUTPUT_DIR = Path("results/model_b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = Path("preprocessing/output/train")
VALID_DIR = Path("preprocessing/output/valid")
TEST_DIR = Path("preprocessing/output/test")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]


# ============================================================
# Load datasets
# ============================================================

print("\nLoading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

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


# ============================================================
# Normalize pixel values
# ============================================================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda images, labels: (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

valid_ds = valid_ds.map(
    lambda images, labels: (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    lambda images, labels: (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# Prefetch data
# ============================================================

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# ============================================================
# Build Model B
# ============================================================

print("\nBuilding Model B...")

model = build_model_b()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# Callbacks
# ============================================================

callbacks = [

    # Stop training if validation loss does not improve
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    # Save the model with the best validation accuracy
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "best_model_b.keras"),
        monitor="val_accuracy",
        save_best_only=True
    )
]


# ============================================================
# Train Model B
# ============================================================

print("\nStarting Model B training...\n")

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# Evaluate on test set
# ============================================================

print("\nEvaluating Model B on the test set...")

test_loss, test_accuracy = model.evaluate(test_ds)


print("\n" + "=" * 50)
print("MODEL B TEST RESULTS")
print("=" * 50)

print(f"Test loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_accuracy:.4f}")
print(f"Test accuracy : {test_accuracy * 100:.2f}%")

print("=" * 50)


# ============================================================
# Save final model
# ============================================================

final_model_path = OUTPUT_DIR / "final_model_b.keras"

model.save(final_model_path)

print(f"\nModel saved to: {final_model_path}")
print(
    f"Best model saved to: "
    f"{OUTPUT_DIR / 'best_model_b.keras'}"
)