import tensorflow as tf
from pathlib import Path

from model_a import build_model_a


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 30


OUTPUT_DIR = Path("results/model_a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# DATASET PATHS
# --------------------------------------------------

TRAIN_DIR = Path("preprocessing/output/train")
VALID_DIR = Path("preprocessing/output/valid")
TEST_DIR = Path("preprocessing/output/test")


IMAGE_SIZE = 64

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]


# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

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


# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

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


# --------------------------------------------------
# PREFETCH
# --------------------------------------------------

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# BUILD MODEL A
# --------------------------------------------------

print("\nBuilding Model A...")

model = build_model_a()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "best_model_a.keras"),
        monitor="val_accuracy",
        save_best_only=True
    )
]


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\nStarting Model A training...\n")

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

print("\nEvaluating Model A on the test set...")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\n" + "=" * 50)
print("MODEL A TEST RESULTS")
print("=" * 50)
print(f"Test loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_accuracy:.4f}")
print(f"Test accuracy : {test_accuracy * 100:.2f}%")
print("=" * 50)


# --------------------------------------------------
# SAVE FINAL MODEL
# --------------------------------------------------

model.save(OUTPUT_DIR / "final_model_a.keras")

print(f"\nModel saved to: {OUTPUT_DIR / 'final_model_a.keras'}")
print(f"Best model saved to: {OUTPUT_DIR / 'best_model_a.keras'}")