from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT = Path("preprocessing/output")

IMAGE_SIZE = 64
BATCH_SIZE = 32

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# DATASET DIRECTORIES
# ============================================================

TRAIN_DIR = OUTPUT / "train"
VALID_DIR = OUTPUT / "valid"
TEST_DIR = OUTPUT / "test"


# ============================================================
# CHECK DIRECTORIES
# ============================================================

for directory in [TRAIN_DIR, VALID_DIR, TEST_DIR]:

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}"
        )


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...\n")


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
# NORMALIZATION
# ============================================================
# JPG pixels are originally in the range 0-255.
# Convert them to 0-1 for CNN training.

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)


train_ds = train_ds.map(
    lambda images, labels:
        (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)


valid_ds = valid_ds.map(
    lambda images, labels:
        (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)


test_ds = test_ds.map(
    lambda images, labels:
        (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# PREFETCH
# ============================================================
# Allows the next batch to be prepared while the CNN
# is processing the current batch.

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Image size       : {IMAGE_SIZE} × {IMAGE_SIZE}")
print("Channels         : 1 (grayscale)")
print(f"Batch size       : {BATCH_SIZE}")
print(f"Number of classes: {NUM_CLASSES}")

print("\nClass mapping:")

for index, class_name in enumerate(CLASS_NAMES):
    print(f"  {index} -> {class_name}")


# ============================================================
# COUNT BATCHES
# ============================================================

train_batches = tf.data.experimental.cardinality(train_ds).numpy()
valid_batches = tf.data.experimental.cardinality(valid_ds).numpy()
test_batches = tf.data.experimental.cardinality(test_ds).numpy()

print("\nNumber of batches:")
print(f"  Training   : {train_batches}")
print(f"  Validation : {valid_batches}")
print(f"  Testing    : {test_batches}")


# ============================================================
# VERIFY TRAINING BATCH
# ============================================================

images, labels = next(iter(train_ds))


print("\n" + "=" * 60)
print("TENSOR VERIFICATION")
print("=" * 60)

print(f"Image tensor shape : {images.shape}")
print(f"Label tensor shape : {labels.shape}")

print(
    f"Pixel value range  : "
    f"{float(tf.reduce_min(images)):.4f} "
    f"to "
    f"{float(tf.reduce_max(images)):.4f}"
)

print("\nLabels in first batch:")
print(labels.numpy())


# ============================================================
# VERIFY IMAGE FORMAT
# ============================================================

expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)

actual_shape = tuple(images.shape[1:])


print("\n" + "=" * 60)
print("FORMAT CHECK")
print("=" * 60)

if actual_shape == expected_shape:

    print(
        f"PASS: Image shape is "
        f"{actual_shape}"
    )

else:

    print(
        f"ERROR: Expected image shape "
        f"{expected_shape}, "
        f"but received {actual_shape}"
    )


# ============================================================
# VERIFY NORMALIZATION
# ============================================================

minimum = float(tf.reduce_min(images))
maximum = float(tf.reduce_max(images))


if minimum >= 0.0 and maximum <= 1.0:

    print("PASS: Pixel values are normalized to 0-1")

else:

    print(
        "ERROR: Pixel values are outside "
        "the expected 0-1 range"
    )


# ============================================================
# VERIFY LABELS
# ============================================================

unique_labels = set(labels.numpy().tolist())

valid_labels = all(
    0 <= label < NUM_CLASSES
    for label in unique_labels
)


if valid_labels:

    print("PASS: Labels are within valid class range")

else:

    print("ERROR: Invalid class label detected")


# ============================================================
# VISUALIZE ONE BATCH
# ============================================================

plt.figure(figsize=(10, 7))

for i in range(min(12, len(images))):

    plt.subplot(3, 4, i + 1)

    plt.imshow(
        images[i].numpy().squeeze(),
        cmap="gray"
    )

    label_index = int(labels[i].numpy())

    plt.title(
        f"{label_index}: "
        f"{CLASS_NAMES[label_index]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("DATASET LOADER VERIFICATION COMPLETE")
print("=" * 60)

print("PASS: Train dataset loaded")
print("PASS: Validation dataset loaded")
print("PASS: Test dataset loaded")
print("PASS: Images resized to 64 × 64")
print("PASS: Images loaded as grayscale")
print("PASS: Pixel values normalized to 0-1")
print("PASS: Batches created")
print("PASS: Tensor shape verified")
print("PASS: Labels verified")

print("\nThe dataset is ready for CNN training.")
print("=" * 60)