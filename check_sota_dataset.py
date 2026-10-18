
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "preprocessing" / "output"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

def load_sota_dataset(split):
    folder = DATA_DIR / split

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=42
    )

    return dataset.prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        print(f"\nChecking {split} dataset...")

        dataset = load_sota_dataset(split)
        images, labels = next(iter(dataset))

        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Pixel range:",
              float(tf.reduce_min(images)),
              float(tf.reduce_max(images)))

    print("\nSOTA dataset loading successful!")
