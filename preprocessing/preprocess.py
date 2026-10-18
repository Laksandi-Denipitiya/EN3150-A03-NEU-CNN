import cv2
import random
from pathlib import Path

# =========================
# Dataset paths
# =========================

DATASET = Path(
    r"D:\5th sem\Pattern recognition\Project\NEU-CLS"
)

TRAIN_IMAGES = DATASET / "train" / "train" / "images"
TRAIN_LABELS = DATASET / "train" / "train" / "labels"

VALID_IMAGES = DATASET / "valid" / "valid" / "images"
VALID_LABELS = DATASET / "valid" / "valid" / "labels"

# Output directory
OUTPUT = Path("preprocessing/output")

# CNN input size
IMAGE_SIZE = 64

# Reproducibility
random.seed(42)

# Class names
CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled_in_scale",
    5: "scratches"
}


# =========================
# Create output folders
# =========================

for split in ["train", "valid", "test"]:
    for class_name in CLASS_NAMES.values():
        (OUTPUT / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


# =========================
# Collect all images
# =========================

all_images = []

for image_dir, label_dir in [
    (TRAIN_IMAGES, TRAIN_LABELS),
    (VALID_IMAGES, VALID_LABELS)
]:

    for image_path in image_dir.glob("*.jpg"):

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            all_images.append((image_path, label_path))


print("Total images found:", len(all_images))


# =========================
# Shuffle images
# =========================

random.shuffle(all_images)


# =========================
# 70 / 15 / 15 split
# =========================

total = len(all_images)

train_end = int(0.70 * total)
valid_end = int(0.85 * total)

train_images = all_images[:train_end]
valid_images = all_images[train_end:valid_end]
test_images = all_images[valid_end:]


print("Training images:", len(train_images))
print("Validation images:", len(valid_images))
print("Testing images:", len(test_images))


# =========================
# Process images
# =========================

def process_images(image_list, split):

    crop_count = 0

    for image_path, label_path in image_list:

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read:", image_path)
            continue

        height, width = image.shape[:2]

        # Read YOLO labels
        with open(label_path, "r") as file:
            labels = file.readlines()

        for box_number, line in enumerate(labels):

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, x_center, y_center, box_width, box_height = map(
                float, values
            )

            class_id = int(class_id)

            # Convert YOLO coordinates to pixels
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            # Bounding box coordinates
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Crop defect
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Resize to 64 × 64
            crop = cv2.resize(
                crop,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            # Class folder
            class_name = CLASS_NAMES[class_id]

            output_name = (
                f"{image_path.stem}_box{box_number}.jpg"
            )

            output_path = (
                OUTPUT
                / split
                / class_name
                / output_name
            )

            cv2.imwrite(
                str(output_path),
                crop
            )

            crop_count += 1

    return crop_count


# =========================
# Run preprocessing
# =========================

print("\nProcessing training data...")
train_crops = process_images(train_images, "train")

print("Processing validation data...")
valid_crops = process_images(valid_images, "valid")

print("Processing test data...")
test_crops = process_images(test_images, "test")


# =========================
# Final information
# =========================

print("\n==============================")
print("PREPROCESSING COMPLETE")
print("==============================")

print("Training crops:", train_crops)
print("Validation crops:", valid_crops)
print("Testing crops:", test_crops)

print("\nOutput location:")
print(OUTPUT.resolve())