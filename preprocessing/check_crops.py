import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

OUTPUT = Path("preprocessing/output")

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(9, 6))

for ax, class_name in zip(axes.ravel(), classes):

    class_folder = OUTPUT / "train" / class_name
    images = list(class_folder.glob("*.jpg"))

    image_path = random.choice(images)

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image)
    ax.set_title(class_name)
    ax.axis("off")

plt.tight_layout()
plt.show()