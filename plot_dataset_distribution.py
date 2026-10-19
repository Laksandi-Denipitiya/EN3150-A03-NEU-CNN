
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Locate dataset relative to this script
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "preprocessing" / "output"

# Dataset splits
splits = ["train", "valid", "test"]

# NEU-CLS defect classes
classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]

# Count images
counts = {}

for split in splits:
    counts[split] = []

    for class_name in classes:
        folder = DATA_DIR / split / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {folder}"
            )

        images = [
            file for file in folder.iterdir()
            if file.suffix.lower() in
            [".jpg", ".jpeg", ".png", ".bmp"]
        ]

        counts[split].append(len(images))

# Display counts
print("\nNEU-CLS Dataset Distribution")
print("-" * 55)
print(f"{'Class':<20} {'Train':>8} {'Valid':>8} {'Test':>8}")

for i, class_name in enumerate(classes):
    print(
        f"{class_name:<20}"
        f"{counts['train'][i]:>8}"
        f"{counts['valid'][i]:>8}"
        f"{counts['test'][i]:>8}"
    )

print("-" * 55)

for split in splits:
    print(f"{split}: {sum(counts[split])} images")

# Create grouped bar chart
x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(
    x - width, counts["train"], width,
    label="Training", color="#4472C4"
)

bars2 = ax.bar(
    x, counts["valid"], width,
    label="Validation", color="#ED7D31"
)

bars3 = ax.bar(
    x + width, counts["test"], width,
    label="Testing", color="#70AD47"
)

# Add count labels
for bars in [bars1, bars2, bars3]:
    ax.bar_label(bars, padding=3, fontsize=9)

ax.set_xlabel("Steel Surface Defect Class", fontsize=12)
ax.set_ylabel("Number of Images", fontsize=12)
ax.set_title(
    "NEU-CLS Dataset Distribution",
    fontsize=15,
    fontweight="bold"
)

ax.set_xticks(x)
ax.set_xticklabels(
    ["Cr", "In", "Pa", "Ps", "Rs", "Sc"]
)

ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()

# Save graph
SAVE_PATH = BASE_DIR / "dataset_distribution.png"

plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nGraph saved to: {SAVE_PATH}")
