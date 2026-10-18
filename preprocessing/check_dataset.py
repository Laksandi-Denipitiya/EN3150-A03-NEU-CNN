from pathlib import Path

OUTPUT = Path("preprocessing/output")

splits = ["train", "valid", "test"]

for split in splits:
    print(f"\n{split.upper()}")
    print("-" * 30)

    total = 0

    for class_folder in sorted((OUTPUT / split).iterdir()):
        if class_folder.is_dir():
            count = len(list(class_folder.glob("*.jpg")))
            print(f"{class_folder.name:20s}: {count}")
            total += count

    print(f"{'TOTAL':20s}: {total}")