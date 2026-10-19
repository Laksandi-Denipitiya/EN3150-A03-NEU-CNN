import os
import time
import csv
import tensorflow as tf

from model_a import build_model_a
from model_b import build_model_b


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

TRAIN_DIR = "preprocessing/output/train"
VAL_DIR = "preprocessing/output/valid"
TEST_DIR = "preprocessing/output/test"

RESULTS_DIR = "results/final_custom"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches"
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(directory, shuffle, seed=None):

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=seed
    )

    # Normalize pixels to [0, 1]
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================
# CALLBACK FOR EPOCH TIMING
# ============================================================

class EpochTimer(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self.start_time
        self.epoch_times.append(elapsed)

        print(
            f"Epoch {epoch + 1:02d} completed in "
            f"{elapsed:.3f} s"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(history, epoch_times, output_file):

    keys = [
        "epoch",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy",
        "epoch_time"
    ]

    with open(output_file, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(keys)

        for i in range(len(history.history["loss"])):

            writer.writerow([
                i + 1,
                history.history["loss"][i],
                history.history["accuracy"][i],
                history.history["val_loss"][i],
                history.history["val_accuracy"][i],
                epoch_times[i]
            ])


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(model_name, build_function):

    print("\n")
    print("=" * 80)
    print(f"TRAINING {model_name}")
    print("=" * 80)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_ds = load_dataset(
        TRAIN_DIR,
        shuffle=True,
        seed=SEED
    )

    val_ds = load_dataset(
        VAL_DIR,
        shuffle=False
    )

    test_ds = load_dataset(
        TEST_DIR,
        shuffle=False
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding model...")

    model = build_function()

    model.summary()

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    timer = EpochTimer()

    model_dir = os.path.join(
        RESULTS_DIR,
        model_name.lower().replace(" ", "_")
    )

    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        model_dir,
        f"best_{model_name.lower().replace(' ', '_')}.keras"
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=0
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...\n")

    total_start = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            timer,
            checkpoint
        ],
        verbose=1
    )

    total_time = time.perf_counter() - total_start

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\nEvaluating on test set...")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )

    # --------------------------------------------------------
    # Model statistics
    # --------------------------------------------------------

    trainable_params = model.count_params()

    avg_epoch_time = sum(timer.epoch_times) / len(
        timer.epoch_times
    )

    # Save final model
    final_model_path = os.path.join(
        model_dir,
        f"final_{model_name.lower().replace(' ', '_')}.keras"
    )

    model.save(final_model_path)

    # Actual file size
    model_size_kb = (
        os.path.getsize(final_model_path) / 1024
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = os.path.join(
        model_dir,
        "history.csv"
    )

    save_history(
        history,
        timer.epoch_times,
        history_path
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(
        model_dir,
        "training_summary.txt"
    )

    best_epoch = (
        history.history["val_accuracy"].index(
            max(history.history["val_accuracy"])
        ) + 1
    )

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    with open(summary_path, "w") as f:

        f.write(f"{model_name} FINAL TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Trainable parameters : {trainable_params:,}\n"
        )

        f.write(
            f"Model size on disk   : {model_size_kb:.2f} KB\n"
        )

        f.write(
            f"Epochs trained       : {EPOCHS}\n"
        )

        f.write(
            f"Total training time  : {total_time:.2f} s\n"
        )

        f.write(
            f"Average epoch time   : {avg_epoch_time:.3f} s\n"
        )

        f.write(
            f"Best validation acc. : {best_val_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Best validation epoch: {best_epoch}\n"
        )

        f.write(
            f"Test accuracy        : {test_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Test loss            : {test_loss:.4f}\n"
        )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n")
    print("-" * 60)
    print(f"{model_name} FINAL RESULTS")
    print("-" * 60)

    print(
        f"Trainable parameters : {trainable_params:,}"
    )

    print(
        f"Model size on disk   : {model_size_kb:.2f} KB"
    )

    print(
        f"Best validation acc. : {best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best validation epoch: {best_epoch}"
    )

    print(
        f"Test accuracy        : {test_accuracy * 100:.2f}%"
    )

    print(
        f"Test loss            : {test_loss:.4f}"
    )

    print(
        f"Average epoch time   : {avg_epoch_time:.3f} s"
    )

    print(
        f"Total training time  : {total_time:.2f} s"
    )

    print("-" * 60)

    return {
        "model": model_name,
        "parameters": trainable_params,
        "model_size_kb": model_size_kb,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "avg_epoch_time": avg_epoch_time,
        "total_time": total_time
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("FINAL CUSTOM CNN TRAINING")
    print("=" * 80)
    print(f"Epochs       : {EPOCHS}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Learning rate: 0.001")
    print(f"Optimizer    : Adam")
    print(f"Random seed  : {SEED}")
    print("=" * 80)

    # --------------------------------------------------------
    # Train Model A
    # --------------------------------------------------------

    result_a = train_model(
        "Model_A",
        build_model_a
    )

    # --------------------------------------------------------
    # Train Model B
    # --------------------------------------------------------

    result_b = train_model(
        "Model_B",
        build_model_b
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison_path = os.path.join(
        RESULTS_DIR,
        "custom_model_comparison.csv"
    )

    with open(comparison_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Model",
            "Trainable Parameters",
            "Model Size (KB)",
            "Best Validation Accuracy (%)",
            "Best Validation Epoch",
            "Test Accuracy (%)",
            "Test Loss",
            "Average Epoch Time (s)",
            "Total Training Time (s)"
        ])

        for result in [result_a, result_b]:

            writer.writerow([
                result["model"],
                result["parameters"],
                f"{result['model_size_kb']:.2f}",
                f"{result['best_val_accuracy'] * 100:.2f}",
                result["best_epoch"],
                f"{result['test_accuracy'] * 100:.2f}",
                f"{result['test_loss']:.4f}",
                f"{result['avg_epoch_time']:.3f}",
                f"{result['total_time']:.2f}"
            ])

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("FINAL CUSTOM MODEL COMPARISON")
    print("=" * 90)

    print(
        f"{'Model':<12}"
        f"{'Parameters':>15}"
        f"{'Size KB':>12}"
        f"{'Best Val':>12}"
        f"{'Test Acc.':>12}"
        f"{'Epoch Time':>15}"
    )

    print("-" * 90)

    for result in [result_a, result_b]:

        print(
            f"{result['model']:<12}"
            f"{result['parameters']:>15,}"
            f"{result['model_size_kb']:>12.2f}"
            f"{result['best_val_accuracy'] * 100:>11.2f}%"
            f"{result['test_accuracy'] * 100:>11.2f}%"
            f"{result['avg_epoch_time']:>14.3f} s"
        )

    print("=" * 90)

    print("\nResults saved to:")
    print(RESULTS_DIR)
    print("\nTraining complete.")