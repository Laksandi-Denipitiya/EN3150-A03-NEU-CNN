import tensorflow as tf
import csv
import time
from pathlib import Path

from model_b import build_model_b


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20

OUTPUT_DIR = Path("results/optimizer_experiment")
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
# OPTIMIZERS TO COMPARE
# ============================================================

OPTIMIZERS = {
    "Adam": tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    "SGD": tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.0
    ),

    "SGD_Momentum": tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.9
    )
}


# ============================================================
# DATASET LOADING FUNCTION
# ============================================================

def load_datasets():

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

    # --------------------------------------------------------
    # Normalize to [0, 1]
    # --------------------------------------------------------

    normalization_layer = tf.keras.layers.Rescaling(
        1.0 / 255
    )

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

    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    valid_ds = valid_ds.prefetch(
        tf.data.AUTOTUNE
    )

    test_ds = test_ds.prefetch(
        tf.data.AUTOTUNE
    )

    return train_ds, valid_ds, test_ds


# ============================================================
# EPOCH TIMER
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
            f"    Epoch {epoch + 1:02d} time: "
            f"{elapsed:.3f} s"
        )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(optimizer_name):

    print("\n" + "=" * 70)
    print(f"RUNNING OPTIMIZER: {optimizer_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Reset random seed for fair comparison
    # --------------------------------------------------------

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_ds, valid_ds, test_ds = load_datasets()

    # --------------------------------------------------------
    # Build a fresh Model B
    # --------------------------------------------------------

    model = build_model_b()

    # --------------------------------------------------------
    # Create a fresh optimizer
    # --------------------------------------------------------

    if optimizer_name == "Adam":

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001
        )

    elif optimizer_name == "SGD":

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.0
        )

    elif optimizer_name == "SGD_Momentum":

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------------
    # Timing callback
    # --------------------------------------------------------

    timer = EpochTimer()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...\n")

    total_start = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[timer]
    )

    total_time = time.perf_counter() - total_start

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\nEvaluating on test set...")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR /
        f"{optimizer_name}_history.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "loss",
            "accuracy",
            "val_loss",
            "val_accuracy",
            "epoch_time_seconds"
        ])

        for i in range(EPOCHS):

            writer.writerow([
                i + 1,
                history.history["loss"][i],
                history.history["accuracy"][i],
                history.history["val_loss"][i],
                history.history["val_accuracy"][i],
                timer.epoch_times[i]
            ])

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        OUTPUT_DIR /
        f"{optimizer_name}_model.keras"
    )

    model.save(model_path)

    model_size_kb = (
        model_path.stat().st_size / 1024
    )

    # --------------------------------------------------------
    # Find best validation accuracy
    # --------------------------------------------------------

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_val_epoch = (
        history.history["val_accuracy"].index(
            best_val_accuracy
        ) + 1
    )

    average_epoch_time = (
        sum(timer.epoch_times)
        / len(timer.epoch_times)
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print(f"{optimizer_name} RESULTS")
    print("-" * 60)

    print(
        f"Best validation accuracy : "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best validation epoch    : "
        f"{best_val_epoch}"
    )

    print(
        f"Final validation accuracy: "
        f"{history.history['val_accuracy'][-1] * 100:.2f}%"
    )

    print(
        f"Final validation loss    : "
        f"{history.history['val_loss'][-1]:.4f}"
    )

    print(
        f"Test accuracy            : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test loss                : "
        f"{test_loss:.4f}"
    )

    print(
        f"Average epoch time       : "
        f"{average_epoch_time:.3f} s"
    )

    print(
        f"Total training time      : "
        f"{total_time:.2f} s"
    )

    print(
        f"Model size               : "
        f"{model_size_kb:.2f} KB"
    )

    print("-" * 60)

    return {
        "optimizer": optimizer_name,
        "best_val_accuracy": best_val_accuracy,
        "best_val_epoch": best_val_epoch,
        "final_val_accuracy":
            history.history["val_accuracy"][-1],
        "final_val_loss":
            history.history["val_loss"][-1],
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "average_epoch_time": average_epoch_time,
        "total_training_time": total_time,
        "model_size_kb": model_size_kb
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = []

    for optimizer_name in [
        "Adam",
        "SGD",
        "SGD_Momentum"
    ]:

        result = run_experiment(
            optimizer_name
        )

        results.append(result)

    # ========================================================
    # Save summary
    # ========================================================

    summary_path = (
        OUTPUT_DIR /
        "optimizer_comparison.csv"
    )

    with open(
        summary_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    # ========================================================
    # Final comparison
    # ========================================================

    print("\n\n")
    print("=" * 90)
    print("FINAL OPTIMIZER COMPARISON")
    print("=" * 90)

    print(
        f"{'Optimizer':<20}"
        f"{'Best Val':<15}"
        f"{'Test Acc.':<15}"
        f"{'Epoch Time':<15}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['optimizer']:<20}"
            f"{result['best_val_accuracy'] * 100:>8.2f}%       "
            f"{result['test_accuracy'] * 100:>8.2f}%       "
            f"{result['average_epoch_time']:>8.3f} s"
        )

    print("=" * 90)

    print(
        f"\nComparison saved to:\n"
        f"{summary_path}"
    )
