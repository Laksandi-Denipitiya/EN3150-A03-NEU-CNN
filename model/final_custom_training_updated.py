"""Train and evaluate the two NEU-CLS defect-crop classifiers.

Run from any working directory:
    python model/final_custom_training_updated.py

Requires: tensorflow, numpy, matplotlib, scikit-learn.
"""
from pathlib import Path
import csv
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from model_a import build_model_a
from model_b import build_model_b

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / 'preprocessing' / 'output'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'final_custom_v2'
IMAGE_SIZE = 64
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 5
SEED = 42
PARAMETER_LIMIT = 100_000
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']


class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self.started)


def load_dataset(split, shuffle):
    directory = DATA_ROOT / split
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing dataset directory: {directory}')
    ds = tf.keras.utils.image_dataset_from_directory(
        str(directory), labels='inferred', label_mode='int',
        class_names=CLASS_NAMES, color_mode='grayscale',
        image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
        shuffle=shuffle, seed=SEED if shuffle else None,
    )
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def plot_history(history, out_dir, selected_epoch):
    epochs = np.arange(1, len(history.history['loss']) + 1)
    for metric, ylabel, filename in [
        ('accuracy', 'Accuracy', 'accuracy_vs_epochs.png'),
        ('loss', 'Cross-entropy loss', 'loss_vs_epochs.png'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history.history[metric], marker='o', markersize=3, label='Training')
        ax.plot(epochs, history.history['val_' + metric], marker='o', markersize=3, label='Validation')
        ax.axvline(selected_epoch, linestyle='--', color='gray', label=f'Selected epoch {selected_epoch}')
        ax.set(xlabel='Epoch', ylabel=ylabel, title=f'{ylabel} vs. epochs')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=300)
        plt.close(fig)


def train_one(name, builder, train_ds, val_ds, test_ds):
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = builder()
    parameters = int(model.count_params())
    model.summary()
    if parameters > PARAMETER_LIMIT:
        raise ValueError(f'{name} has {parameters:,} parameters, exceeding the {PARAMETER_LIMIT:,} limit')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint_path = out_dir / 'best_model.keras'
    timer = EpochTimer()
    callbacks = [
        timer,
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor='val_loss', mode='min', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=PATIENCE,
            restore_best_weights=True),
    ]
    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=MAX_EPOCHS, callbacks=callbacks, verbose=1)
    training_time = time.perf_counter() - start

    best_epoch = int(np.argmin(history.history['val_loss']) + 1)
    plot_history(history, out_dir, best_epoch)
    with (out_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'epoch_time_seconds'])
        for i in range(len(history.history['loss'])):
            writer.writerow([i + 1, history.history['loss'][i], history.history['accuracy'][i],
                             history.history['val_loss'][i], history.history['val_accuracy'][i],
                             timer.epoch_times[i]])

    # Always evaluate the validation-selected checkpoint, never the final epoch.
    best_model = tf.keras.models.load_model(str(checkpoint_path))
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=0)
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probabilities = best_model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                                   target_names=CLASS_NAMES, digits=4, zero_division=0,
                                   output_dict=True)
    with (out_dir / 'classification_report.txt').open('w', encoding='utf-8') as f:
        f.write(classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                                      target_names=CLASS_NAMES, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap='Blues',
                                                                 xticks_rotation=40, colorbar=False)
    ax.set_title(f'{name}: test confusion matrix')
    fig.tight_layout()
    fig.savefig(out_dir / 'confusion_matrix.png', dpi=300)
    plt.close(fig)
    np.savetxt(out_dir / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d')

    result = {
        'model': name,
        'parameters': parameters,
        'epochs_trained': len(history.history['loss']),
        'best_epoch_by_val_loss': best_epoch,
        'best_val_loss': float(min(history.history['val_loss'])),
        'val_accuracy_at_best_epoch': float(history.history['val_accuracy'][best_epoch - 1]),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'macro_precision': float(report['macro avg']['precision']),
        'macro_recall': float(report['macro avg']['recall']),
        'macro_f1': float(report['macro avg']['f1-score']),
        'training_time_seconds': float(training_time),
        'avg_epoch_time_seconds': float(np.mean(timer.epoch_times)),
        'model_size_kb': checkpoint_path.stat().st_size / 1024,
    }
    (out_dir / 'training_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'\n{name}: best epoch={best_epoch}, test accuracy={test_accuracy:.2%}, '
          f'macro F1={result["macro_f1"]:.3f}, params={parameters:,}')
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    train_ds = load_dataset('train', shuffle=True)
    val_ds = load_dataset('valid', shuffle=False)
    test_ds = load_dataset('test', shuffle=False)
    results = [
        train_one('model_a', build_model_a, train_ds, val_ds, test_ds),
        train_one('model_b', build_model_b, train_ds, val_ds, test_ds),
    ]
    with (RESULTS_DIR / 'model_comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print('\nSaved all results to:', RESULTS_DIR)


if __name__ == '__main__':
    main()
