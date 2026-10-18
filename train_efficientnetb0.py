from pathlib import Path
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'preprocessing' / 'output'
SAVE_DIR = ROOT / 'results' / 'sota' / 'efficientnetb0'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']
SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
tf.keras.utils.set_random_seed(SEED)


def load_data(split):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split, class_names=CLASSES, label_mode='int',
        color_mode='rgb', image_size=SIZE, batch_size=BATCH_SIZE,
        shuffle=(split == 'train'), seed=SEED
    )
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = load_data('train')
val_ds = load_data('valid')
test_ds = load_data('test')

# EfficientNetB0 from tf.keras includes its own input rescaling.
# Do not divide dataset pixels by 255 or use MobileNetV2 preprocessing.
base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_shape=(128, 128, 3)
)
base.trainable = False
inputs = tf.keras.Input(shape=(128, 128, 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
print('Total parameters:', model.count_params())


def callbacks(stage):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(SAVE_DIR / f'{stage}_best.keras'),
            monitor='val_loss', save_best_only=True
        ),
    ]


start = time.perf_counter()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('\nStage 1: train classification head')
h1 = model.fit(train_ds, validation_data=val_ds, epochs=15,
               callbacks=callbacks('stage1'))

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('\nStage 2: fine-tune last layers')
h2 = model.fit(train_ds, validation_data=val_ds, epochs=20,
               callbacks=callbacks('stage2'))
training_time = time.perf_counter() - start

# Compare stage checkpoints using validation loss only (no test-set selection).
stage1 = tf.keras.models.load_model(str(SAVE_DIR / 'stage1_best.keras'))
stage2 = tf.keras.models.load_model(str(SAVE_DIR / 'stage2_best.keras'))
v1 = stage1.evaluate(val_ds, verbose=0)[0]
v2 = stage2.evaluate(val_ds, verbose=0)[0]
selected_stage = 'Stage 1' if v1 <= v2 else 'Stage 2'
best_model = stage1 if v1 <= v2 else stage2
best_path = SAVE_DIR / 'best_model.keras'
best_model.save(str(best_path))

test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_prob = best_model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_prob, axis=1)
macro_f1 = f1_score(y_true, y_pred, average='macro')
(SAVE_DIR / 'classification_report.txt').write_text(
    classification_report(y_true, y_pred, target_names=CLASSES,
                          digits=4, zero_division=0), encoding='utf-8'
)
cm = confusion_matrix(y_true, y_pred)
np.savetxt(SAVE_DIR / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
fig, ax = plt.subplots(figsize=(10, 7))
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
    ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False
)
fig.tight_layout()
fig.savefig(SAVE_DIR / 'confusion_matrix.png', dpi=300)
plt.close(fig)

keys = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
history = {k: h1.history[k] + h2.history[k] for k in keys}
boundary = len(h1.history['loss'])
with open(SAVE_DIR / 'history.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['epoch', 'stage'] + keys)
    for i in range(len(history['loss'])):
        w.writerow([i + 1, 1 if i < boundary else 2] +
                   [history[k][i] for k in keys])
for metric in ['accuracy', 'loss']:
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(1, len(history[metric]) + 1)
    ax.plot(epochs, history[metric], label=f'Training {metric}')
    ax.plot(epochs, history[f'val_{metric}'], label=f'Validation {metric}')
    ax.axvline(boundary + 0.5, linestyle='--', color='gray',
               label='Fine-tuning starts')
    ax.set(xlabel='Epoch', ylabel=metric.title(),
           title=f'EfficientNet-B0: {metric.title()} vs. Epoch')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SAVE_DIR / f'{metric}_vs_epochs.png', dpi=300)
    plt.close(fig)

size_mb = best_path.stat().st_size / (1024 ** 2)
with open(SAVE_DIR / 'summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Model', 'Selected Stage', 'Parameters', 'Model Size (MB)',
                'Test Accuracy (%)', 'Test Loss', 'Macro F1', 'Training Time (s)'])
    w.writerow(['EfficientNet-B0', selected_stage, best_model.count_params(),
                round(size_mb, 3), round(test_acc * 100, 2),
                round(test_loss, 4), round(macro_f1, 4),
                round(training_time, 2)])

print('\nEFFICIENTNET-B0 RESULTS')
print('Selected:', selected_stage)
print(f'Test accuracy: {test_acc * 100:.2f}%')
print(f'Total parameters: {best_model.count_params():,}')
print(f'Model size: {size_mb:.3f} MB')
print(f'Macro F1: {macro_f1:.4f}')
print(f'Training time: {training_time:.2f} seconds')
print('Saved results to:', SAVE_DIR)
