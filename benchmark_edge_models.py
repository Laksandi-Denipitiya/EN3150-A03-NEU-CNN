"""Compare batch-1 forward-pass latency on the same machine.
Run from the EN3150-A03-NEU-CNN project root.
Measures TensorFlow/Keras inference, not deployment-device latency or FLOPs.
"""
from pathlib import Path
import csv
import statistics
import time
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'sota' / 'inference_benchmark.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)

MODELS = {
    'Custom Model B': ROOT / 'results' / 'final_custom_v2' / 'model_b' / 'best_model.keras',
    'MobileNetV2': ROOT / 'results' / 'sota' / 'mobilenetv2' / 'best_model.keras',
    'EfficientNet-B0': ROOT / 'results' / 'sota' / 'efficientnetb0' / 'best_model.keras',
}
WARMUP = 20
REPEATS = 100

print('TensorFlow:', tf.__version__)
print('CPU:', tf.config.list_physical_devices('CPU'))
print('GPU:', tf.config.list_physical_devices('GPU'))
print('Batch size: 1; warmup:', WARMUP, '; measured passes:', REPEATS)

rows = []
for name, path in MODELS.items():
    if not path.is_file():
        print(f'NOT FOUND: {name}: {path}')
        continue
    print(f'\nLoading {name} ...')
    model = tf.keras.models.load_model(path, compile=False)
    shape = model.input_shape
    if isinstance(shape, list):
        raise ValueError(f'Expected one input for {name}: {shape}')
    if any(dim is None for dim in shape[1:]):
        raise ValueError(f'Unexpected dynamic input dimensions: {shape}')
    # Values do not affect shape-based forward-pass timing; each model's
    # in-graph preprocessing, if any, remains included in the measurement.
    x = tf.random.uniform((1, *shape[1:]), minval=0, maxval=255, dtype=tf.float32)

    @tf.function
    def forward(batch):
        return model(batch, training=False)

    for _ in range(WARMUP):
        forward(x).numpy()  # synchronizes device before the next iteration

    timings_ms = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        forward(x).numpy()
        timings_ms.append((time.perf_counter() - start) * 1000)

    row = {
        'model': name,
        'parameters': model.count_params(),
        'keras_file_mb': round(path.stat().st_size / 1024**2, 3),
        'input_shape': str(shape),
        'batch_size': 1,
        'warmup_runs': WARMUP,
        'timed_runs': REPEATS,
        'mean_latency_ms': round(statistics.mean(timings_ms), 3),
        'median_latency_ms': round(statistics.median(timings_ms), 3),
        'p95_latency_ms': round(sorted(timings_ms)[int(0.95 * (REPEATS - 1))], 3),
    }
    rows.append(row)
    print(row)

if rows:
    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('\nSaved:', OUT)
else:
    print('\nNo models found. Check model paths at the top of this script.')
