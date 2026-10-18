
import tensorflow as tf


IMAGE_SIZE = 64
NUM_CLASSES = 6

def build_model_a():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 1)),

        tf.keras.layers.Conv2D(
            16, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Additional pooling to reduce parameters
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            NUM_CLASSES, activation="softmax"
        )
    ])
    return model

if __name__ == "__main__":
    model = build_model_a()
    model.summary()