import tensorflow as tf

IMAGE_SIZE = 64
NUM_CLASSES = 6


def build_model_b():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(IMAGE_SIZE, IMAGE_SIZE, 1)
        ),

        # Depthwise-Separable Convolution Block 1
        tf.keras.layers.SeparableConv2D(
            16,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Depthwise-Separable Convolution Block 2
        tf.keras.layers.SeparableConv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Depthwise-Separable Convolution Block 3
        tf.keras.layers.SeparableConv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        # Reduce spatial dimensions without a large
        # fully-connected parameter count
        tf.keras.layers.GlobalAveragePooling2D(),

        # Lightweight classifier
        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),
        tf.keras.layers.Dropout(0.3),

        # Six defect classes
        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    return model


if __name__ == "__main__":
    model = build_model_b()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
