import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

#teacher model (larger model)
def create_teacher_model():
    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),  # Updated input shape for CIFAR-10
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    return model


# Load and prepare CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Create and compile teacher model
teacher_model = create_teacher_model()
teacher_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

# Train the teacher model
teacher_model.fit(x_train, y_train, epochs=20, batch_size=64)

# Evaluate the teacher model
teacher_model.evaluate(x_test, y_test)

# Save the trained model
teacher_model.save("teacher_model.h5")
