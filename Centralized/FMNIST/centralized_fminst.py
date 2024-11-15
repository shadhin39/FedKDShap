import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from keras import layers

# Download the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Reshape and normalize data
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255

# Create the teacher model (larger model)
def create_model():
  model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
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

model = create_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

# Train the teacher model with validation split
history = model.fit(x_train, y_train, epochs=20, validation_split=0.2)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Centralized Learning: Training and Validation Accuracy over Epochs on FMNIST Dataset')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Creating DataFrame and saving as CSV
df = pd.DataFrame({
    "Training Accuracy": history.history['accuracy'],
    "Validation Accuracy": history.history['val_accuracy']
})

# Save DataFrame as CSV
output_path = "Centralized Learning Training and Validation Accuracy over Epochs on FMNIST Dataset.csv"
df.to_csv(output_path, index=False)