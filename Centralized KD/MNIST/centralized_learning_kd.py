import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from keras import layers
import matplotlib.pyplot as pyplot
# Download the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape and normalize data
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255

class Distiller(keras.Model):
  def __init__(self, student, teacher):
      super().__init__()
      self.teacher = teacher
      self.student = student

  def compile(
      self,
      optimizer,
      metrics,
      student_loss_fn,
      distillation_loss_fn,
      alpha=0.1,
      temperature=3,
  ):
      """ Configure the distiller.

      Args:
          optimizer: Keras optimizer for the student weights
          metrics: Keras metrics for evaluation
          student_loss_fn: Loss function of difference between student
              predictions and ground-truth
          distillation_loss_fn: Loss function of difference between soft
              student predictions and soft teacher predictions
          alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
          temperature: Temperature for softening probability distributions.
              Larger temperature gives softer distributions.
      """
      super().compile(optimizer=optimizer, metrics=metrics)
      self.student_loss_fn = student_loss_fn
      self.distillation_loss_fn = distillation_loss_fn
      self.alpha = alpha
      self.temperature = temperature

  def train_step(self, data):
      # Unpack data
      x, y = data

      # Forward pass of teacher
      teacher_predictions = self.teacher(x, training=False)

      with tf.GradientTape() as tape:
          # Forward pass of student
          student_predictions = self.student(x, training=True)

          # Compute losses
          student_loss = self.student_loss_fn(y, student_predictions)

          # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
          # The magnitudes of the gradients produced by the soft targets scale
          # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
          distillation_loss = (
              self.distillation_loss_fn(
                  tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                  tf.nn.softmax(student_predictions / self.temperature, axis=1),
              )
              * self.temperature**2
          )

          loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

      # compute gradients
      trainable_vars = self.student.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)

      # Update weights
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))

      # Update the metrics configured in `compile()`.
      self.compiled_metrics.update_state(y, student_predictions)

      # Return a dict of performance
      results = {m.name: m.result() for m in self.metrics}
      results.update(
          {"student_loss": student_loss, "distillation_loss": distillation_loss}
      )
      return results

  def test_step(self, data):
      # Unpack the data
      x, y = data

      # Compute predictions
      y_prediction = self.student(x, training=False)

      # Calculate the loss
      student_loss = self.student_loss_fn(y, y_prediction)

      # Update the metrics.
      self.compiled_metrics.update_state(y, y_prediction)

      # Return a dict of performance
      results = {m.name: m.result() for m in self.metrics}
      results.update({"student_loss": student_loss})
      return results



# Student Model (Smaller Model)
def create_student_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    return model

# Load and compile the student model
student_model = create_student_model()
# Load the pre-trained teacher model
teacher_model = keras.models.load_model("teacher_model.h5")


# Initialize and compile distiller
distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student
history = distiller.fit(x_train, y_train,validation_data=(x_test,y_test), epochs=20, batch_size=64)

# Evaluate student on test dataset
distiller.evaluate(x_test, y_test)


# Plot training and validation loss using seaborn
pyplot.figure(figsize=(10, 6))
pyplot.title('Centralized KD: Training and Validation Accuracy Over Epochs on MNIST Dataset')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.plot(history.history['sparse_categorical_accuracy'], label='train')
pyplot.plot(history.history['val_sparse_categorical_accuracy'], label='test')
pyplot.savefig('Centralized KD+SHAP: Training and Validation Accuracy Over Epochs on MNIST Dataset.png')
pyplot.legend()
pyplot.show()


# Plot training and validation loss using seaborn
pyplot.figure(figsize=(10, 6))
pyplot.title('Centralized KD: Training and Validation Loss Over Epochs on MNIST Dataset')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.savefig('Centralized KD+SHAP: Training and Validation Loss Over Epochs on MNIST Dataset.png')
pyplot.legend()
pyplot.show()

# Creating DataFrame and saving as CSV
df = pd.DataFrame({
    "Training Accuracy": history.history['sparse_categorical_accuracy'],
    "Validation Accuracy": history.history['val_sparse_categorical_accuracy']
})

# Save DataFrame as CSV
output_path = "Centralized KD Training and Validation Accuracy Over Epochs on MNIST Dataset.csv"
df.to_csv(output_path, index=False)

# Creating DataFrame and saving as CSV
df1 = pd.DataFrame({
    "Training Accuracy": history.history['loss'],
    "Validation Accuracy": history.history['val_loss']
})

# Save DataFrame as CSV
output_path = "Centralized KD Training and Validation Loss Over Epochs on MNIST Dataset.csv"
df1.to_csv(output_path, index=False)