import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import shap
from keras import layers
from calculate_feature_importance import calculate_feature_importance

# Download the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Reshape and normalize data (CIFAR-10 is already in the correct shape, so just normalize)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

class Distiller(keras.Model):
    def __init__(self, student, teacher, feature_importance):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.feature_importance = feature_importance

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            loss = tf.reduce_mean(loss * self.feature_importance)

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        x, y = data
        y_prediction = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)
        self.compiled_metrics.update_state(y, y_prediction)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


# Student Model (Smaller Model, updated input shape for CIFAR-10)
def create_student_model():
    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),  # Updated input shape for CIFAR-10
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


# Aggregate Shapley values (e.g., average absolute values)
feature_importance = calculate_feature_importance(teacher_model, x_train, y_test)

# Initialize and compile distiller
distiller = Distiller(student=student_model, teacher=teacher_model, feature_importance=feature_importance)
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
pyplot.title('Centralized KD+SHAP: Training and Validation Accuracy Over Epochs on CIFAR-10 Dataset')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.plot(history.history['sparse_categorical_accuracy'], label='train')
pyplot.plot(history.history['val_sparse_categorical_accuracy'], label='test')
pyplot.savefig('Centralized KD+SHAP:  Training and Validation Accuracy Over Epochs on CIFAR-10 Dataset.png')
pyplot.legend()
pyplot.show()


# Plot training and validation loss using seaborn
pyplot.figure(figsize=(10, 6))
pyplot.title('Centralized KD+SHAP: Training and Validation Loss Over Epochs on CIFAR-10 Dataset')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.savefig('Centralized KD+SHAP: Training and Validation Loss Over Epochs on CIFAR-10 Dataset.png')
pyplot.legend()
pyplot.show()

# Creating DataFrame and saving as CSV
df = pd.DataFrame({
    "Training Accuracy": history.history['sparse_categorical_accuracy'],
    "Validation Accuracy": history.history['val_sparse_categorical_accuracy']
})

# Save DataFrame as CSV
output_path = "Centralized KD+SHAP Training and Validation Accuracy Over Epochs on CIFAR-10 Dataset.csv"
df.to_csv(output_path, index=False)

# Creating DataFrame and saving as CSV
df1 = pd.DataFrame({
    "Training Accuracy": history.history['loss'],
    "Validation Accuracy": history.history['val_loss']
})

# Save DataFrame as CSV
output_path = "Centralized KD_SHAP Training and Validation Loss Over Epochs on CIFAR-10 Dataset.csv"
df1.to_csv(output_path, index=False)