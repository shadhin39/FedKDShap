import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from data_loader import partition_data

client_id = 2
train_accuracies = []
# Auxiliary methods
# def getDist(y):
#     ax = sns.countplot(y)
#     ax.set(title="Count of data classes")
#     plt.show()

# def getData(dist, x, y):
#     dx = []
#     dy = []
#     counts = [0 for i in range(10)]
#     for i in range(len(x)):
#         if counts[y[i]] < dist[y[i]]:
#             dx.append(x[i])
#             dy.append(y[i])
#             counts[y[i]] += 1
        
#     return np.array(dx), np.array(dy)

# Load and compile Keras model
# Create the teacher

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

        # Compute gradients
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


# Load and compile the student model and teacher model
student_model = create_student_model()



# Load dataset
x_train, y_train, x_test, y_test = partition_data(client_id)

# Train and evaluate teacher on data.
# Load the pre-trained teacher model
teacher_model = keras.models.load_model("teacher_model.h5")



# Load dataset
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# # Normalize data
# x_train = x_train.astype("float32") / 255.0
# x_train = np.reshape(x_train, (-1, 28, 28, 1))

# x_test = x_test.astype("float32") / 255.0
# x_test = np.reshape(x_test, (-1, 28, 28, 1))
# dist = [0, 10, 10, 10, 4000, 3000, 4000, 5000, 10, 4500]
# # getDist(y_train)
# x_train, y_train = getData(dist, x_train, y_train)

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


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return distiller.get_weights()  # Return the student model's weights

    def fit(self, parameters, config):
        distiller.set_weights(parameters)
        # Distill teacher to student
        r = distiller.fit(x_train, y_train, epochs=20, batch_size=64)
        hist = r.history
        print("Fit history : " ,hist)
        # Ensure history is captured and parsed correctly
        if 'sparse_categorical_accuracy' in hist:
            accuracy = hist['sparse_categorical_accuracy'][-1]
            train_accuracies.append(accuracy)  # Append the accuracy to the list
            print(f"Round training accuracy: {accuracy}")
        else:
            print("No accuracy data available in history.")
            
        return distiller.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        distiller.set_weights(parameters)  # Update student model with the latest global model weights
        results = distiller.evaluate(x_test, y_test)
        print(results)
        accuracy = results[2]
        student_loss = results[3]
        print("Evaluated Accuracy: ", accuracy)
        print("Evaluated Student Loss: ", student_loss)
        return student_loss, len(x_test), {"accuracy": accuracy}





# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", 
    client=FlowerClient(), 
    grpc_max_message_length=1024*1024*1024
)

# Plot training accuracy after federated learning rounds
def plot_training_accuracy():
    if train_accuracies:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', color='b', label='Training Accuracy')
        plt.title(f'FL+KD: MNIST Dataset, Training Accuracy of Client {client_id} Over Rounds', fontsize=16)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.ylim(min(train_accuracies) - 0.01, 1.0)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc="lower right", fontsize=12)
        plt.show()
    else:
        print("No training accuracy data available to plot.")

# Call the plot function after federated learning is complete
plot_training_accuracy()
# Creating DataFrame and saving as CSV
df = pd.DataFrame({
    "Training Accuracy": train_accuracies
})

# Save DataFrame as CSV
output_path = "FL_KD MNIST Dataset, Training Accuracy of Client 2 Over Rounds.csv"
df.to_csv(output_path, index=False)