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
import argparse
from flwr.client import start_client, start_numpy_client
import os
from flwr.client import NumPyClient
import tensorflow as tf
from flwr_datasets import FederatedDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

client_id = 0
train_accuracies = []

# Student Model (Smaller Model)
def create_model():
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
    
    
 # Load and compile the model
model = create_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

def partition_data(client_id):
    # Partitioning the data using Dirichlet distribution to ensure non-IID
    partitioner = DirichletPartitioner(
        num_partitions=10, partition_by="label",
        alpha=0.5, min_partition_size=10, self_balancing=True
    )

    fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": partitioner})

    partition = fds.load_partition(client_id, split="train")
    print(partition[client_id])
    partition_sizes = [
        len(fds.load_partition(partition_id)) for partition_id in range(10)
    ]
    print(sorted(partition_sizes))
    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train = [np.array(image).reshape(28, 28, 1) for image in partition["train"]["image"]]
    y_train = np.array(partition["train"]["label"])

    x_test = [np.array(image).reshape(28, 28, 1) for image in partition["test"]["image"]]
    y_test = np.array(partition["test"]["label"])

    # Convert to NumPy arrays and normalize
    x_train, x_test = np.array(x_train) / 255.0, np.array(x_test) / 255.0

    return x_train, y_train, x_test, y_test



    return x_train, y_train, x_test, y_test

# Load dataset
x_train, y_train, x_test, y_test = partition_data(client_id)

# Plot training accuracy after federated learning rounds
def plot_training_accuracy():
    if train_accuracies:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', color='b', label='Training Accuracy')
        plt.title(f'Centralized FL (Student): Fashion_MNIST Dataset, Training Accuracy of Client {client_id} Over Rounds', fontsize=16)
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

# Define Flower client-here actually the federated learning magic happening
class FlowerClient(NumPyClient):
    def get_parameters(self, config):  #this function needs to exsist if on the server side we don't initialize any weight the server is actually going to pick a random client from those they are connected and call this function to initialize it's weight
        return model.get_weights()

    def fit(self, parameters, config): #parameters = the parameters are sent from server to the client for a given round, config = It is a dictionary of strings to any scholar, this is also passes from the server to the client
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=1, batch_size=64)
        hist = r.history
        print("Fit history : " ,hist)
        # Ensure history is captured and parsed correctly
        if 'accuracy' in hist:
            accuracy = hist['accuracy'][-1]
            train_accuracies.append(accuracy)  # Append the accuracy to the list
            print(f"Round training accuracy: {accuracy}")
        else:
            print("No accuracy data available in history.")
        return model.get_weights(), len(x_train), {} #return 3 things weight of the model after traning, length of the traning data (if the clients have different sizes of data we might want on the server side to aggregate them differently), a empty dictionary
 
    def evaluate(self, parameters, config): # this parameter, server has aggregated all the parameters from the client after the training, it is going to send back the aggregated weight to every client for them to evaluate this new model on their data
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
        


# starting the client with the server address
# In the client side this address is going to be my local host
start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(), grpc_max_message_length = 1024*1024*1024) 
# Call the plot function after federated learning is complete
plot_training_accuracy()

# Creating DataFrame and saving as CSV
df = pd.DataFrame({
    "Training Accuracy": train_accuracies
})

# Save DataFrame as CSV
output_path = "Centralized FL (Student) Fashion_MNIST Dataset, Training Accuracy of Client 0 Over Rounds.csv"
df.to_csv(output_path, index=False)




