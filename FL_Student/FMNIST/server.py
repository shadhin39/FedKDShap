import flwr as fl
import sys
import numpy as np
from typing import List, Tuple
from flwr.common import Metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Lists to store accuracies and losses over rounds
accuraciess = []
losses = []

# Define custom aggregation functions
def aggregate_fit_metrics(metrics):
    aggregated_metrics = {}
    for metric_tuple in metrics:
        for key, value in metric_tuple[1].items():  # metric_tuple[1] is the actual metrics dictionary
            if key not in aggregated_metrics:
                aggregated_metrics[key] = []
            aggregated_metrics[key].append(value)
    aggregated_metrics = {key: np.mean(values) for key, values in aggregated_metrics.items()}
    return aggregated_metrics

def aggregate_evaluate_metrics(metrics):
    aggregated_metrics = {}
    for metric_tuple in metrics:
        for key, value in metric_tuple[1].items():  # metric_tuple[1] is the actual metrics dictionary
            if key not in aggregated_metrics:
                aggregated_metrics[key] = []
            aggregated_metrics[key].append(value)
    aggregated_metrics = {key: np.mean(values) for key, values in aggregated_metrics.items()}  
    return aggregated_metrics

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {aggregated_accuracy}")
        print(f"Round {rnd} loss aggregated from client results: {aggregated_loss}")
        accuraciess.append(aggregated_accuracy)
        losses.append(aggregated_loss)
        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}



# Create strategy and run server
strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    fit_metrics_aggregation_fn=aggregate_fit_metrics,
    evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
)

# Start Flower server for five rounds of federated learning
fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=10) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)



# Plot evaluation accuracies using Seaborn
def plot_accuracies():
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        x=range(1, len(accuraciess) + 1),
        y=accuraciess,
        marker="o",
        markersize=8,
        lw=2,
        color="b",
        label="Accuracy"
    )

    plt.fill_between(
        range(1, len(accuraciess) + 1), 
        [acc - 0.01 for acc in accuraciess], 
        [acc + 0.01 for acc in accuraciess], 
        color="blue", 
        alpha=0.1
    )

    plt.title("Global Model Evaluation Accuracies for Centralized FL (Student) Over Rounds on FMNIST Dataset", fontsize=16, fontweight='bold')
    plt.xlabel("Federated Learning Round", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(min(accuraciess) - 0.01, 1.0)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

# Plot evaluation losses using Seaborn
def plot_losses():
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        x=range(1, len(losses) + 1),
        y=losses,
        marker="o",
        markersize=8,
        lw=2,
        color="r",  # Customize loss line color
        label="Loss"
    )

    plt.fill_between(
        range(1, len(losses) + 1), 
        [loss - 0.01 for loss in losses], 
        [loss + 0.01 for loss in losses], 
        color="red", 
        alpha=0.1
    )

    plt.title("Global Model Evaluation Losses for Centralized FL (Student) Over Rounds on FMINST", fontsize=16, fontweight='bold')
    plt.xlabel("Federated Learning Round", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.ylim(min(losses) - 0.01, max(losses) + 0.01)  # Adjust limits to reflect the range of losses
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

# Call the plotting functions after federated learning is complete
if accuraciess and losses:  # Ensure there is data to plot
    plot_accuracies()
    plot_losses()
else:
    print("No data to plot. Make sure the client is properly evaluating the model.")
    
# Creating DataFrame and saving as CSV
df = pd.DataFrame({
    "Training Accuracy": accuraciess,
    "Loss": losses
})

# Save DataFrame as CSV
output_path = "Global Model Evaluation Accuracies and Losses for Centralized FL (Student) Over Rounds on FMNIST.csv"
df.to_csv(output_path, index=False)