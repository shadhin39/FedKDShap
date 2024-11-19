# FedKDShap: Enhancing Federated Learning Via Shapley Values Driven Knowledge Distillation on Non-IID Data. 

## Contents

This repository includes the implementation of the Federated Learning (FL) method FedKDShap with Flower Framework utilizing Knowledge Distillation (KD) and SHAP or Shapley Values, and the implementation of the combination between FedKShap and FedAvg. Currently, it can process CIFAR10, FMNIST, and MNIST datasets.

## Data Preparation

The types of dataset segmentation in this project are IID and Non-IID.
### IID (identifically and independently distributed):
It merely uniformly splits the dataset. The II dataset is used for centralized learning algorithms.
### Non-IID (non-identifically and independently distributed):
The dataset is segmented with its natural property. A popular way to realize non-iid dataset segmentation is by utilizing Dirichlet distribution, which was adopted in this project. It uses Dirichlet distribution on the label ratios to ensure uneven label distributions among clients for non-IID splits.



### Dependencies

- Python >= 3.12.4 
- TensorFlow >= 2.17.0
- Numpy >= 1.26.4
- Keras >= 3.5.0
- flwr >= 1.11.1
- Shap >= 0.46.0
- matplotlib >= 3.8.4
- pandas >= 2.2.2
- CUDA >= 11.2
- cuDNN >= 8.0.4



### Dataset

- CIFAR-10
- FMNIST
- MNIST



### Parameters

The following arguments to the important parameters of the experiment.

| Argument                    | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `num_classes`               | Number of classes                                            |
| `num_clients`               | Number of all clients.                                       |
| `fraction_fit`	            | Number of fraction fit.                                      |
| `fraction_fit`	            | Number of fractions evaluate.                            		 |
| `min_fit_client`            | Number of minimum clients to be fit.                         |
| `min_fit_evaluate`          | Number of minimum clients to be evaluated.                   |
| `min_available_client`      | Number of minimum available clients to be present in server. |
| `federated_rounds`          | Number of federated rounds of the experiment.                |
| `num_partitions`            | Number of partitions used for non-iid data settings.         |
| `min_partitions_size`       | Learning rate of server updating.                            |
| `alpha`                     | Controls the degree of non-IIDness in the data distribution  |
| `ld`                        | Controls the trade-off between $L_{CE}$ and $\lambda L_{KL}.$|



### Usage

Here is an example to run FedKDShap on CIFAR-10, FMNIST, and MNIST datasets.
Change your current working directory to the desired location with the terminal, and run the following commands:
```python
-- cd ../your_location
-- python teacher_model.py
-- python data_loader.py
-- python calculate_feature_importance.py
-- python server.py
Now, open another terminal
-- python client.py
Again, open another terminal
-- python client1.py
.................
```



