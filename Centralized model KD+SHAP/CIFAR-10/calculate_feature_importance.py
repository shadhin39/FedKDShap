import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import shap

def calculate_feature_importance(teacher_model, x_train, y_test):
    # Calculate Shapley values for feature importance (client-side)
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

    explainer = shap.DeepExplainer(teacher_model, background)
    shap_values = explainer.shap_values(x_train[1:5])

    # Aggregate Shapley values (e.g., average absolute values)
    feature_importance = np.mean(np.abs(shap_values), axis=0) #This aggregation results in a single array, feature_importance, 
                                                              # representing the average importance of each pixel in the CIFAR-10 images.

    return feature_importance