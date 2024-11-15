import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import shap

def calculate_feature_importance(teacher_model, x_train, y_test):
    # Calculate Shapley values for feature importance (client-side)
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(teacher_model, background)
    # Example of each class from the test set
    x_test_dict = dict()
    for i, l in enumerate(y_test):
        if len(x_test_dict) == 10:
            break
        if l not in x_test_dict.keys():
            x_test_dict[l] = x_train[i]

    # Convert to list preserving order of classes
    x_test_each_class = [x_test_dict[i] for i in sorted(x_test_dict)]
    x_test_each_class = np.asarray(x_test_each_class)
    # Print shape of tensor
    print(f"x_test_each_class tensor has shape: {x_test_each_class.shape}")
    
    # Compute predictions
    predictions = teacher_model.predict(x_test_each_class)

    # Apply argmax to get predicted class
    np.argmax(predictions, axis=1)

    # Compute Shapley values using DeepExplainer instance
    shap_values = explainer.shap_values(x_test_each_class)

    # Aggregate Shapley values (e.g., average absolute values)
    feature_importance = np.mean(np.abs(shap_values), axis=0) #This aggregation results in a single array, feature_importance, 
                                                              # representing the average importance of each pixel in the MNIST images.

    return feature_importance