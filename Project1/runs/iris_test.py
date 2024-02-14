# %%
import sys

sys.path.append('c:/Users/Jonah/Documents/git/IT3030-Deep-Learning/Project1/core')

# %%
import numpy as np
from setup import *
from functions import L1, CrossEntropy, Sigmoid, Softmax


# %%
def load_iris():
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split

    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    encoder = OneHotEncoder()
    y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    #split the train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    # One-hot encode the target variable
    return X_train, X_val, X_test, y_train, y_val, y_test

# %%
def initialize_network():
    network = Network(output_function=Softmax(), error_function=CrossEntropy(),regularization=L1(),regularization_rate=0.001)
    network.add_layer(Layer(num_inputs=4, num_neurons=6, activation=Sigmoid(),learning_rate=0.01,weight_range=[-0.5,0.5]))
    network.add_layer(Layer(num_inputs=6, num_neurons=3, activation=Sigmoid(),learning_rate=0.01,weight_range=[-0.5,0.5]))
    return network

# %%
test_error, training_errors, validation_errors = train_model(
    batch_size=10,
    num_epochs=100,
    dataset=load_iris,
    network=initialize_network(),
    error_function=CrossEntropy(),
)
print(f"Test error: {test_error:.4f}")

# %%
import matplotlib.pyplot as plt
# Plot the accuracy         
plt.xlabel('Epoch')
plt.ylabel('Error')

plt.plot(training_errors)
plt.plot(validation_errors)
plt.scatter(len(training_errors), test_error, color='red', label='Test')
plt.annotate(str(round(test_error,5)), (len(training_errors), test_error))
plt.legend(['Training', 'Validation', 'Test'])
plt.show()



