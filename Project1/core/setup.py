import numpy as np

from functions import ACTIVATION_FUNCTIONS, ERROR_FUNCTIONS

class Layer:
    def __init__(self, num_inputs,num_neurons, activation, learning_rate, weight_range):
        self.num_neurons = num_neurons
        self.activation = activation
        self.weights = np.random.uniform(weight_range[0], weight_range[1], (num_inputs, num_neurons))
        self.biases = np.zeros((1, self.num_neurons))
        self.learning_rate = learning_rate

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation.forward(np.dot(inputs, self.weights) + self.biases)
        return self.output
    
    def backward(self, delta, regularization, regularization_rate):
        delta *= self.activation.backward(self.output)
        self.weights -= self.learning_rate * np.dot(self.inputs.T, delta)
        self.biases -= self.learning_rate * np.sum(delta, axis=0, keepdims=True)
        if regularization:
            self.weights -= regularization_rate * regularization.backward(self.weights)
        
        return np.dot(delta, self.weights.T) #new delta for upstream layer
    
    def get_weights(self):
        return self.weights

class Network:
    def __init__(self, output_function, error_function, regularization=None, regularization_rate=0.001):
        self.layers = []
        self.output_function = output_function
        self.error_function = error_function
        self.regularization = regularization
        self.regularization_rate = regularization_rate

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        self.output = self.output_function.forward(output)
        return np.asarray(self.output)
    
    def backward(self, target):
        delta = self.error_function.backward(target,self.output)*self.output_function.backward(self.output)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.regularization, self.regularization_rate)
            
    def error(self, target):
        return self.error_function.forward(target, self.output)
    
    def penalty_term(self):
        if self.regularization == None:
            return 0
        
        penalty = 0
        for layer in self.layers:
            penalty += self.regularization.forward(layer.get_weights())
        return penalty * self.regularization_rate

def train_model(batch_size:int, num_epochs:int,dataset,network:Network, error_function):
    X_train, X_val, X_test, y_train, y_val, y_test = dataset()

    training_errors = []
    training_losses = []
    validation_errors = []
    validation_losses = []
    training_accuracy = []
    validation_accuracy = []

    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(num_epochs):
        # Shuffle the training data
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Split the shuffled data into mini-batches
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size

            # Forward pass
            output = network.forward(X_train_shuffled[start:end])

            # Backward pass
            network.backward(target=y_train_shuffled[start:end])

        # training error
        output = network.forward(X_train)
        training_pred = np.argmax(output, axis=1)
        training_errors.append(error_function.forward(y_train, output))
        training_losses.append(training_errors[-1]+network.penalty_term())
        # validation error
        output = network.forward(X_val)
        validation_pred = np.argmax(output, axis=1)
        validation_errors.append(error_function.forward(y_val, output))
        validation_losses.append(validation_errors[-1]+network.penalty_term())
        #training accuracy
        y_train_label = np.argmax(y_train, axis=1)
        training_accuracy.append(np.mean(y_train_label == training_pred))
        #validation accuracy
        y_val_label = np.argmax(y_val, axis=1)
        validation_accuracy.append(np.mean(y_val_label == validation_pred))
        
    # test error
    output = network.forward(X_test)
    test_error = error_function.forward(y_test, output)
    return test_error, training_errors, validation_errors, training_losses, validation_losses, training_accuracy, validation_accuracy

def train_XOR():
    dataset = np.asarray([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    network = Network(output_function=ACTIVATION_FUNCTIONS['linear'], error_function=ERROR_FUNCTIONS['mse'])
    network.add_layer(Layer(num_inputs=2,num_neurons=2, activation=ACTIVATION_FUNCTIONS['relu'], learning_rate=0.1, weight_range=(-0.1,0.1)))
    network.add_layer(Layer(num_inputs=2,num_neurons=1, activation=ACTIVATION_FUNCTIONS['sigmoid'], learning_rate=0.1, weight_range=(-0.1,0.1)))
    for i in range(100):
        for data in dataset:
            network.forward(np.reshape(data[:2], (1, 2)))
            network.backward(data[2])
        print(network.error(np.reshape(dataset[:,2], (4, 1))))
        # target = np.reshape(dataset[:,2], (4,1))
        # network.forward(dataset[:,:2])
        # print(network.error(target))
        # network.backward(target)
        
