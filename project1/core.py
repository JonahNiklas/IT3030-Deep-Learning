import numpy as np

class Layer:
    def __init__(self, num_inputs,num_neurons, activation, learning_rate):
        self.num_neurons = num_neurons
        self.activation = activation
        self.weights = np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, self.num_neurons))
        self.learning_rate = learning_rate

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation.forward(np.dot(inputs, self.weights) + self.biases)
        return self.output
    
    def backward(self, delta):
        delta = self.activation.backward(delta) * np.dot(delta, self.weights.T)
        self.weights -= self.learning_rate * np.dot(self.inputs.T, delta)
        self.biases -= self.learning_rate * np.sum(delta, axis=0, keepdims=True)
        return delta
        

    

class Network:
    def __init__(self, output_function):
        self.layers = []
        self.output_function = output_function
        

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        self.output = self.output_function.forward(output)
        return self.output
    
    def backward(self, error_function,error):
        delta = error_function.backward(error)*self.output_function.backward(self.output)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)


            