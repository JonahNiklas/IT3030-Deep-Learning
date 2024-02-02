import numpy as np

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
    
    def backward(self, delta):
        delta *= self.activation.backward(self.output)
        self.weights -= self.learning_rate * np.dot(self.inputs.T, delta)
        self.biases -= self.learning_rate * np.sum(delta, axis=0, keepdims=True)
        return np.dot(delta, self.weights.T) #new delta for upstream layer
        

class Network:
    def __init__(self, output_function, error_function):
        self.layers = []
        self.output_function = output_function
        self.error_function = error_function

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
            delta = layer.backward(delta)
            
    def error(self, target):
        return self.error_function.forward(target, self.output)

# logsumexp function to avoid overflow/underflow
def logsumexp(x):
    offset = np.max(x, axis=0)
    return offset + np.log(np.sum(np.exp(x - offset), axis=0))


# log of sigmoid is easier to define apply the logsumexp to
def logsigma(x):
    if isinstance(x, np.ndarray):
        return -logsumexp(np.array([np.zeros(x.shape), -x]))
    return -logsumexp(np.array([0,-x]))


# activation functions
class Sigmoid():
    def forward(self,x):
        return np.exp(logsigma(x))
    
    def backward(self,sigmoid):
        return sigmoid * (1 - sigmoid)

class ReLU():
    def forward(self,x):
        return np.maximum(0, x)
    
    def backward(self,x):
        return np.where(x > 0, 1, 0)

class Tanh():
    def forward(self,x):
        return np.tanh(x)
    def backward(self,x):
        return 1 - np.tanh(x)**2

# output_functions
class Softmax():
    def forward(self, x):
        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, output):
        m, n = output.shape
        a = np.eye(output.shape[-1])
        diagonal = np.zeros((m, n, n),dtype=np.float32)
        off_diagonal = np.zeros((m, n, n),dtype=np.float32)
        diagonal = np.einsum('ij,jk->ijk',output,a)
        off_diagonal = np.einsum('ij,ik->ijk',output,output)
        jacobian = diagonal - off_diagonal
        return np.einsum('ijk,ik->ij', jacobian, output)
    
class Linear():
    def forward(self, x):
        return x

    def backward(self, output):
        return output

ACTIVATION_FUNCTIONS = {
    'sigmoid': Sigmoid(),
    'relu': ReLU(),
    'tanh': Tanh(),
    'softmax': Softmax(),
    'linear': Linear(),
}


class MSE():
    def forward(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class CrossEntropy():
    def forward(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred))

    def backward(self, y_true, y_pred):
        return -y_true / y_pred

ERROR_FUNCTIONS = {
    'cross_entropy': CrossEntropy(),
    'mse': MSE(),
}
def train_model(batch_size:int, num_epochs:int,dataset,network:Network, error_function):
    X_train, X_val, X_test, y_train, y_val, y_test = dataset()

    training_errors = []
    validation_errors = []

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

        # training accuracy
        output = network.forward(X_train)
        training_errors.append(error_function.forward(y_train, output))
        
        # validation accuracy
        output = network.forward(X_val)
        validation_errors.append(error_function.forward(y_val, output))
    # test accuracy
    output = network.forward(X_test)
    test_error = error_function.forward(y_test, output)
    return test_error, training_errors, validation_errors

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
        
