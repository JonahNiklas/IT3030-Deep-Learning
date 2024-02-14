import numpy as np

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

class L1():
    def forward(self, weights):
        return np.sum(np.abs(weights))

    def backward(self, weights):
        return np.sign(weights)
    
class L2():
    def forward(self, weights):
        return 1/2 *np.sum(np.square(weights))

    def backward(self, weights):
        return weights
    
REGULARIZATION_FUNCTIONS = {
    'L1': L1(),
    'L2': L2(),
}