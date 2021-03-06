import numpy as np

import json
from random import shuffle


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        rng = np.random.default_rng()
        # a is which neuron in the layer
        # b is each weight corresponding to neuron in the previous layer
        self.weights = [rng.standard_normal(size=[a, b]) for a, b in zip(sizes[1:], sizes[:-1])]
        self.biases = [rng.standard_normal(size=[a]) for a in sizes[1:]]
        self.training_history = []

    def feed_forward(self, arr):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, arr) + b
            arr = sigmoid(z)
        return arr

    def backprop(self, data):
        arr, y = data

        activations = [arr]
        zs = []
        nabla_w = [np.zeros(a.shape) for a in self.weights]
        nabla_b = [np.zeros(a.shape) for a in self.biases]

        # feedforward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, arr) + b
            zs.append(z)
            arr = sigmoid(z)
            activations.append(arr)

        # compute output error
        delta = cost_deriv(activations[-1], y)  # * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.einsum('i,j->ij', delta, activations[-2])
        
        # backprop error
        for l in range(2, len(self.sizes)):
            w = self.weights[-l + 1].transpose()
            z = zs[-l]
            delta = w.dot(delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.einsum('i,j->ij', delta, activations[-l-1])
        return nabla_b, nabla_w

    def gradient_descent(self, data, lr):
        net_nabla_b = [np.zeros(b.shape) for b in self.biases]
        net_nabla_w = [np.zeros(w.shape) for w in self.weights]
        for d in data:
            nabla_b, nabla_w = self.backprop(d)
            net_nabla_b = [b + db for b, db in zip(net_nabla_b, nabla_b)]
            net_nabla_w = [w + dw for w, dw in zip(net_nabla_w, nabla_w)]
        m = len(data)
        self.biases = [b - db * (lr/m) for b, db in zip(self.biases, net_nabla_b)]
        self.weights = [w - dw * (lr/m) for w, dw in zip(self.weights, net_nabla_w)]

    def train(self, training_data, epochs, batch_size, lr, test_data=None):
        for i in range(epochs):
            shuffle(training_data)
            for j in range(0, len(training_data), batch_size):
                self.gradient_descent(training_data[j:j+batch_size], lr)
            print(f'Epoch {i + 1} of {epochs}')

            test_results = -1
            # test
            if test_data is not None:
                test_results = self.test(test_data)
            self.training_history.append((batch_size, lr, test_results))

    def test(self, test_data):
        correct = 0
        for input, actual in test_data:
            if np.argmax(self.feed_forward(input)) == actual:
                correct += 1
        print(f'{correct}/{len(test_data)} correct')
        return correct / len(test_data)

    def to_file(self, path):
        s = {
            'sizes': self.sizes,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'history': self.training_history,
        }
        with open(path, mode='w') as f:
            json.dump(s, f)

    @classmethod
    def from_file(cls, path):
        with open(path, mode='r') as f:
            s = json.load(f)
        net = cls(s['sizes'])
        net.weights = [np.array(w) for w in s['weights']]
        net.biases = [np.array(b) for b in s['biases']]
        net.training_history = s['history']
        return net


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    t = sigmoid(z)
    return t * (1 - t)

# def relu(z):
#     return np.where(z<0, 0, z)

# def relu_prime(z):
#     return np.where(z<0, 0, 1)

def cost_deriv(a, y):
    """Mean squared error derivative"""
    return a - y