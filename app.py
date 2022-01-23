from mnist import MNIST
import numpy as np

from random import shuffle


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        rng = np.random.default_rng()
        # a is which neuron in the layer
        # b is each weight corresponding to neuron in the previous layer
        self.weights = [rng.standard_normal(size=[a, b]) for a, b in zip(sizes[1:], sizes[:-1])]
        self.biases = [rng.standard_normal(size=[a]) for a in sizes[1:]]

    def feed_forward(self, arr):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, arr) + b
            arr = sigmoid(z)
        return arr

    def backprop(self, data):
        arr, y = data

        activations = [arr]
        zs = []
        nabla_w = []
        nabla_b = []

        # feedforward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, arr) + b
            zs.append(z)
            arr = sigmoid(z)
            activations.append(arr)

        # compute output error
        delta = cost_deriv(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b.append(delta)
        nabla_w.append(activations[-2] * delta.reshape((len(delta), 1)))
        
        # backprop error
        for l in range(2, len(self.sizes)):
            w = self.weights[-l + 1].transpose()
            z = zs[-l]
            delta = w.dot(delta) * sigmoid_prime(z)
            nabla_b.insert(0, (delta))
            nabla_w.insert(0, (activations[-l-1] * delta.reshape((len(delta), 1))))
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

    def train(self, training_data, epochs, batch_size, lr, test_data):
        for i in range(epochs):
            shuffle(training_data)
            for j in range(0, len(training_data), batch_size):
                self.gradient_descent(training_data[j:j+batch_size], lr)

            # test
            correct = 0
            for input, actual in test_data:
                if np.argmax(self.feed_forward(input)) == actual:
                    correct += 1
            print(f'Epoch {i + 1} of {epochs}\n{correct}/{len(test_data)} correct')
            with open('results.txt', mode='a') as f:
                f.write(f'{correct}/{len(test_data)}\n')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    t = sigmoid(z)
    return t * (1 - t)

def cost_deriv(output, y):
    return output - y

def get_mnist():
    mndata = MNIST('data')
    images, labels = mndata.load_training()
    t = []
    for label in labels:
        a = np.zeros(10)
        a[label] = 1
        t.append(a)
    images = [np.array(i) for i in images]
    training_data = list(zip(images, t))
    test_images, test_labels = mndata.load_testing()
    test_images = [np.array(i) for i in test_images]
    test_data = list(zip(test_images, test_labels))
    return training_data, test_data

if __name__ == '__main__':
    training_data, test_data = get_mnist()

    net = Network((784, 128, 64, 10))
    net.train(training_data, 1000, 1000, 3, test_data)