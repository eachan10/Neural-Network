from mnist import MNIST
import numpy as np


def load_mnist():
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
