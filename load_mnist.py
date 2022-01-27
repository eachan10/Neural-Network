import numpy as np


def read_images(path):
    with open(path, mode='rb') as f:
        magic_number = f.read(4)
        images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        pixels = np.fromfile(f, np.uint8)
    return pixels.reshape([images,cols*rows])

def read_labels(path):
    with open(path, mode='rb') as f:
        magic_number = f.read(4)
        label_count = f.read(4)
        labels = np.fromfile(f, np.uint8)
    return labels

def load_mnist():
    test_images = read_images('data/t10k-images-idx3-ubyte')
    test_labels = read_labels('data/t10k-labels-idx1-ubyte')
    training_images = read_images('data/train-images-idx3-ubyte')
    training_labels = read_labels('data/train-labels-idx1-ubyte')
    t= []
    for label in training_labels:
        a = np.zeros(10)
        a[label] = 1
        t.append(a)
    training_data = list(zip(training_images, t))
    test_data = list(zip(test_images, test_labels))
    return training_data, test_data
