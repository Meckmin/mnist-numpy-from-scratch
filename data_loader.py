import gzip
import os
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

def load_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        file_content = f.read()

    image_data = np.frombuffer(file_content, dtype=np.ubyte, offset=16)
    return np.reshape(image_data, (-1, 784))

def load_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        file_content = f.read()

    return np.frombuffer(file_content, dtype=np.ubyte, offset=8)

def preprocess_data(X: np.ndarray, Y: np.ndarray) -> tuple:
    X = X / 255
    Y = np.eye(10)[Y]
    return X, Y

def load_mnist(val_size: int = 10000, data_dir: str = DATA_DIR) -> tuple:
    X_train = load_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    Y_train = load_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test = load_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    Y_test = load_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train))
    idx_train = idx[val_size:]
    idx_val = idx[:val_size]

    # Separating validation and training matrices
    X_val = X_train[idx_val]
    Y_val = Y_train[idx_val]
    X_train = X_train[idx_train]
    Y_train = Y_train[idx_train]

    # Normalization and convertion to one-hot format
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_val, Y_val = preprocess_data(X_val, Y_val)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist(val_size=10000)

    # Checking shapes
    assert np.shape(X_train) == (50000, 784), f"X_train shape is wrong, get = {np.shape(X_train)}"
    assert np.shape(Y_train) == (50000, 10), f"Y_train shape is wrong, get = {np.shape(Y_train)}"
    assert np.shape(X_val) == (10000, 784), f"X_val shape is wrong, get = {np.shape(X_val)}"
    assert np.shape(Y_val) == (10000, 10), f"Y_val shape is wrong, get = {np.shape(Y_val)}"
    assert np.shape(X_test) == (10000, 784), f"X_test shape is wrong, get = {np.shape(X_test)}"
    assert np.shape(Y_test) == (10000, 10), f"Y_test shape is wrong, get = {np.shape(Y_test)}"

    # Min and max values of X_train
    print(f"Min of X_train: {np.min(X_train)}")
    print(f"Max of X_train: {np.max(X_train)}")

    # Checking if Y_train rows add up to 1
    assert np.all(np.sum(Y_train, axis=1) == 1), f"Y_train rows do not add up to 1, get = {np.sum(Y_train, axis=1)}"

    # Counting the samples of each number in dataset
    result = np.bincount(np.argmax(Y_train, axis=1))
    for i in range(10):
        print(f"# of {i} = {result[i]} samples")

    print(f"Total = {result.sum()} samples")
