import time
import numpy as np
import data_loader
from nn.dense import Dense
from nn.model import Model
from nn.relu import ReLU
from nn.sgd import SGD
from nn.softmax_cross_entropy import SoftmaxCrossEntropy
import json

BATCH_SIZE = 64

def train(X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, lr: float, hidden_size: int, epochs: int):
    rng = np.random.default_rng(42)
    np.random.seed(0)

    layers = [Dense(784, hidden_size), ReLU(), Dense(hidden_size, 10)]
    softmax = SoftmaxCrossEntropy()
    model = Model(layers, softmax)
    sgd = SGD(lr)

    val_losses, val_accuracies, epoch_times, train_losses = [], [], [], []

    def accuracy(X, Y):
        return 100 * np.mean(model.predict(X) == np.argmax(Y, axis=1))

    def shuffle(X, Y):
        idx = rng.permutation(len(X))
        return X[idx], Y[idx]

    # sanity check
    val_losses.append(model.compute_loss(X_val, Y_val))
    val_accuracies.append(accuracy(X_val, Y_val))
    train_losses.append(model.compute_loss(X_train, Y_train))

    for i in range(epochs):
        start_time = time.time()
        X_train_shuffled, Y_train_shuffled = shuffle(X_train, Y_train)

        for start in range(0, np.shape(X_train)[0], BATCH_SIZE):
            X = X_train_shuffled[start: start + BATCH_SIZE]
            Y = Y_train_shuffled[start: start + BATCH_SIZE]
            model.compute_loss(X, Y)
            model.backward()
            sgd.step(model.layers)

        val_accuracies.append(accuracy(X_val, Y_val))
        val_losses.append(model.compute_loss(X_val, Y_val))
        train_losses.append(model.compute_loss(X_train, Y_train))

        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

    history = {
        "train_losses" : train_losses,
        "val_losses" : val_losses,
        "val_accuracies" : val_accuracies,
        "epoch_times" : epoch_times
    }

    return model, history

if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_loader.load_mnist()

    results = {}
    experiments = {
        "lr=0.01" : (0.01, 128, 10),
        "lr=0.05" : (0.05, 128, 10),
        "lr=0.5" : (0.5, 128, 10),
        "hidden_size=32" : (0.1, 32, 10),
        "hidden_size=512" : (0.1, 512, 10),
        "epochs=30" : (0.1, 128, 30)
    }

    for exp_name, params in experiments.items():
        model, history = train(X_train, Y_train, X_val, Y_val, *params)
        val_accs = history["val_accuracies"]
        last_val_acc, max_val_acc, idx_max_val_acc, tr_time = (
            val_accs[-1], np.max(val_accs), np.argmax(val_accs), np.sum(history["epoch_times"]))
        results[exp_name] = history
        print(f"Exp : {exp_name}")
        print(f"Last validation accuracy : %{last_val_acc:.2f}")
        print(f"Max validation accuracy : %{max_val_acc:.2f} at Epoch {idx_max_val_acc}/{params[2]}")
        print(f"Training time : {tr_time:.2f} sec")
        print()

    with open("results.json", "w") as f:
        json.dump(results, f)