import numpy as np
from nn.sgd import SGD
from nn.softmax_cross_entropy import SoftmaxCrossEntropy
from nn.dense import Dense
from nn.relu import ReLU
import matplotlib.pyplot as plt

def train(lr):
    np.random.seed(0)
    dense_1 = Dense(4, 5)
    relu = ReLU()
    dense_2 = Dense(5, 3)
    softmax = SoftmaxCrossEntropy()
    sgd = SGD(lr)
    X = np.random.randn(3, 4)
    Y = np.eye(3)
    params = {
            "W1": dense_1.W,
            "b1": dense_1.b,
            "W2": dense_2.W,
            "b2": dense_2.b
            }
    costs = []

    def compute_cost() -> float:
        Z1 = dense_1.forward(X)
        A1 = relu.forward(Z1)
        Z2 = dense_2.forward(A1)
        return softmax.forward(Z2, Y)

    for i in range(200):
        costs.append(compute_cost())
        dZ2 = softmax.backward()
        dA1 = dense_2.backward(dZ2)
        dZ1 = relu.backward(dA1)
        _ = dense_1.backward(dZ1)
        sgd.step([dense_1, relu, dense_2, softmax])

    return costs

if __name__ == "__main__":
    learning_rates = [0.01, 0.1, 1.92, 10, 100]

    results = {}
    for lr in learning_rates:
        results[lr] = train(lr)
        costs = results[lr]
        print(f"lr = {lr:<6} first J = {costs[0]:.4f}   last J = {costs[-1]:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, scale in zip(axes, ["linear", "log"]):
        for lr in learning_rates:
            ax.plot(results[lr], linewidth=1.8, label=f"lr = {lr}")
        ax.set_yscale(scale)
        ax.set_title(f"Cost vs. iteration ({scale} scale)", fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost J")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig("../sgd_learning_rates.png", dpi=150)
    plt.show()



