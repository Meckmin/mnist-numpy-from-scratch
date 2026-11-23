import numpy as np
from nn.softmax_cross_entropy import SoftmaxCrossEntropy
from nn.dense import Dense
from nn.relu import ReLU
from nn.model import Model

np.random.seed(0)
dense_1 = Dense(4, 5)
relu = ReLU()
dense_2 = Dense(5, 3)
softmax = SoftmaxCrossEntropy()
model = Model([dense_1, relu, dense_2], softmax)
X = np.random.randn(3, 4)
Y = np.eye(3)
params = {
        "W1": dense_1.W,
        "b1": dense_1.b,
        "W2": dense_2.W,
        "b2": dense_2.b
        }

def compute_cost() -> float:
    return model.compute_loss(X, Y)

def analytic_gradient() -> dict:
    model.compute_loss(X, Y)
    model.backward()
    gradients = {
        "W1": dense_1.dW.copy(),
        "b1": dense_1.db.copy(),
        "W2": dense_2.dW.copy(),
        "b2": dense_2.db.copy()
    }
    return gradients

def numerical_gradient(param: np.ndarray) -> np.ndarray:
    eps = 1e-5
    param_numerical = np.zeros(param.shape)
    for k in range(param.shape[0]):
        for j in range(param.shape[1]):
            original = param[k, j]
            param[k, j] = original + eps
            J_plus = compute_cost()
            param[k, j] = original - eps
            J_minus = compute_cost()
            param_numerical[k, j] = (J_plus - J_minus) / (2*eps)
            param[k, j] = original
    return param_numerical

def relative_error(anl: np.ndarray, num: np.ndarray) -> np.ndarray:
    return (np.abs(num - anl)) / (np.maximum(np.abs(num), np.abs(anl)) + 1e-12)

def comparison() -> bool:
    gradients = analytic_gradient()
    all_ok = True

    for name, param in params.items():
        num = numerical_gradient(param)
        anl = gradients[name]
        err = relative_error(anl, num).max()
        print(f"{name}: {err:.1e}")
        if err > 1e-7:
            all_ok = False

    return all_ok

def test_gradients():
    assert comparison(), "Some gradients are wrong"

if __name__ == "__main__":
    assert test_gradients()
    print("All gradients are OK")

