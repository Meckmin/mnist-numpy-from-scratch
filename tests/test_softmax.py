import numpy as np
from nn.softmax_cross_entropy import SoftmaxCrossEntropy

def test_known_values():
    layer = SoftmaxCrossEntropy()
    Z = np.array([[2.0, 1.0, 0.1], [1000, 1001, 1002]])
    Y = np.eye(3)[[0, 2]]
    J = layer.forward(Z, Y)
    assert np.isclose(J, 0.412, atol=1e-3), f"Expected ≈ 0.412, got = {J}"

    dZ = layer.backward()
    assert dZ.shape == Z.shape, f"Shapes of dZ and Z are different, dZ = {dZ.shape}, Z = {Z.shape}"
    assert np.allclose(dZ[0], np.array([-0.170, 0.121, 0.049]), atol=1e-2), f"Expected ≈ [-0.170, 0.121, 0.049], got = {dZ}"
    assert np.allclose(np.sum(dZ, axis=1), 0.0, atol=1e-3), f"Sum of dZ rows are not equal 0, got = {np.sum(dZ, axis=1)}"

def test_uniform():
    layer = SoftmaxCrossEntropy()
    Z = np.zeros((5, 10))
    Y = np.eye(10)[np.random.randint(0, 10, 5)]
    J = layer.forward(Z, Y)
    assert np.isclose(J, np.log(10), atol=1e-3), f"Expected ≈ 2.302, got = {J}"

if __name__ == "__main__":
    test_known_values()
    test_uniform()