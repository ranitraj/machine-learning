"""
Forward propagation function for a simple neural network in Python using NumPy.

Network Architecture Requirements
---------------------------------

- The network must have one hidden layer.
- Each layer should perform a linear transformation (matrix multiplication plus bias).
- Apply a nonlinear activation function (e.g., Sigmoid) after the linear step in each layer.
- The final output layer must use a Sigmoid activation function (for binary classification).

"""
import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid activation function.

    Parameters
    ----------
    z : ndarray or float
        Input value or array of values.

    Returns
    -------
    output : ndarray or float
        Sigmoid activation of the input. Has the same shape as `z`.

    Examples
    --------
    >>> sigmoid(0)
        0.5
    >>> sigmoid(np.array([0, 1, -1]))
        array([0.5, 0.73105858, 0.26894142])

    """
    return 1 / (1 + np.exp(-z))


def forward_propagation(X, W1, b1, W2, b2):
    """
    Perform forward propagation for a simple 2-layer neural network.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    W1 : ndarray of shape (n_features, n_hidden)
        Weights for the first (hidden) layer.
    b1 : ndarray of shape (1, n_hidden)
        Biases for the first (hidden) layer.
    W2 : ndarray of shape (n_hidden, 1)
        Weights for the output layer.
    b2 : ndarray of shape (1, 1)
        Bias for the output layer.

    Returns
    -------
    a1 : ndarray of shape (n_samples, n_hidden)
        Activation from the hidden layer (useful for back-propagation).
    a2 : ndarray of shape (n_samples, 1)
        Output of the network (predicted values).
    z1 : ndarray of shape (n_samples, n_hidden)
        Linear component for the first layer (pre-activation values).
    z2 : ndarray of shape (n_samples, 1)
        Linear component for the output layer (pre-activation values).

    Examples
    --------
    >>> X = np.array([[0.2, 0.4]])
    >>> W1 = np.random.randn(2, 3)
    >>> b1 = np.zeros((1, 3))
    >>> W2 = np.random.randn(3, 1)
    >>> b2 = np.zeros((1, 1))
    >>> a2, a1, z1, z2 = forward_propagation(X, W1, b1, W2, b2)

    """
    # 1. Compute the first layer linear transformation
    z1 = np.dot(X, W1) + b1
    
    # 2. Compute the activation for the first layer
    a1 = sigmoid(z1)
    
    # 3. Compute the second layer linear transformation
    z2 = np.dot(a1, W2) + b2
    
    # 4. Compute the output layer activation:
    a2 = sigmoid(z2)
    
    # 5. Return a1, a2, z1, z2
    return a1, a2, z1, z2


if __name__ == "__main__":
    # 1. Set the random seed for reproducibility.
    np.random.seed(42)
    
    # 2. Define sample input data (2 samples, each with 2 features)
    X = np.array([[0.5, -0.2], [0.1, 0.4]])
    
    # 3. Initialize parameters for a network with 2 input features, 3 hidden units, and 1 output unit.
    W1 = np.random.randn(2, 3) * 0.02
    b1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 1) * 0.02
    b2 = np.zeros((1, 1))
    
    # 4. Run forward propagation by calling forward_propagation(X, W1, b1, W2, b2)
    a1, a2, z1, z2 = forward_propagation(X, W1, b1, W2, b2)
    
    # 5. Print the output (a2) of the forward pass.
    print(f"Output Probabilities = {a2}")