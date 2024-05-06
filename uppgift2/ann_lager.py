import numpy as np

def layer_forward(inputs, weights, bias, sigmoid):
    # Perform matrix multiplication and add bias
    layer_activity = np.dot(inputs, weights.T) + bias.T
    output = sigmoid(layer_activity)
    return output

inputs = np.array([[0.8, -0.2, 0.1]])
weights = np.random.randn(3, 3)
bias = np.random.randn(3, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

output = layer_forward(inputs, weights, bias, sigmoid)

print("Output:", output)