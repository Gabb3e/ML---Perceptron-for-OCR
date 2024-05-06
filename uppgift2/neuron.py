import numpy as np

# Implementera en egen neuron (med for-loop över alla inputs).
# Kapsla in funktionaliteten i en Python-klass.
# Optional: Implementera några olika activation functions, t.ex. Sigmoid, ReLU, Leaky-ReLU och Tanh.

class Neuron():
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        # Initialize weights randomly and bias as 0
        self.weights = np.random.randn(num_inputs)
        self.bias = 0
    
    def forward(self, inputs):
        # Check if the number of inputs matches the number of weights
        assert len(inputs) == len(self.weights), "Number of inputs must match number of weights"

        # Sum up the weighted inputs and add bias
        total = 0
        for i in range(self.num_inputs):
            total += inputs[i] * self.weights[i]
        total += self.bias

        output = self.sigmoid(total)
        return output 

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        # ReLU activation function
        return max(0, x)

if __name__ == "__main__":
    # Create a neuron with inputs
    neuron = Neuron(3)
    # Inputs values
    inputs = [0.8, -0.2, 0.1]
    output = neuron.forward(inputs)
    print("Output:", output)
