import numpy as np
from typing import Tuple

class NeuralNetwork:
    """Simple feedforward neural network implementation."""
    
    def __init__(self, layers: list):
        """Initialize neural network with specified layer sizes.
        
        Args:
            layers: List of integers representing the number of neurons in each layer
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights and biases randomly."""
        for i in range(len(self.layers) - 1):
            w = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation."""
        self.cache = [X]
        for w, b in zip(self.weights, self.biases):
            X = self.sigmoid(np.dot(X, w) + b)
            self.cache.append(X)
        return X
    
    def backward(self, X, y, learning_rate):
        """Backward propagation and weight update."""
        m = X.shape[0]
        delta = (self.cache[-1] - y) * self.sigmoid_derivative(self.cache[-1])
        
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.cache[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.cache[i])
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """Train the neural network."""
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict(self, X):
        """Make predictions."""
        return self.forward(X)
