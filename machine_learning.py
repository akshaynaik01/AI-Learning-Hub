import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

class KNearestNeighbors:
    """K-Nearest Neighbors classifier implementation."""
    
    def __init__(self, k=3):
        """Initialize KNN classifier.
        
        Args:
            k: Number of nearest neighbors to consider
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Fit the model with training data."""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """Make predictions for the given data."""
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class LogisticRegression:
    """Logistic Regression classifier implementation."""
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        """Initialize logistic regression."""
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the logistic regression model."""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            dW = (1/m) * np.dot(X.T, (predictions - y))
            db = (1/m) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """Make predictions."""
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)
