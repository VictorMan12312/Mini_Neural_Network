import numpy as np
from typing import Tuple
def load_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a CSV where the last column is the binary target."""
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    # Assumes last col is target and previous are features
    X = data[:, :-1]
    y = data[:, -1:]
    return X, y
def generate_synthetic(samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic classification data forming a circular boundary."""
    np.random.seed(42)
    X = np.random.randn(samples, 2)
    # Target is 1 if outside circle (radius 1), 0 if inside
    y = (X[:, 0]**2 + X[:, 1]**2 > 1.0).astype(int).reshape(-1, 1)
    return X, y
def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """Splits data into training and validation sets."""
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(X.shape[0] * (1.0 - test_size))
    
    X_train = X[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    
    X_test = X[indices[split_idx:]]
    y_test = y[indices[split_idx:]]
    
    return X_train, X_test, y_train, y_test