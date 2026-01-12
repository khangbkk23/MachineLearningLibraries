import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, mse=None, n_samples=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.mse = mse
        self.n_samples = n_samples
class DecisionTreeRegressor:
	"""
	using MSE for splitting.
	using CART algorithm
	"""
    def __init__(self, )