import numpy as np

class Utils:
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state) 
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        
        test_count = int(len(X) * test_size)
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-8)
    
    def transform(self, X):
        if self.mean is None or self.std is None:
            raise Exception("Scaler chưa được fit trên dữ liệu train.")
        return (X - self.mean) / (self.std + 1e-8)