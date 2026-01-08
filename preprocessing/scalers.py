from networkx import maximum_flow_value
import numpy as np

class StandardScaler:
    def fit(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_  = np.std(X, axis=0, ddof=0)
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / (self.std_ + 1e-8)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MinMaxScaler:
    def __init__(self, feature_range=(0,1)):
        self.min_value = feature_range[0]
        self.max_value = feature_range[1]
    
    def fit(self, X):
        X = np.array(X)
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        
        self.data_range = self.data_max - self.data_min
        self.data_range[self.data_range] = 1.0
        return self
    
    def transform(self, X):
        X = np.array(X)
        
        # (X - min) / (max - min)
        temp = (X - self.data_min)/self.data_range
        X_scaled = temp * (self.max_value - self.min_value) + self.min_value
        
        return X_scaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

