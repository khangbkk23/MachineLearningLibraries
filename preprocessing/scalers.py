from networkx import maximum_flow_value
import numpy as np

class StandardScaler:
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X):
        X = np.array(X)
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = None

        if self.with_std:
            self.std_ = np.std(X, axis=0, ddof=0)
        else:
            self.std_ = None
            
        return self

    def transform(self, X):
        X = np.array(X)
        
        if self.with_mean:
             X = X - self.mean_
        
        if self.with_std:
            scale = self.std_ + 1e-8
            X = X / scale
            
        return X
    
    def inverse_transform(self, X):
        X = np.array(X)
        if self.with_std:
            X = X * (self.std_ + 1e-8)
        if self.with_mean:
            X = X + self.mean_
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1), clip=False):
        self.feature_range = feature_range
        self.min_val = feature_range[0]
        self.max_val = feature_range[1]
        self.clip = clip # Option mới

    def fit(self, X):
        X = np.array(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # FIX LỖI: Xử lý trường hợp range = 0 (max == min)
        # Nếu range = 0, ta gán = 1 để tránh chia cho 0.
        self.data_range_[self.data_range_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.array(X)
        
        if self.clip:
            X = np.clip(X, self.data_min_, self.data_max_)

        temp = (X - self.data_min_) / self.data_range_
        X_scaled = temp * (self.max_val - self.min_val) + self.min_val
        
        return X_scaled

    def inverse_transform(self, X):
        X = np.array(X)
        
        temp = (X - self.min_val) / (self.max_val - self.min_val)
        X_org = temp * self.data_range_ + self.data_min_
        
        return X_org

    def fit_transform(self, X):
        return self.fit(X).transform(X)