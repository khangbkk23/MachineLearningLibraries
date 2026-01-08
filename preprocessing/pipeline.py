import numpy as np

class Pipeline:
    def __init__(self, steps):
        self.steps = steps
    
    def fit(self, X):
        for _, step in self.steps:
            X = step.fit_transform(X)
            
        return self
    
    def transform(self, X):
        for _, step in self.steps:
            X = step.transform(X)
        
        return X
    
    def fit_transform(self, X):
        for _, step in self.steps:
            X = step.fit_transform(X)
        
        return X