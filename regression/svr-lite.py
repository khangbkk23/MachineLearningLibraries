import numpy as np
class SVR:
    def __init__(self, C=1.0, epsilon=0.1, coef0=1.0, kernel="rbf", degree=3, gamma=None,learning_rate=1e-3, n_iters=1e3):
        self.C = C
        self.epsilon = epsilon
        self.coef0 = coef0
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.intercept_ = 0.0
        self.n_features_in_ = 0
        self._gamma_fit = None
        
    def _get_gamma(self, n_features, X_var):
        if self.gamma is None:
            return 1.0 / (n_features * X_var) if X_var > 0 else 1.0
        return self.gamma
    
    def _kernel_function(self, X1, X2, gamma):
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "poly":
            return (gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            sq_dist = X1_norm + X2_norm - 2 * (X1 @ X2.T)
            sq_dist = np.maximum(sq_dist, 0)
            return np.exp(-gamma * sq_dist)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        gamma = self._get_gamma(n_features, X.var())
        self._gamma_fit = gamma
        
        self.K_train_ = self._kernel_function(X, X, gamma)
        self.dual_coef_ = np.zeros(n_samples)
        self.intercept_ = 0.0
        
        # Gradient Descent Loop
        for _ in range(self.n_iters):
            y_pred = self.K_train_ @ self.dual_coef_ + self.intercept_
            error = y - y_pred
            
            mask_up = error > self.epsilon
            mask_down = error < -self.epsilon
            
            grad = np.zeros(n_samples)
            grad[mask_up] = 1.0
            grad[mask_down] = -1.0
            
            reg_term = self.dual_coef_ / (self.C * n_samples)
            self.dual_coef_ += self.lr * (grad - reg_term)
            self.intercept_ += self.lr * np.sum(grad)

        # Sparsify
        threshold = 1e-4
        sv_mask = np.abs(self.dual_coef_) > threshold
        
        self.support_vectors_ = X[sv_mask]
        self.dual_coef_ = self.dual_coef_[sv_mask]
        
        return self
        
    def predict(self, X):
        if self.support_vectors_ is None:
            raise RuntimeError("Model hasn't been fitted.")
        X = np.array(X, dtype=np.float64)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but SVR is expecting {self.n_features_in_} features.")
        
        K_new = self._kernel_function(X, self.support_vectors_, self._gamma_fit)
        return K_new @ self.dual_coef_ + self.intercept_