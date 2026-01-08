import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None, positive=False):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def fit(self, X, y, sample_weight=None):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.copy_X:
            X = X.copy()
            y = y.copy()

        n_samples, n_features = X.shape
        if sample_weight is not None:
            sample_weight = np.array(sample_weight, dtype=np.float64)
            if sample_weight.ndim == 1:
                sample_weight = sample_weight * (n_samples / np.sum(sample_weight))
            else:
                 raise ValueError("sample_weight must be 1D array")
        
        if self.fit_intercept:
            if sample_weight is None:
                X_mean = np.mean(X, axis=0)
                y_mean = np.mean(y, axis=0)
            else:
                X_mean = np.average(X, axis=0, weights=sample_weight)
                y_mean = np.average(y, axis=0, weights=sample_weight)
            
            X -= X_mean
            y -= y_mean
        else:
            X_mean = np.zeros(n_features)
            y_mean = 0.0 if y.ndim == 1 else np.zeros(y.shape[1])

        if sample_weight is not None:
            sqrt_weight = np.sqrt(sample_weight)[:, np.newaxis]
            X_weighted = X * sqrt_weight
            y_weighted = y * sqrt_weight if y.ndim == 1 else y * sqrt_weight
        else:
            X_weighted = X
            y_weighted = y

        if self.positive:
            if y.ndim > 1:
                raise ValueError("Positive constraint")
            
            self.coef_ = self._solve_nnls_coordinate_descent(X_weighted, y_weighted)
            
        else:
            coef, _, _, _ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)

            if y.ndim == 1:
                self.coef_ = coef
            else:
                self.coef_ = coef.T

        if self.fit_intercept:
            if self.coef_.ndim == 1:
                self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
            else:
                self.intercept_ = y_mean - np.dot(self.coef_, X_mean)
        else:
            self.intercept_ = 0.0

        return self

    def _solve_nnls_coordinate_descent(self, X, y, max_iter=1000, tol=1e-4):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        XtX = X.T @ X
        Xty = X.T @ y
        
        for iteration in range(max_iter):
            w_old = w.copy()
            
            for j in range(n_features):
                residual_j = Xty[j] - (np.dot(XtX[j], w) - XtX[j, j] * w[j])
                
                if XtX[j, j] > 1e-10:
                    w[j] = residual_j / XtX[j, j]
                else:
                    w[j] = 0.0
                    
                if w[j] < 0:
                    w[j] = 0
            
            if np.sum(np.abs(w - w_old)) < tol:
                break
                
        return w

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        if hasattr(self, 'coef_'):
            if self.coef_.ndim == 1:
                return X @ self.coef_ + self.intercept_
            else:
                return X @ self.coef_.T + self.intercept_
        else:
            raise RuntimeError("Model chưa được fit")

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        y = np.array(y)
        
        if sample_weight is not None:
            weight = sample_weight
        else:
            weight = 1.0

        u = np.sum(weight * (y - y_pred) ** 2)
        v = np.sum(weight * (y - np.average(y, weights=sample_weight)) ** 2)
        
        if v == 0: return 0.0
        return 1 - u / v
    
class Ridge:
    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        
    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if self.copy_X:
            X = X.copy()
            y = y.copy()

        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y, axis=0)
            X -= X_mean
            y -= y_mean
        else:
            X_mean = np.zeros(n_features)
            y_mean = 0.0

        XtX = X.T @ X

        A = XtX + self.alpha * np.eye(n_features)

        Xty = X.T @ y
        # A * w = b
        try:
            self.coef_ = np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(A, Xty, rcond=None)[0]

        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        X = np.array(X)
        return X @ self.coef_ + self.intercept_
    
class Lasso:
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic'):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X = X - X_mean
            y = y - y_mean

        w = np.zeros(n_features)

        norm_cols_X = np.sum(X**2, axis=0)
        for iteration in range(self.max_iter):
            w_old = w.copy()
            
            # Thứ tự duyệt features
            if self.selection == 'random':
                features_idx = np.random.permutation(n_features)
            else:
                features_idx = range(n_features)

            max_change = 0.0

            for j in features_idx:
                if norm_cols_X[j] == 0:
                    continue
                
                predictions = X @ w
                residual = y - predictions
                rho = np.dot(X[:, j], residual + X[:, j] * w[j])
                
                w[j] = self._soft_threshold(rho, self.alpha * n_samples) / norm_cols_X[j]
                
                max_change = max(max_change, abs(w[j] - w_old[j]))

            
            if max_change < self.tol:
                break
        
        self.coef_ = w
        
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
            
        self.n_iter_ = iteration + 1
        return self

    def _soft_threshold(self, rho, lam):
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        else:
            return 0.0

    def predict(self, X):
        X = np.array(X)
        return X @ self.coef_ + self.intercept_

class ElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', copy_X=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection
        self.copy_X = copy_X
        
    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        if self.copy_X:
            X = X.copy()
            y = y.copy()
        
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            
            X = X - X_mean
            y = y - y_mean
            
        w = np.zeros(n_features)
        norm_cols_X = np.sum(X**2, axis=0)
        
        l1_reg = self.alpha * self.l1_ratio * n_samples
        l2_reg = self.alpha * (1.0 - self.l1_ratio) *n_samples
        
        for iteration in range(self.max_iter):
            w_old = w.copy()
            if self.selection == 'random':
                features_idx = np.random.permutation(n_features)
            else:
                features_idx = range(n_features)

            max_change = 0.0

            for j in features_idx:
                if norm_cols_X[j] == 0:
                    continue

                predictions = X @ w
                residual = y - predictions
                rho = np.dot(X[:, j], residual + X[:, j] * w[j])
                
                # Tử số: Soft Thresholding
                numerator = self._soft_threshold(rho, l1_reg)
                
                # Mẫu số: Norm + L2 Penalty
                denominator = norm_cols_X[j] + l2_reg
                
                w[j] = numerator / denominator
                
                max_change = max(max_change, abs(w[j] - w_old[j]))

            if max_change < self.tol:
                break
        
        self.coef_ = w
        
        if self.fit_intercept:
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
        else:
            self.intercept_ = 0.0
            
        self.n_iter_ = iteration + 1
        return self

    def _soft_threshold(self, rho, lam):
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        else:
            return 0.0

    def predict(self, X):
        X = np.array(X)
        return X @ self.coef_ + self.intercept_