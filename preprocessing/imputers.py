import numpy as np
from collections import Counter

class SimpleImputer:
    def __init__(self, strategy="mean", fill_value=None, missing_values=np.nan):
        if strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError("strategy must be one of: mean, median, most_frequent, constant")

        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self.statistics_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=object)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_features = X.shape[1]
        stats = []

        for j in range(n_features):
            col = X[:, j]

            if self.missing_values is np.nan:
                mask = ~np.isnan(col.astype(float, copy=False))
            else:
                mask = col != self.missing_values

            valid = col[mask]

            if self.strategy == "mean":
                stat = np.mean(valid.astype(float))

            elif self.strategy == "median":
                stat = np.median(valid.astype(float))

            elif self.strategy == "most_frequent":
                stat = Counter(valid).most_common(1)[0][0]

            elif self.strategy == "constant":
                if self.fill_value is None:
                    raise ValueError("fill_value must be specified for constant strategy")
                stat = self.fill_value

            stats.append(stat)

        self.statistics_ = np.array(stats, dtype=object)
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer has not been fitted yet")

        X = np.asarray(X, dtype=object)
        X_out = X.copy()

        for j in range(X_out.shape[1]):
            if self.missing_values is np.nan:
                mask = np.isnan(X_out[:, j].astype(float, copy=False))
            else:
                mask = X_out[:, j] == self.missing_values

            X_out[mask, j] = self.statistics_[j]

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)
