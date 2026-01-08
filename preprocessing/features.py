import numpy as np
from itertools import combinations_with_replacement, combinations

class PolynomialFeatures:
    def __init__(self, degree=2, interaction_only=False, include_bias=True, order="C"):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order
    
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        self.n_features_in_ = n_features
        comb_func = (
            combinations if self.interaction_only
            else combinations_with_replacement
        )

        powers = []

        # bias term
        if self.include_bias:
            powers.append(np.zeros(n_features, dtype=int))

        for deg in range(1, self.degree + 1):
            for comb in comb_func(range(n_features), deg):
                power = np.zeros(n_features, dtype=int)
                for idx in comb:
                    power[idx] += 1
                powers.append(power)

        self.powers_ = np.array(powers, dtype=int)
        self.n_output_features_ = len(self.powers_)

        return self


    def transform(self, X):
        X = np.array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features than during fit")

        X_out = np.ones((X.shape[0], self.n_output_features_), dtype=float)

        for i, power in enumerate(self.powers_):
            X_out[:, i] = np.prod(X ** power, axis=1)

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        feature_names = []

        for power in self.powers_:
            if np.all(power == 0):
                feature_names.append("1")
            else:
                terms = []
                for feature, p in zip(input_features, power):
                    if p == 1:
                        terms.append(feature)
                    elif p > 1:
                        terms.append(f"{feature}^{p}")
                feature_names.append(" ".join(terms))

        return np.array(feature_names, dtype=object)
