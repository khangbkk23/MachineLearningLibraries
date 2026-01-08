import numpy as np

class LabelEncoder:
    def __init__(self):
        pass

    def fit(self, y):
        y = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        y = np.array(y)
        return np.searchsorted(self.classes_, y)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

class OrdinalEncoder:
    def __init__(self):
        pass

    def fit(self, X):
        X = np.array(X, dtype=object)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.array(X, dtype=object)
        X_out = np.zeros(X.shape, dtype=int)

        for i in range(X.shape[1]):
            X_out[:, i] = np.searchsorted(self.categories_[i], X[:, i])

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class OneHotEncoder:
    def __init__(self):
        pass

    def fit(self, X):
        X = np.array(X, dtype=object)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.array(X, dtype=object)
        outputs = []

        for i in range(X.shape[1]):
            cats = self.categories_[i]
            col = X[:, i]

            one_hot = np.zeros((X.shape[0], len(cats)))
            for j, cat in enumerate(cats):
                one_hot[:, j] = (col == cat).astype(int)

            outputs.append(one_hot)

        return np.hstack(outputs)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


X = [
    ["red", "S"],
    ["blue", "M"],
    ["red", "M"]
]

print("Ordinal:")
print(OrdinalEncoder().fit_transform(X))

print("OneHot:")
print(OneHotEncoder().fit_transform(X))

y = ["yes", "no", "yes"]
print("Label:")
print(LabelEncoder().fit_transform(y))

