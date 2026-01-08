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
        diff = np.setdiff1d(y, self.classes_)
        if len(diff) > 0:
            raise ValueError(f"y contains new labels: {diff}")
        return np.searchsorted(self.classes_, y)

    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        y = np.array(y)
        return self.classes_[y]

class OrdinalEncoder:
    
    def __init__(self, handle_unknown='error', unknown_value=None, encoded_missing_value=np.nan):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.categories_ = []

    def fit(self, X):
        X = np.array(X, dtype=object)
        self.categories_ = []

        for i in range(X.shape[1]):
            col = X[:, i]
            valid_mask = self._get_valid_mask(col)
            self.categories_.append(np.unique(col[valid_mask]))

        return self

    def transform(self, X):
        X = np.array(X, dtype=object)
        n_samples, n_features = X.shape
        X_trans = np.empty((n_samples, n_features), dtype=np.float64)
        
        for i in range(n_features):
            col = X[:, i]
            cats = self.categories_[i]
            
            valid_mask = self._get_valid_mask(col)
            nan_mask = ~valid_mask
            
            idxs = np.searchsorted(cats, col[valid_mask])
            
            idxs_clipped = np.clip(idxs, 0, len(cats) - 1)
            
            found_mask = (cats[idxs_clipped] == col[valid_mask])
            
            res_col = np.zeros(len(col[valid_mask]), dtype=np.float64)
            res_col[found_mask] = idxs[found_mask]
            
            if not np.all(found_mask):
                if self.handle_unknown == 'error':
                    raise ValueError(f"Found unknown category in column {i}")
                else:
                    res_col[~found_mask] = self.unknown_value

            X_trans[valid_mask, i] = res_col
            X_trans[nan_mask, i] = self.encoded_missing_value
            
        return X_trans

    def inverse_transform(self, X):
        n_samples, n_features = X.shape
        X_inv = np.empty((n_samples, n_features), dtype=object)

        for i in range(n_features):
            cats = self.categories_[i]
            col = X[:, i]
            
            mask_known = np.isin(col, np.arange(len(cats))) 
            known_indices = col[mask_known].astype(int)
            X_inv[mask_known, i] = cats[known_indices]
            
            X_inv[~mask_known, i] = np.nan
            
        return X_inv

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def _get_valid_mask(self, col):
        return np.array([not (isinstance(x, float) and np.isnan(x)) for x in col])


class OneHotEncoder:
    def __init__(self, handle_unknown="ignore", encoded_missing_value=0):
        self.handle_unknown = handle_unknown
        self.encoded_missing_value = encoded_missing_value
        self.categories_ = []
        self.drop_idx_ = []

    def fit(self, X):
        X = np.array(X, dtype=object)
        self.categories_ = []
        self.drop_idx_ = []

        for i in range(X.shape[1]):
            col = X[:, i]
            valid_mask = np.array([x == x and x is not np.nan for x in col])
            cats = np.unique(col[valid_mask])
            self.categories_.append(cats)
            
            if self.drop == 'first' and len(cats) > 1:
                self.drop_idx_.append(0)
            else:
                self.drop_idx_.append(None)
                
        return self

    def transform(self, X):
        X = np.array(X, dtype=object)
        output_blocks = []

        for i in range(X.shape[1]):
            col = X[:, i]
            cats = self.categories_[i]
            n_cats = len(cats)
            
            mat = np.zeros((len(col), n_cats))
            
            for j, val in enumerate(col):
                if not (val == val and val is not np.nan): 
                    continue 
                
                idx = np.where(cats == val)[0]
                if len(idx) > 0:
                    mat[j, idx[0]] = 1
                else:
                    if self.handle_unknown == 'error':
                        raise ValueError(f"Unknown category {val}")
            
            if self.drop_idx_[i] is not None:
                mat = np.delete(mat, self.drop_idx_[i], axis=1)
                
            output_blocks.append(mat)

        return np.hstack(output_blocks)

    def inverse_transform(self, X):
        pass 
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


X = [
    ["red", "S"],
    ["blue", "M"],
    [np.nan, "M"]
]

print("Ordinal:")
print(OrdinalEncoder(encoded_missing_value=-1).fit_transform(X))

print("OneHot:")
print(OneHotEncoder().fit_transform(X))

y = ["yes", "no", "yes"]
print("Label:")
print(LabelEncoder().fit_transform(y))
