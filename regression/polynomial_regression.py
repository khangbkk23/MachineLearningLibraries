from django.urls import include
import numpy
from preprocessing.features import PolynomialFeatures
from regression.linear.linear_regression import LinearRegression
class PolynomialRegression(LinearRegression):
    def __init__(self, degree=2, interaction_only=False, include_bias=False, fit_intercept=True, copy_X=True, n_jobs=None, positive=False):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, 
                         n_jobs=n_jobs, positive=positive)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly_transformer = None
        
    def fit(self, X, y, sample_weight=None):
        self.poly_transformer = PolynomialFeatures(
			degree = self.degree,
			interaction_only = self.interaction_only,
			include_bias = self.include_bias,
			order = "C"
		)
        X_poly = self.poly_transformer.fit_transform(X)
        super().fit(X_poly, y, sample_weight=sample_weight)
        self.n_features_in_ = self.poly_transformer.n_features_in_
        return self
    
    def predict(self, X):
        if self.poly_transformer is None:
            raise RuntimeError("Model haven't been fitted.")

        X_poly = self.poly_transformer.transform(X)
        return super().predict(X_poly)
    
    def get_feature_names_out(self, input_features=None):
        if self.poly_transformer is None:
            raise RuntimeError("Model haven't been fitted.")
        return self.poly_transformer.get_feature_names_out(input_features)