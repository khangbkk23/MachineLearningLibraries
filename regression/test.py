import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.preprocessing import PolynomialFeatures as SklearnPoly
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from polynomial_regression import PolynomialRegression as MyPolyReg

def generate_data(n_samples=50):
    np.random.seed(42)
    X = np.sort(np.random.rand(n_samples, 1) * 10 - 5, axis=0)
    y = 0.5 * X**2 - 2 * X + 1 + np.random.randn(n_samples, 1) * 2.0 
    return X, y.ravel()

def run_comparison():
    X, y = generate_data()
    DEGREE = 2
    
    print(f"--- COMPARISON TEST (Degree={DEGREE}) ---")

    sk_model = make_pipeline(
        SklearnPoly(degree=DEGREE, include_bias=False),
        SklearnLR(fit_intercept=True)
    )
    sk_model.fit(X, y)
    sk_pred = sk_model.predict(X)
    sk_intercept = sk_model.named_steps['linearregression'].intercept_
    sk_coef = sk_model.named_steps['linearregression'].coef_
    
    print(f"\n[Scikit-Learn]")
    print(f"Intercept : {sk_intercept:.6f}")
    print(f"Coef      : {sk_coef}")
    print(f"MSE       : {mean_squared_error(y, sk_pred):.6f}")


    my_model = MyPolyReg(degree=DEGREE, fit_intercept=True)
    my_model.fit(X, y)
    my_pred = my_model.predict(X)
    
    print(f"\n[My Library]")
    print(f"Intercept : {my_model.intercept_:.6f}")
    print(f"Coef      : {my_model.coef_}")
    print(f"MSE       : {mean_squared_error(y, my_pred):.6f}")
    
    print("\n[Validation Result]")
    
    diff_intercept = abs(sk_intercept - my_model.intercept_)
    diff_coef = np.sum(np.abs(sk_coef - my_model.coef_))
    diff_pred = np.sum(np.abs(sk_pred - my_pred))
    
    print(f"Diff Intercept : {diff_intercept:.15f}")
    print(f"Diff Coeffs    : {diff_coef:.15f}")
    print(f"Diff Predicts  : {diff_pred:.15f}")
    
    # Kiểm tra tên đặc trưng (Feature Names)
    print("\n[Feature Names Check]")
    try:
        names = my_model.get_feature_names_out(input_features=['x'])
        print(f"Output Features: {names}")

        equation = f"y = {my_model.intercept_:.2f}"
        for name, w in zip(names, my_model.coef_):
            equation += f" + ({w:.2f} * {name})"
        print(f"Equation: {equation}")
    except Exception as e:
        print(f"Error getting feature names: {e}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='black', label='Data')
    plt.plot(X, sk_pred, color='blue', linewidth=2, label='Scikit-Learn', linestyle='--')
    plt.plot(X, my_pred, color='red', linewidth=2, label='My Library', alpha=0.7)
    plt.title(f'Polynomial Regression Comparison (Degree {DEGREE})')
    plt.legend()
    plt.show()

    if diff_pred < 1e-10:
        print("\nSuccessfully!")
    else:
        print("\nFailed")

if __name__ == "__main__":
    run_comparison()