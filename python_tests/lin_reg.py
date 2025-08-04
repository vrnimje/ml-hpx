from ml_hpx import LinearRegression
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('./datasets/linear_regressor_dataset_10000.csv')
X = df['x']
Y = df['y']

X_values = X.values
Y_values = Y.values

# ML-HPX Linear Regression
lin_reg = LinearRegression(5000, 1e-5)

lin_reg.fit(X_values.tolist(), Y_values.tolist())
y_pred_hpx = lin_reg.predict(X_values.tolist())

hpx_mse = mean_squared_error(Y_values, y_pred_hpx)

print(f"ML-HPX Linear Regression MSE: {hpx_mse:.4f}")

# sklearn Linear Regression
sklearn_lin_reg = SklearnLinearRegression()

sklearn_lin_reg.fit(X_values.reshape(-1, 1), Y_values)
y_pred_sklearn = sklearn_lin_reg.predict(X_values.reshape(-1, 1))

sklearn_mse = mean_squared_error(Y_values, y_pred_sklearn)

print(f"Sklearn Linear Regression MSE: {sklearn_mse:.4f}")
