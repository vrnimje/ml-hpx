from ml_hpx import LinearRegression
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error
from time import perf_counter

print("### Simple Linear Regression")
# Load data
df = pd.read_csv('./datasets/linear_regressor_dataset_10000.csv')
X = df[['x']]
Y = df['y']

X_values = X.values
Y_values = Y.values

# ML-HPX Linear Regression
lin_reg = LinearRegression(5000, 1e-5)

start_hpx = perf_counter()
hpx_mse = lin_reg.fit(X_values.tolist(), Y_values.tolist())

y_pred_hpx = lin_reg.predict(X_values.tolist())

end_hpx = perf_counter()

print(f"ML-HPX Linear Regression MSE: {hpx_mse:.4f}")
print(f"ML-HPX Linear Regression Time: {end_hpx - start_hpx:.4f} seconds")

# sklearn Linear Regression
sklearn_lin_reg = SklearnLinearRegression()

start_sklearn = perf_counter()
sklearn_lin_reg.fit(X_values, Y_values)
y_pred_sklearn = sklearn_lin_reg.predict(X_values)

sklearn_mse = mean_squared_error(Y_values, y_pred_sklearn)
end_sklearn = perf_counter()

print(f"Sklearn Linear Regression MSE: {sklearn_mse:.4f}")
print(f"Sklearn Linear Regression Time: {end_sklearn - start_sklearn:.4f} seconds")

print("\n### Multi-feature Linear Regression")

# Load data
df = pd.read_csv('./datasets/multi-feature-linear-regression.csv')

# Use all feature columns
X = df.drop(columns=['y'])
Y = df['y']

X_values = X.values
Y_values = Y.values

# ML-HPX Linear Regression
# Here, 5000 could be the number of iterations, and 1e-5 the learning rate (depending on implementation)
lin_reg = LinearRegression(5000, 1e-5)

start_hpx = perf_counter()
hpx_mse = lin_reg.fit(X_values.tolist(), Y_values.tolist())
y_pred_hpx = lin_reg.predict(X_values.tolist())
end_hpx = perf_counter()

print(f"ML-HPX Linear Regression MSE: {hpx_mse:.4f}")
print(f"ML-HPX Linear Regression Time: {end_hpx - start_hpx:.4f} seconds")

# sklearn Linear Regression
sklearn_lin_reg = SklearnLinearRegression()

start_sklearn = perf_counter()
sklearn_lin_reg.fit(X_values, Y_values)
y_pred_sklearn = sklearn_lin_reg.predict(X_values)
sklearn_mse = mean_squared_error(Y_values, y_pred_sklearn)
end_sklearn = perf_counter()

print(f"Sklearn Linear Regression MSE: {sklearn_mse:.4f}")
print(f"Sklearn Linear Regression Time: {end_sklearn - start_sklearn:.4f} seconds")
