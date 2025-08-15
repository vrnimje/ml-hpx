from ml_hpx import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.preprocessing import StandardScaler
from time import perf_counter

print("### Simple Logistic Regression")
# Load data
df = pd.read_csv('./datasets/logistic_regression_dataset_10000.csv')
X = df[['x']]
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_flat = X_train.tolist()
X_test_flat = X_test.tolist()
y_train_vals = y_train.values

# ML-HPX Logistic Regression
log_reg = LogisticRegression(5000, 0.005, 123)

start_hpx = perf_counter()
log_reg.fit(X_train_flat, y_train_vals)
y_pred_hpx = log_reg.predict(X_test_flat)
end_hpx = perf_counter()

hpx_accuracy = accuracy_score(y_test, y_pred_hpx)

print(f"ML-HPX Logistic Regression accuracy: {hpx_accuracy:.4f}")
print(f"ML-HPX Logistic Regression time: {end_hpx - start_hpx:.4f} seconds")

# sklearn Logistic Regression
sklearn_model = SklearnLogReg()

start_sklearn = perf_counter()
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)
end_sklearn = perf_counter()

sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)

print(f"Sklearn Logistic Regression accuracy: {sklearn_accuracy:.4f}")
print(f"Sklearn Logistic Regression time: {end_sklearn - start_sklearn:.4f} seconds")

print("\n### Multi-feature Logistic Regression")

# Load multi-feature dataset
df_multi = pd.read_csv('./datasets/multi-feature-log-regression.csv')
X_multi = df_multi.drop(columns=['y'])
y_multi = df_multi['y']

# Split into train/test
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Standardize features
scaler_multi = StandardScaler()
X_train_multi = scaler_multi.fit_transform(X_train_multi)
X_test_multi = scaler_multi.transform(X_test_multi)

X_train_multi_list = X_train_multi.tolist()
X_test_multi_list = X_test_multi.tolist()
y_train_multi_vals = y_train_multi.values

# ML-HPX Logistic Regression
log_reg_multi = LogisticRegression(5000, 0.005, 123)

start_hpx_multi = perf_counter()
log_reg_multi.fit(X_train_multi_list, y_train_multi_vals)
y_pred_multi_hpx = log_reg_multi.predict(X_test_multi_list)
end_hpx_multi = perf_counter()

hpx_accuracy_multi = accuracy_score(y_test_multi, y_pred_multi_hpx)

print(f"ML-HPX Logistic Regression accuracy: {hpx_accuracy_multi:.4f}")
print(f"ML-HPX Logistic Regression time: {end_hpx_multi - start_hpx_multi:.4f} seconds")

# sklearn Logistic Regression
sklearn_model_multi = SklearnLogReg()

start_sklearn_multi = perf_counter()
sklearn_model_multi.fit(X_train_multi, y_train_multi)
y_pred_multi_sklearn = sklearn_model_multi.predict(X_test_multi)
end_sklearn_multi = perf_counter()

sklearn_accuracy_multi = accuracy_score(y_test_multi, y_pred_multi_sklearn)

print(f"Sklearn Logistic Regression accuracy: {sklearn_accuracy_multi:.4f}")
print(f"Sklearn Logistic Regression time: {end_sklearn_multi - start_sklearn_multi:.4f} seconds")
