from ml_hpx import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as SklearnSVC
from sklearn.preprocessing import StandardScaler
from time import perf_counter

# Load data
df = pd.read_csv('./datasets/svc_test_dataset.csv')
X = df[['x', 'y']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_flat = X_train.tolist()
X_test_flat = X_test.tolist()
y_train_vals = y_train.values

# ML-HPX SVC
svc = SVC()

start_hpx = perf_counter()
svc.fit(X_train_flat, y_train_vals)
y_pred_hpx = svc.predict(X_test_flat)
# print(x)
end_hpx = perf_counter()

hpx_accuracy = accuracy_score(y_test, y_pred_hpx)

print(f"ML-HPX SVC accuracy: {hpx_accuracy:.4f}")
print(f"ML-HPX SVC time: {end_hpx - start_hpx:.4f} seconds")

# sklearn SVC
sklearn_model = SklearnSVC()

start_sklearn = perf_counter()
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)
end_sklearn = perf_counter()

sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)
print(f"Sklearn SVC accuracy: {sklearn_accuracy:.4f}")
print(f"Sklearn SVC time: {end_sklearn - start_sklearn:.4f} seconds")
