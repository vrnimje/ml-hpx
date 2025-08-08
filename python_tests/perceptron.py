from ml_hpx import Perceptron
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.preprocessing import StandardScaler
from time import perf_counter

# Load data
df = pd.read_csv('./datasets/perceptron_test_dataset.csv')
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

# ML-HPX Perceptron
perceptron = Perceptron()

start_hpx = perf_counter()
perceptron.fit(X_train_flat, y_train_vals)
y_pred_hpx = perceptron.predict(X_test_flat)
end_hpx = perf_counter()

hpx_accuracy = accuracy_score(y_test, y_pred_hpx)

print(f"ML-HPX Perceptron accuracy: {hpx_accuracy:.4f}")
print(f"ML-HPX Perceptron time: {end_hpx - start_hpx:.4f} seconds")

# sklearn Perceptron
sklearn_model = SklearnPerceptron()

start_sklearn = perf_counter()
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)
end_sklearn = perf_counter()

sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)

print(f"Sklearn Perceptron accuracy: {sklearn_accuracy:.4f}")
print(f"Sklearn Perceptron time: {end_sklearn - start_sklearn:.4f} seconds")
