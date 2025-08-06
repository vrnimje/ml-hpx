from ml_hpx import KNearestNeighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from time import perf_counter

# Load dataset
df = pd.read_csv('./datasets/classified_points_dataset.csv')

X = df[['x', 'y']]
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data for HPX version
X_train_list = X_train.tolist()
X_test_list = X_test.tolist()
y_train_values = y_train.values

### ML-HPX ###
knn = KNearestNeighbours(k=5)

start_hpx = perf_counter()
knn.fit(X_train_list, y_train_values)
y_pred_hpx = knn.predict(X_test_list)
end_hpx = perf_counter()

hpx_accuracy = accuracy_score(y_pred_hpx, y_test)

print(f"ML-HPX accuracy: {hpx_accuracy:.4f}")
print(f"ML-HPX time: {end_hpx - start_hpx:.4f} seconds")

### scikit-learn ###
knn_sklearn = KNeighborsClassifier(n_neighbors=5)

start_sklearn = perf_counter()
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)
end_sklearn = perf_counter()

sklearn_accuracy = accuracy_score(y_pred_sklearn, y_test)

print(f"Sklearn accuracy: {sklearn_accuracy:.4f}")
print(f"Sklearn time: {end_sklearn - start_sklearn:.4f} seconds")
