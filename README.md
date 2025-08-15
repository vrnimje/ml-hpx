# ML - HPX

Implementation of common ML algorithms from scratch in C++, utilisng HPX as the runtime and providing bindings for them in Python using Nanobind.

## Build Instructions
Pre-requisites: [Installing HPX](https://hpx-docs.stellar-group.org/latest/html/quickstart.html)

```sh
cd build
git clone https://github.com/vrnimje/ml-hpx.git

cd ml-hpx
pip install .
```

### Utilising the bindings

1. Linear Regression

[Script](./python_tests/lin_reg.py):

```py
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
```

```sh
$ python python_tests/lin_reg.py
### Simple Linear Regression
ML-HPX Linear Regression MSE: 51.9648
ML-HPX Linear Regression Time: 1.8472 seconds
Sklearn Linear Regression MSE: 99.7971
Sklearn Linear Regression Time: 0.0022 seconds

### Multi-feature Linear Regression
ML-HPX Linear Regression MSE: 15.8709
ML-HPX Linear Regression Time: 1.2715 seconds
Sklearn Linear Regression MSE: 25.0206
Sklearn Linear Regression Time: 0.0029 seconds
```

As `scikit-learn` uses a closed-form solution to solve the linear regression , while `ml-hpx` uses an iterative optimization algorithm, hence, we see these large performance differences.

2. Logistic Regression

[Script](./python_tests/log_reg.py):

```py
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
```

```sh
$ python python_tests/log_reg.py
### Simple Logistic Regression
ML-HPX Logistic Regression accuracy: 0.9820
ML-HPX Logistic Regression time: 1.2471 seconds
Sklearn Logistic Regression accuracy: 0.9830
Sklearn Logistic Regression time: 0.0279 seconds

### Multi-feature Logistic Regression
ML-HPX Logistic Regression accuracy: 0.9530
ML-HPX Logistic Regression time: 1.0675 seconds
Sklearn Logistic Regression accuracy: 0.9535
Sklearn Logistic Regression time: 0.0142 seconds
```

3. K-Nearest Neighbours classifier

[Script](./python_tests/knn.py):

```py
from ml_hpx import KNearestNeighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('./datasets/classified_points_dataset.csv')

X = df[['x', 'y']]
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNearestNeighbours(k=3)

knn.fit(X_train.tolist(), y_train.values)

y_pred = knn.predict(X_test.tolist())

print("ML-HPX accuracy", accuracy_score(y_pred, y_test))

knn_sklearn = KNeighborsClassifier(n_neighbors=3)

knn_sklearn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Sklearn accuracy", accuracy_score(y_pred, y_test))
```

```sh
$ python python_tests/knn.py
ML-HPX accuracy: 1.0000
ML-HPX time: 0.1780 seconds
Sklearn accuracy: 1.0000
Sklearn time: 0.0090 seconds
```

4. k-Means Clustering

[Script](./python_tests/kmeans.py):

```py
from ml_hpx import KMeansClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from time import perf_counter

# Load dataset
df = pd.read_csv('./datasets/kmeans_test_dataset.csv')

X = df[['x', 'y']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### scikit-learn KMeans ###
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)

start_kmeans = perf_counter()
kmeans.fit(X_scaled)
end_kmeans = perf_counter()

# Add cluster labels to DataFrame
df['cluster'] = kmeans.labels_

# Print performance
print(f"Sklearn KMeans time: {end_kmeans - start_kmeans:.4f} seconds")
print(f"Inertia: {kmeans.inertia_:.4f}")

kmeans_hpx = KMeansClustering(k=5)

start_hpx = perf_counter()
sse = kmeans_hpx.fit(X_scaled.tolist())
end_hpx = perf_counter()

print(f"ML-HPX KMeans time: {end_hpx - start_hpx:.4f} seconds")
print(f"Inertia: {sse:.4f}")
```

```sh
$ python python_tests/kmeans.py
Sklearn KMeans time: 0.0140 seconds
Inertia: 543.3955
ML-HPX KMeans time: 0.0085 seconds
Inertia: 543.3952
```

5. Perceptron classifier

[Script](./python_tests/perceptron.py):

```py
from ml_hpx import Perceptron
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.preprocessing import StandardScaler
from time import perf_counter

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

X_train_flat = X_train.flatten().tolist()
X_test_flat = X_test.flatten().tolist()
y_train_vals = y_train.values

# ML-HPX Logistic Regression
perc = Perceptron()

start_hpx = perf_counter()
perc.fit(X_train_flat, y_train_vals)
y_pred_hpx = perc.predict(X_test_flat)
end_hpx = perf_counter()

hpx_accuracy = accuracy_score(y_test, y_pred_hpx)

print(f"ML-HPX Logistic Regression accuracy: {hpx_accuracy:.4f}")
print(f"ML-HPX Logistic Regression time: {end_hpx - start_hpx:.4f} seconds")

# sklearn Logistic Regression
sklearn_model = SklearnPerceptron()

start_sklearn = perf_counter()
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)
end_sklearn = perf_counter()

sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)

print(f"Sklearn Logistic Regression accuracy: {sklearn_accuracy:.4f}")
print(f"Sklearn Logistic Regression time: {end_sklearn - start_sklearn:.4f} seconds")
```

```sh
$ python python_tests/perceptron.py
ML-HPX Perceptron accuracy: 1.0000
ML-HPX Perceptron time: 0.0016 seconds
Sklearn Perceptron accuracy: 0.9995
Sklearn Perceptron time: 0.0028 seconds
```

6. SVM Classifier

[Script](./python_tests/svc.py):

```py
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
```

```sh
$ python python_tests/svc.py
ML-HPX SVC accuracy: 1.0000
ML-HPX SVC time: 0.0220 seconds
Sklearn SVC accuracy: 0.9970
Sklearn SVC time: 0.1196 seconds
```

## Misc. Benchmarks

### 1. Simple Linear Regression (One Variable), using Gradient Descent
[linear_reg.cpp](./linear_reg.cpp)

<img src="https://miro.medium.com/v2/resize:fit:900/1*G3evFxIAlDchOx5Wl7bV5g.png" width="100%">

- Linear Regression GD, seq for-loop: Used a classic for loop for performing Gradient Descent (GD)
- Linear Regression GD, STL, seq: Using [`std::transform_reduce`](https://en.cppreference.com/w/cpp/algorithm/transform_reduce)
- Linear Regression GD, STL, par: Using `std::transform_reduce` with [`std::execution::par`](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag) execution policy
- Linear Regression GD, HPX, par: Using [`hpx::transform_reduce`](https://hpx-docs.stellar-group.org/latest/html/libs/core/algorithms/api/transform_reduce.html) with [`hpx::execution::par`](https://hpx-docs.stellar-group.org/branches/master/html/libs/core/executors/api/execution_policy.html) execution policy

Results with [dataset containing 10000 points](./datasets/linear_regressor_dataset_10000.csv):
```sh
$ ./build/linear_reg datasets/linear_regressor_dataset_10000.csv
Final Parameters: W = 2.577991, B = -0.136139
Final Parameters: W = 2.514959, B = 3.057930
Final Parameters: W = 2.530195, B = 2.043939
Final Parameters: W = 2.523242, B = 2.506657
Results:

name: Sequential GD, for-loop,
executor: seq,
average: 0.72105570925


name: Linear Regression GD, STL, seq,
executor: seq  ,
average: 0.961342297166667


name: Linear Regression, GD, STL, par,
executor: std::execution::par,
average: 0.945831261333333


name: Linear Regression, GD, HPX, par,
executor: hpx::execution::par,
average: 0.51301071275

```

Note: Execution time of each implementation is averaged for 25 runs

Plot:

<img src="./assets/linear_reg.png" width="100%">

### 2. Simple Logistic Regression (One Variable), using Gradient Descent
[log_reg.cpp](./log_reg.cpp)

Results with [dataset containing 10000 points](./datasets/logistic_regression_dataset_10000.csv):

```sh
Results:

name: Sequential GD, for-loop
executor: seq
average: 9.99655853479999999983e-01
Final Parameters: W = 0.129403, B = -0.390624
Accuracy: 0.976667

name: Logistic Regression GD, STL, seq
executor: seq
average: 1.01029237284000000005e+00
Final Parameters: W = 0.001630, B = -0.004954
Accuracy: 0.976333

name: Logistic Regression, GD, STL, par
executor: std::execution::par
average: 1.01217441688000000003e+00
Final Parameters: W = 0.105150, B = -0.318801
Accuracy: 0.976333

name: Logistic Regression, GD, HPX, par
executor: hpx::execution::par
average: 6.61743677679999999972e-01
Final Parameters: W = 0.002196, B = -0.006469
Accuracy: 0.977667
```

And the accuracy is comparable to [scikit-learn](./log_reg.py)

```sh
$ python3 log_reg.py
Accuracy: 0.98
```
