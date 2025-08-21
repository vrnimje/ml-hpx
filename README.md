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

```sh
$ python python_tests/knn.py
ML-HPX accuracy: 1.0000
ML-HPX time: 0.1780 seconds
Sklearn accuracy: 1.0000
Sklearn time: 0.0090 seconds
```

4. k-Means Clustering

[Script](./python_tests/kmeans.py):

5. Perceptron classifier

[Script](./python_tests/perceptron.py):

```sh
$ python python_tests/perceptron.py
ML-HPX Perceptron accuracy: 1.0000
ML-HPX Perceptron time: 0.0016 seconds
Sklearn Perceptron accuracy: 0.9995
Sklearn Perceptron time: 0.0028 seconds
```

6. SVM Classifier

[Script](./python_tests/svc.py):

```sh
$ python python_tests/svc.py
ML-HPX SVC accuracy: 1.0000
ML-HPX SVC time: 0.0220 seconds
Sklearn SVC accuracy: 0.9970
Sklearn SVC time: 0.1196 seconds
```

7. Neural Network

[Script](./python_tests/nn.py):

```sh
$ python python_tests/nn.py
HPX NN training time: 0.001217 sec
HPX Predictions: [[0.06723812758653255], [0.9431358899595929], [0.9532029734896459], [0.050276559281559836]]
TensorFlow training time: 19.081340 sec
TF Predictions: [[0.5016673803329468], [0.9870261549949646], [0.5016673803329468], [0.010493488050997257]]
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
