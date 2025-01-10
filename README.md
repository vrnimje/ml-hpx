# ML - HPX

Implementation of common ML algorithms from scratch, and improving their runtime using HPX.

## Build Instructions
Pre-requisites: [Installing HPX](https://hpx-docs.stellar-group.org/latest/html/quickstart.html)

```sh
cmake -S . -Bbuild -GNinja
cmake --build build
./build/linear_reg [path_to_dataset]
```

## Algorithms

### 1. Simple Linear Regression (One Variable), using Gradient Descent
[linear_reg.cpp](./linear_reg.cpp)

- Linear Regression GD, seq for-loop: Used a classic for loop for performing Gradient Descent (GD)
- Linear Regression GD, STL, seq: Using [`std::transform_reduce`](https://en.cppreference.com/w/cpp/algorithm/transform_reduce)
- Linear Regression GD, STL, par: Using `std::transform_reduce` with [`std::execution::par`](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag) execution policy
- Linear Regression GD, HPX, par: Using [`hpx::transform_reduce`](https://hpx-docs.stellar-group.org/latest/html/libs/core/algorithms/api/transform_reduce.html) with [`hpx::execution::par`](https://hpx-docs.stellar-group.org/branches/master/html/libs/core/executors/api/execution_policy.html) execution policy

Results with [dataset containing 10000 points](./datasets/linear_regressor_dataset_10000.csv):
```sh
Final Parameters: W = 2.530294, B = 3.038263
Final Parameters: W = 2.558153, B = 0.183712
Final Parameters: W = 2.558576, B = 0.155537
Final Parameters: W = 2.559148, B = 0.117489

Results:

name: Sequential GD, for-loop,
executor: seq,
average: 0.742246513333333


name: Linear Regression GD, STL, seq,
executor: seq,
average: 0.976717908291667


name: Linear Regression, GD, STL, par,
executor: std::execution::par,
average: 0.975341845875


name: Linear Regression, GD, HPX, par,
executor: hpx::execution::par,
average: 0.62639202725
```

Plot:
<img src="./assets/linear_reg.png" width="100%">
