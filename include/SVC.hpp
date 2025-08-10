#include <cmath>
#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>

#include <random>
#include <vector>

class SVC {
private:
    // Hyperplane: W1 * x_1 + W2 * x_2 + B = 0
    std::pair<double, double> W;
    double B;
    double alpha; // Learning rate
    double C; // Regularization parameter

    int max_iter;

    std::mt19937 gen;
    std::uniform_real_distribution<double> distribution;

    // Linear equation: Y = W*X + B
    double h(std::pair<double, double> X) const {
        return (W.first * X.first) + (W.second * X.second) + B;
    }

public:
    SVC(int max_iter = 1000, double learning_rate = 1e-3, unsigned int seed = 0)
        :   gen(seed),                  // 1. Seed the generator using the device
            distribution(-1.0, 1.0),    // 2. Initialize the distribution range
            W({distribution(gen), distribution(gen)}), // 3. Initialize W with a random value
            B(distribution(gen)),       // 4. Initialize B with a random value
            alpha(learning_rate),
            max_iter(max_iter)
    {}

    double predict(std::pair<double, double> X) const {
        return (h(X) > 0.0) ? 1 : -1;
    }

    std::vector<int> predict(std::vector<std::pair<double, double>> X) const {
        std::vector<int> Y;
        for (auto& x : X) {
            Y.push_back(predict(x));
        }
        return Y;
    }

    double fit(std::vector<std::tuple<double, double, int>> D);

    double fit(std::vector<std::pair<double, double>> X, std::vector<int> Y);
};
