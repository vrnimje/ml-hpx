#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>

#include <random>
#include <vector>

class LinearRegression {
private:
    double W, B; // Weights
    double alpha; // Learning rate
    int epochs; // Epochs

    std::mt19937 gen;
    std::uniform_real_distribution<double> distribution;

public:
    LinearRegression(int num_epochs = 5000, double learning_rate = 1e-3, unsigned int seed = 0)
        :   gen(seed),                  // 1. Seed the generator using the device
            distribution(-1.0f, 1.0f),  // 2. Initialize the distribution range
            W(distribution(gen)),       // 3. Initialize W with a random value
            B(distribution(gen)),       // 4. Initialize B with a random value (fixed)
            alpha(learning_rate),
            epochs(num_epochs)
    {}

    // Linear equation: Y = WX + B
    double f(double X) const {
        return (W*X + B);
    }

    double predict(double X) const {
        return f(X);
    }

    std::vector<double> predict(std::vector<double> X) const {
        std::vector<double> Y;
        for (auto& x : X) {
            Y.push_back(f(x));
        }
        return Y;
    }

    double fit(std::vector<std::pair<double, double>> D);

    double fit(std::vector<double> X, std::vector<double> Y);
};
