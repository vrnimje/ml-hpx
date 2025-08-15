#include <cmath>
#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>

#include <random>
#include <vector>

using gradient_pair = std::pair<std::vector<double>, double>;

class LogisticRegression {
private:
    std::vector<double> W; // Weights
    double B; // Bias
    double alpha; // Learning rate
    int epochs; // Epochs
    int num_features; // Number of features

    std::mt19937 gen;
    std::uniform_real_distribution<double> distribution;

    bool is_init;

    void init_weights(size_t features) {
        num_features = features;
        W.resize(num_features);
        // Initialize weights with small random values
        for (size_t i = 0; i < num_features; ++i) {
            W[i] = distribution(gen);
        }
        B = distribution(gen);
    }

public:
    LogisticRegression(int num_epochs = 5000, double learning_rate = 1e-3, unsigned int seed = 0)
        :   gen(seed),                  // 1. Seed the generator using the device
            distribution(-1.0, 1.0),    // 2. Initialize the distribution range
            alpha(learning_rate),
            epochs(num_epochs),
            is_init(false)
    {}

    // Y = 1 / (1 + exp(-(W . X) - B))
    double f(const std::vector<double>& X, bool is_hpx);

    int predict(const std::vector<double>& X, bool is_hpx);

    std::vector<int> predict(const std::vector<std::vector<double>>& X);

    double fit(std::vector<std::vector<double>> X, std::vector<int> Y);
};
