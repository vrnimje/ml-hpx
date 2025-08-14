#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>

#include <random>
#include <vector>

using gradient_pair = std::pair<std::vector<double>, double>;

class LinearRegression {
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
    LinearRegression(int num_epochs = 5000, double learning_rate = 1e-3, unsigned int seed = 0)
        :   gen(seed),                  // 1. Seed the generator using the device
            distribution(-1.0, 1.0),    // 2. Initialize the distribution range
            alpha(learning_rate),
            epochs(num_epochs),
            is_init(false)
    {}

    // Y = (W . X) + B
    double predict(const std::vector<double>& X);

    std::vector<double> predict(std::vector<std::vector<double>>& X);

    double fit(std::vector<std::vector<double>> X, std::vector<double> Y);
};
