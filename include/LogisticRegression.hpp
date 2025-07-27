#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>

#include <random>
#include <vector>

class LogisticRegression {
private:
    float W, B; // Weights
    float alpha; // Learning rate
    int epochs; // Epochs

    std::mt19937 gen;
    std::uniform_real_distribution<float> distribution;

public:
    LogisticRegression(int num_epochs = 5000, float learning_rate = 1e-3, unsigned int seed = 0)
        :   gen(seed),                  // 1. Seed the generator using the device
            distribution(-1.0f, 1.0f),  // 2. Initialize the distribution range
            W(distribution(gen)),       // 3. Initialize W with a random value
            B(distribution(gen)),       // 4. Initialize B with a random value (fixed)
            alpha(learning_rate),
            epochs(num_epochs)
    {}

    // Linear equation: Y = 1 / (1 + exp(-Wx - B))
    float f(float X) const {
        return 1 / (1 + std::exp(-1 * (W*X + B)));
    }

    float predict(float X) const {
        return (f(X) > 0.5) ? 1 : 0;
    }

    std::vector<int> predict(std::vector<float> X) const {
        std::vector<int> Y;
        for (auto& x : X) {
            Y.push_back((f(x) > 0.5) ? 1 : 0);
        }
        return Y;
    }

    float train(std::vector<std::pair<float, int>> D);
};
