#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>

#include <random>
#include <vector>

class LinearRegression {
private:
    float W, B; // Weights
    float alpha; // Learning rate
    int epochs; // Epochs

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen;
    std::uniform_real_distribution<float> distribution;

public:
    LinearRegression(int num_epochs = 5000, float learning_rate = 1e-3)
        :   rd(),                       // 1. Initialize the random device
            gen(rd()),                  // 2. Seed the generator using the device
            distribution(-1.0f, 1.0f),  // 3. Initialize the distribution range
            W(distribution(gen)),       // 4. Initialize W with a random value
            B(distribution(gen)),       // 5. Initialize B with a random value (fixed)
            alpha(learning_rate),
            epochs(num_epochs)
    {}

    // Linear equation: Y = WX + B
    float f(float X) const {
        return (W*X + B);
    }

    float predict(float X) const {
        return f(X);
    }

    std::vector<float> predict(std::vector<float> X) const {
        std::vector<float> Y;
        for (auto& x : X) {
            Y.push_back(f(x));
        }
        return Y;
    }

    float train(std::vector<std::pair<float, float>> D);
};
