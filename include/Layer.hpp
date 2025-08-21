#ifndef Layer_hpp_
#define Layer_hpp_

#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>

#include <random>
#include <vector>

#include "Activation.hpp"

struct Gradients {
    std::vector<double> dW;   // same size as weights
    std::vector<double> dB;   // same size as biases
    std::vector<double> dX;   // same size as input (for previous layer)
};

class Layer {
private:
    int num_neurons, num_inputs;

    // i + n*j indexing
    std::vector<double> weights;
    std::vector<double> biases;

    Activation activation_function;

public:
    Layer(int num_neurons, int num_inputs, std::string activation_function)
        : num_neurons(num_neurons), num_inputs(num_inputs) {
        if (activation_function == "sigmoid") {
            this->activation_function = Activation::Sigmoid();
        }
        else if (activation_function == "relu") {
            this->activation_function = Activation::ReLU();
        }
        else if (activation_function == "tanh") {
            this->activation_function = Activation::Tanh();
        }
        else {
            throw std::invalid_argument("Invalid activation function");
        }

        // Random initialization (Xavier)
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = std::sqrt(6.0 / (num_inputs + num_neurons));
        std::uniform_real_distribution<double> dist(-limit, limit);

        weights.resize(num_neurons * num_inputs);
        biases.resize(num_neurons);

        hpx::generate(weights.begin(), weights.end(), [&] { return dist(gen); });
        hpx::generate(biases.begin(), biases.end(), [&] { return dist(gen); });
    }

    std::vector<double> forward(const std::vector<double>& input);

    Gradients backward(const std::vector<double>& input,
                       const std::vector<double>& output,
                       const std::vector<double>& dY);

    friend class NeuralNetwork;
};
#endif
