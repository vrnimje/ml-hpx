#include "Layer.hpp"

std::vector<double> Layer::forward(const std::vector<double>& input) {
    std::vector<double> output(num_neurons);

    for (int i = 0; i < num_neurons; ++i) {
        double sum = 0.0;
        sum = hpx::transform_reduce(input.begin(), input.end(),
            weights.begin() + (i * num_inputs), 0.0, std::plus<double>(), std::multiplies<double>());
        sum += biases[i];
        output[i] = activation_function.func(sum);
    }

    return std::move(output);
}

Gradients Layer::backward(const std::vector<double>& input,
                   const std::vector<double>& output,
                   const std::vector<double>& dY)
{
    Gradients grads;
    grads.dW.resize(weights.size());
    grads.dB.resize(biases.size());
    grads.dX.resize(input.size());

    // loop per neuron
    for (int j = 0; j < num_neurons; j++) {
        double delta = dY[j] * activation_function.deriv(output[j]);  // local gradient

        // bias grad
        grads.dB[j] = delta;

        // weight grads
        auto w_begin = grads.dW.begin() + j * num_inputs;

        hpx::transform(
            input.begin(), input.end(),
            w_begin, [&](double x) {
                return delta * x;
            }
        );

        for (int i = 0; i < num_inputs; i++) {
            grads.dX[i] += weights[j * num_inputs + i] * delta;
        }
    }
    return grads;
}
