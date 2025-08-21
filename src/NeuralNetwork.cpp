#include "NeuralNetwork.hpp"

void NeuralNetwork::fit(const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets, int epochs) {

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;

        for (size_t n = 0; n < inputs.size(); n++) {
            // ---- Forward Pass ----
            std::vector<std::vector<double>> activations;
            activations.push_back(inputs[n]);

            for (auto& layer : layers) {
                activations.push_back(layer->forward(activations.back()));
            }

            auto prediction = activations.back();

            // Loss calculation (MSE)
            std::vector<double> dY(prediction.size());
            double sample_loss = 0.0;
            for (size_t i = 0; i < prediction.size(); i++) {
                double diff = prediction[i] - targets[n][i];
                dY[i] = diff;  // derivative of MSE wrt output
                sample_loss += diff * diff;
            }
            epoch_loss += sample_loss / prediction.size();

            // ---- Backward Pass ----
            for (int l = layers.size() - 1; l >= 0; --l) {
                auto grads = layers[l]->backward(
                    activations[l],
                    activations[l+1], dY
                );
                optimizer->update(layers[l]->weights,
                    layers[l]->biases,
                    grads.dW, grads.dB
                );
                // propogate backwards
                dY = grads.dX;
            }
        }
    }
}

std::vector<std::vector<double>> NeuralNetwork::predict(const std::vector<std::vector<double>>& inputs) {
    std::vector<std::vector<double>> predictions;
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> activations = inputs[i];

        for (auto& layer : layers) {
            activations = layer->forward(activations);
        }

        predictions.push_back(activations);
    }
    return predictions;
}
