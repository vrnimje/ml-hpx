#ifndef NeuralNetwork_hpp_
#define NeuralNetwork_hpp_

#include "Layer.hpp"
#include "Optimizer.hpp"

class NeuralNetwork {
private:
    std::vector<Layer*> layers;
    Optimizer* optimizer;

public:
    NeuralNetwork(std::vector<Layer*> layers, Optimizer* optimizer) {
        this->layers = layers;
        this->optimizer = optimizer;
    }

    void fit(const std::vector<std::vector<double>>& inputs,
             const std::vector<std::vector<double>>& targets,
             int epochs);

    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& inputs);
};

#endif
