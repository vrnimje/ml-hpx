#ifndef Activation_hpp_
#define Activation_hpp_

#include <cmath>
#include <functional>

// Activation Functions
struct Activation {
    std::function<double(double)> func;
    std::function<double(double)> deriv;

    static Activation ReLU() {
        return {
            [](double x) { return x > 0 ? x : 0; },
            [](double x) { return x > 0 ? 1.0 : 0.0; }
        };
    }

    static Activation Sigmoid() {
        return {
            [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
            [](double y) {
                return y * (1 - y);
            }
        };
    }

    static Activation Tanh() {
        return {
            [](double x) { return std::tanh(x); },
            [](double y) { return 1 - y * y; }
        };
    }
};
#endif
