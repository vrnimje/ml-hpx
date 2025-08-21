#ifndef Optimizer_hpp_
#define Optimizer_hpp_

#include <hpx/parallel/algorithm.hpp>
#include <vector>

class Optimizer {
public:
    virtual void update(std::vector<double>& weights,
                        std::vector<double>& biases,
                        const std::vector<double>& dW,
                        const std::vector<double>& dB) = 0;

    virtual ~Optimizer() = default;
};

class SGD : public Optimizer {
private:
    double lr;
public:
    SGD(double lr) : lr(lr) {}

    void update(std::vector<double>& weights, std::vector<double>& biases, const std::vector<double>& dW, const std::vector<double>& dB) override {
        hpx::transform(
            dW.begin(), dW.end(), weights.begin(), weights.begin(), [&](double dw, double w) {
                return w - lr * dw;
            }
        );

        hpx::transform(
            dB.begin(), dB.end(), biases.begin(), biases.begin(), [&](double db, double b) {
                return b - lr * db;
            }
        );
    }
};

#endif
