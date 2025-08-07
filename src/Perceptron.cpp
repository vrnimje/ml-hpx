#include "Perceptron.hpp"

double Perceptron::fit(std::vector<std::pair<double, int>> D) {
    size_t n = D.size();

    hpx::for_each(
        D.begin(),
        D.end(),
        [&](auto a) {
            double diff = (predict(a.first) - a.second);
            W -= alpha * diff * a.first;
            B -= alpha * diff;
        }
    );

    double train_accuracy = hpx::transform_reduce(
        D.begin(),
        D.end(),
        0.0,
        std::plus<int>(),
        [&](auto a) {
            return (predict(a.first) == a.second) ? 1 : 0;
        }
    );

    train_accuracy /= n;

    return train_accuracy;
}

double Perceptron::fit(std::vector<double> X, std::vector<int> Y) {
    std::vector<std::pair<double, int>> D;

    for (size_t i = 0; i < X.size(); i++) {
        D.push_back(std::make_pair(X[i], Y[i]));
    }

    return Perceptron::fit(D);
}
