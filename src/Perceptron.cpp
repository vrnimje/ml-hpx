#include "Perceptron.hpp"

double Perceptron::fit(std::vector<std::tuple<double, double, int>> D) {
    size_t n = D.size();

    hpx::for_each(
        D.begin(),
        D.end(),
        [&](auto a) {
            double diff = (predict(std::make_pair(std::get<0>(a), std::get<1>(a))) - std::get<2>(a));
            W.first -= alpha * diff * std::get<0>(a);
            W.second -= alpha * diff * std::get<1>(a);
            B -= alpha * diff;
        }
    );

    double train_accuracy = hpx::transform_reduce(
        D.begin(),
        D.end(),
        0.0,
        std::plus<int>(),
        [&](auto a) {
            return (predict(std::make_pair(std::get<0>(a), std::get<1>(a))) == std::get<2>(a)) ? 1 : 0;
        }
    );

    train_accuracy /= n;

    return train_accuracy;
}

double Perceptron::fit(std::vector<std::pair<double, double>> X, std::vector<int> Y) {
    std::vector<std::tuple<double, double, int>> D;

    for (size_t i = 0; i < X.size(); i++) {
        D.push_back(std::make_tuple(X[i].first, X[i].second, Y[i]));
    }

    return Perceptron::fit(D);
}
