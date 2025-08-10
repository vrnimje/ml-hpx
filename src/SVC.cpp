#include "SVC.hpp"

double SVC::fit(std::vector<std::tuple<double, double, int>> D) {
    size_t n = D.size();

    for (int i = 0; i < max_iter; ++i) {
        hpx::for_each(
            hpx::execution::seq,
            D.begin(),
            D.end(),
            [&](auto a) {
                int y_i = std::get<2>(a);
                auto x_i = std::make_pair(std::get<0>(a), std::get<1>(a));

                if (y_i * h(x_i) < 1) {
                    W.first += alpha * y_i * x_i.first;
                    W.second += alpha * y_i * x_i.second;
                    B += alpha * y_i;
                }
        });
    }

    double train_accuracy = hpx::transform_reduce(
        hpx::execution::par,
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

double SVC::fit(std::vector<std::pair<double, double>> X, std::vector<int> Y) {
    std::vector<std::tuple<double, double, int>> D;

    for (size_t i = 0; i < X.size(); i++) {
        D.push_back(std::make_tuple(X[i].first, X[i].second, Y[i]));
    }

    return SVC::fit(D);
}
