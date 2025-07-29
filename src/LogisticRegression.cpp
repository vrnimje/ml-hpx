#include "LogisticRegression.hpp"

double LogisticRegression::train(std::vector<std::pair<double, int>> D) {
    size_t n = D.size();

    for (int k=0; k<epochs; k++) {
        std::pair<double, double> res;

        hpx::run_as_hpx_thread([&]{
            res = hpx::transform_reduce(
                hpx::execution::par,
                D.begin(),
                D.end(),
                std::make_pair(0.0, 0.0),
                [&](auto a, auto b) {
                    a.first = a.first + b.first;
                    a.second = a.second + b.second;
                    return a;
                },
                [&](auto a) {
                    double diff = (f(a.first) - a.second);
                    return std::make_pair(diff * a.first, diff);
                }
            );
        });

        double dj_dw = res.first, dj_db = res.second;

        dj_dw /= n;
        dj_db /= n;

        W -= alpha * dj_dw;
        B -= alpha * dj_db;
    }

    double train_accuracy;

    hpx::run_as_hpx_thread([&]{
        train_accuracy = hpx::transform_reduce(
            D.begin(),
            D.end(),
            0.0,
            std::plus<int>(),
            [&](auto a) {
                return (predict(a.first) == a.second) ? 1 : 0;
            }
        );
    });

    train_accuracy /= n;

    return train_accuracy;
}
