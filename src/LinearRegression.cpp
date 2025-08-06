#include "LinearRegression.hpp"

double LinearRegression::fit(std::vector<std::pair<double, double>> D) {
    size_t n = D.size();

    for (int k=0; k<epochs; k++) {
        std::pair<double, double> res = hpx::transform_reduce(
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

        double dj_dw = res.first, dj_db = res.second;

        dj_dw /= n;
        dj_db /= n;

        W -= alpha * dj_dw;
        B -= alpha * dj_db;
    }

    double loss = hpx::transform_reduce(
        D.begin(),
        D.end(),
        0.0,
        std::plus<double>(),
        [&](auto a) {
            return std::pow(f(a.first) - a.second, 2);
        }
    );

    loss /= (2 * n);

    return loss;
}

double LinearRegression::fit(std::vector<double> X, std::vector<double> Y) {
    std::vector<std::pair<double, double>> D;

    for (size_t i = 0; i < X.size(); i++) {
        D.push_back(std::make_pair(X[i], Y[i]));
    }

    return LinearRegression::fit(D);
}
