#include "LogisticRegression.hpp"

double LogisticRegression::f(const std::vector<double>& X, bool is_hpx) {
    double dot_product;
    if (is_hpx) {
        dot_product = hpx::transform_reduce(
            hpx::execution::par,
            W.begin(), W.end(), X.begin(), 0.0
        );
    }
    else {
        dot_product = hpx::transform_reduce(
            W.begin(), W.end(), X.begin(), 0.0
        );
    }

    return std::pow(1 + std::exp(-1 * (dot_product + B)), -1);
}

int LogisticRegression::predict(const std::vector<double>& X, bool is_hpx = false) {
    double Y = this->f(X, is_hpx);
    return (Y > 0.5) ? 1 : 0;
}

std::vector<int> LogisticRegression::predict(const std::vector<std::vector<double>>& X) {
    std::vector<int> Y_pred(X.size());
    hpx::run_as_hpx_thread([&] {
        hpx::transform(hpx::execution::par,
            X.begin(), X.end(), Y_pred.begin(),
            [this](const auto& x) { return this->predict(x, true); }
        );
    });

    return Y_pred;
}

double LogisticRegression::fit(std::vector<std::vector<double>> X, std::vector<int> Y) {
    size_t n_samples = X.size();

    if (!is_init) {
        init_weights(X[0].size());
    }

    auto zip_begin = hpx::util::zip_iterator(X.begin(), Y.begin());
    auto zip_end = hpx::util::zip_iterator(X.end(), Y.end());

    for (int k = 0; k < epochs; ++k) {
        // The initial value for reduction
        gradient_pair initial_grads(std::vector<double>(num_features, 0.0), 0.0);

        // This transform_reduce calculates the sum of gradients across all samples
        gradient_pair total_grads;

        hpx::run_as_hpx_thread([&] {
            total_grads = hpx::transform_reduce(
                hpx::execution::par,
                zip_begin, zip_end,
                initial_grads,

                // 2. Reduction operation: sums up the gradients from each sample
                [](const gradient_pair& a, const gradient_pair& b) {
                    std::vector<double> w_grad(a.first.size());
                    hpx::transform(a.first.begin(), a.first.end(), b.first.begin(), w_grad.begin(), std::plus<double>());
                    return std::make_pair(w_grad, a.second + b.second);
                },

                // 1. Transform operation: calculates gradient for a single sample (x_i, y_i)
                [this](const auto& zipped_val) {
                    const auto& x_i = hpx::get<0>(zipped_val);
                    double y_i = hpx::get<1>(zipped_val);
                    double diff = predict(x_i) - y_i;

                    std::vector<double> dj_dw_i(num_features);
                    // dj_dw for sample i is diff * x_i
                    hpx::transform(x_i.begin(), x_i.end(), dj_dw_i.begin(),
                        [diff](double feature_val) { return diff * feature_val; });

                    double dj_db_i = diff;
                    return std::make_pair(dj_dw_i, dj_db_i);
                }
            );

            // Average the gradients
            std::vector<double> dj_dw = total_grads.first;
            double dj_db = total_grads.second;

            hpx::for_each(hpx::execution::par, dj_dw.begin(), dj_dw.end(), [n_samples](double& grad){
                grad /= n_samples;
            });
            dj_db /= n_samples;

            // Update weights and bias
            hpx::transform(hpx::execution::par, W.begin(), W.end(), dj_dw.begin(), W.begin(),
                [this](double w, double grad) { return w - alpha * grad; });

            B -= alpha * dj_db;
        });
    }

    double train_accuracy;

    hpx::run_as_hpx_thread([&] {
        train_accuracy = hpx::transform_reduce(
            hpx::execution::par,
            zip_begin, zip_end,
            0.0,
            std::plus<int>(),
            [&](const auto& zipped_val) {
                return (predict(hpx::get<0>(zipped_val)) == hpx::get<1>(zipped_val)) ? 1 : 0;
            }
        );
    });

    train_accuracy /= n_samples;

    return train_accuracy;
}
