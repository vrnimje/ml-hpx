#include "KNearestNeighbours.hpp"

int KNearestNeighbours::predict(std::pair<double, double> X) {
    std::vector<std::pair<double, int>> dist_labels(D.size());

    hpx::run_as_hpx_thread([&]{
        // Compute all distances in parallel
        hpx::transform(hpx::execution::par, D.begin(), D.end(), dist_labels.begin(),
            [&](const std::tuple<double, double, int>& d) {
                double dist_sq = euclidean_dist_sq(X, std::make_pair(std::get<0>(d), std::get<1>(d)));
                return std::make_pair(dist_sq, std::get<2>(d));
            }
        );

        // Get k-nearest points
        hpx::nth_element(dist_labels.begin(), dist_labels.begin() + k, dist_labels.end());
    });

    // Count labels among k nearest neighbors
    std::unordered_map<int, int> label_counts;
    label_counts.reserve(k); // Reserve space for efficiency

    for (int i = 0; i < k; ++i) {
        ++label_counts[dist_labels[i].second];
    }

    // Find majority label
    return std::max_element(label_counts.begin(), label_counts.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; })->first;
}

std::vector<int> KNearestNeighbours::predict(std::vector<std::pair<double, double>> X) {
    std::vector<int> predictions(X.size());

    std::transform(X.begin(), X.end(), predictions.begin(),
        [this](const std::pair<double, double>& point) {
            return this->predict(point);
        }
    );

    return predictions;
}
