#include "KNearestNeighbours.hpp"

int KNearestNeighbours::predict(std::pair<double, double> X) {
    std::vector<int> neigh_labels(k);

    hpx::run_as_hpx_thread([&]{
        hpx::sort(hpx::execution::par, D.begin(), D.end(),
            [&](const std::tuple<double, double, int>& a, const std::tuple<double, double, int>& b) {
                return euclidean_dist(X, std::make_pair(std::get<0>(a), std::get<1>(a))) <
                       euclidean_dist(X, std::make_pair(std::get<0>(b), std::get<1>(b)));
            }
        );

        hpx::transform(hpx::execution::par, D.begin(), D.begin() + k, neigh_labels.begin(),
            [&](const std::tuple<double, double, int>& d) {
                return std::get<2>(d);
            }
        );
    });

    std::unordered_map<int, int> label_counts;
    for (int label : neigh_labels) {
        label_counts[label]++;
    }

    int max_count = 0;
    int maj_label = 0;
    for (const auto& pair : label_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            maj_label = pair.first;
        }
    }

    return maj_label;
}

std::vector<int> KNearestNeighbours::predict(std::vector<std::pair<double, double>> X) {
    std::vector<int> pred(X.size());
    for (int i=0; i<X.size(); i++) {
        pred[i] = KNearestNeighbours::predict(X[i]);
    }
    return pred;
}
