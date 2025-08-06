#include "KMeansClustering.hpp"

double KMeansClustering::fit(std::vector<std::pair<double, double>> D) {
    this->D = D;

    int current_iter = 0;
    while (current_iter < max_iter) {
        std::vector<int> assignments(D.size());
        hpx::transform(D.cbegin(), D.cend(), assignments.begin(),
            [this](const std::pair<double, double>& point) {
                return this->predict(point);
            }
        );

        auto old_centroids = centroids;
        std::vector<std::pair<double, double>> new_sums(k, {0.0, 0.0});
        std::vector<size_t> new_counts(k, 0);

        for (size_t i = 0; i < D.size(); ++i) {
            int cluster_idx = assignments[i];
            new_sums[cluster_idx].first += D[i].first;
            new_sums[cluster_idx].second += D[i].second;
            new_counts[cluster_idx]++;
        }

        for (size_t i = 0; i < k; ++i) {
            if (new_counts[i] > 0) {
                centroids[i] = {
                    new_sums[i].first / new_counts[i],
                    new_sums[i].second / new_counts[i]
                };
            }
        }

        // Check for convergence
        double total_movement = 0.0;
        for (size_t i = 0; i < k; ++i) {
            total_movement += std::hypot(centroids[i].first - old_centroids[i].first,
                                         centroids[i].second - old_centroids[i].second);
        }

        if (total_movement < tol) {
            break; // Exit the loop if converged
        }

        current_iter++;
    }

    double sse = 0.0;
    for (size_t i = 0; i < D.size(); ++i) {
        int cluster_idx = predict(D[i]);
        sse += std::pow(D[i].first - centroids[cluster_idx].first, 2) +
               std::pow(D[i].second - centroids[cluster_idx].second, 2);
    }

    return sse;
}
