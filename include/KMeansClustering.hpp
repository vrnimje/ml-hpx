#include <cmath>
#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/parallel/algorithms/transform.hpp>
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>

#include <vector>
#include <unordered_map>
#include <random>

class KMeansClustering {
private:
    std::vector<std::pair<double, double>> centroids;
    std::unordered_map<int, std::vector<std::pair<double, double>>> clusters;
    std::vector<std::pair<double, double>> D;
    int k, max_iter;
    double tol;

    std::mt19937 gen;
    std::uniform_real_distribution<double> distribution;

public:
    KMeansClustering(int k = 8, int max_iter = 300, unsigned int seed = 0) : k(k), max_iter(max_iter), gen(seed), distribution(0.0, 1.0) {
        for (size_t i = 0; i < k; ++i) {
            centroids.push_back({distribution(gen), distribution(gen)});
        }
        tol = 0.01;
    }

    int predict(const std::pair<double, double>& point) const {
        int closest_cluster_idx = 0;
        double min_dist_sq = std::numeric_limits<double>::max();

        for (size_t i = 0; i < k; ++i) {
            // Using squared distance avoids expensive sqrt and yields the same result for comparison
            double dist_sq = std::pow(point.first - centroids[i].first, 2) +
                             std::pow(point.second - centroids[i].second, 2);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                closest_cluster_idx = i;
            }
        }
        return closest_cluster_idx;
    }

    double fit(std::vector<std::pair<double, double>> D);
};
