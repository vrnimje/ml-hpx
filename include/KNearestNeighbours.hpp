#include <cmath>
#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/parallel/algorithms/transform.hpp>
#include <hpx/init.hpp>
#include <hpx/hpx_start.hpp>

#include <tuple>
#include <unordered_map>
#include <algorithm>

class KNearestNeighbours {
private:
    std::vector<std::tuple<double, double, int>> D; // Seen data points
    int k;

    inline double euclidean_dist_sq(const std::pair<double, double>& a, const std::pair<double, double>& b) {
        return std::pow(a.first - b.first, 2) + std::pow(a.second - b.second, 2);
    }

public:
    KNearestNeighbours(int k = 5) : k(k) {}

    void fit(std::vector<std::tuple<double, double, int>> D) {
        this->D = D;
    }

    void fit(std::vector<std::pair<double, double>> X, std::vector<int> Y) {
        this->D = std::vector<std::tuple<double, double, int>>(X.size());
        std::transform(X.begin(), X.end(), Y.begin(), D.begin(),
            [](const std::pair<double, double>& x, int y) {
                return std::make_tuple(x.first, x.second, y);
            });
    }

    int predict(std::pair<double, double> X);

    std::vector<int> predict(std::vector<std::pair<double, double>> X);
};
