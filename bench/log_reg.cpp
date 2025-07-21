#include <cmath>
#include <cstddef>
#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/testing/performance.hpp>

#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <fstream>
#include <algorithm>
#include <ranges>
#include <execution>

// Linear equation: Y = 1 / (1 + exp(-Wx - B))
float f(float X, float W, float B) {
    return 1 / (1 + std::exp(-1 * (W*X + B)));
}

int pred(float res) {
    return (res > 0.5) ? 1 : 0;
}

float loss_fn(float sigmoid, int y) {
    return -1 * ((y*std::log(sigmoid)) + ((1 - y) * (std::log(1 - sigmoid))));
}

int hpx_main(int argc, char* argv[]) {
    std::string filename;
    if (argc > 1) {
        filename = argv[1];
    } else {
        std::cerr << "Error: No input file provided." << std::endl;
        return 1;
    }
    std::ifstream in(filename, std::ios_base::in);

    if (!in) {
        std::cerr << "Error: Could not open input file: " << filename << std::endl;
        return 1;
    }

    std::vector<std::pair<float, int>> D;
    std::string line;

    std::getline(in, line); // Skip column names
    int l = 0;

    while (std::getline(in, line)) {
        std::stringstream L(line);
        std::string data;
        std::pair<float, float> p;

        while(getline(L, data, ',')) {
            // std::cout << data << "\n";
            l++;
            if (l%2 != 0) {
                // Value
                p.first = std::stof(data);
            }
            else {
                // Class labels
                p.second = std::stoi(data);
            }
        }

        D.push_back(std::move(p));
    }

    // No. of data points
    int n = D.size();

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    float W, B;

    int N = 4000;
    float alpha = 0.001;

    size_t n_train = 0.7 * n;
    size_t n_test = (n - n_train);

    std::cout << n_train << ", " << n_test << std::endl;

    auto split_iter = D.begin();
    std::advance(split_iter, n_train);

    std::vector<std::pair<float, int>> D_train(D.begin(), split_iter);
    std::vector<std::pair<float, int>> D_test(split_iter, D.end());
    int correct;

    // Sequential Gradient Descent - Using for loop
    hpx::util::perftests_report("Sequential GD, for-loop", "seq", 25, [&]{
        correct = 0;
        W = distribution(gen), B = distribution(gen);

        for (int k=0; k<N; k++) {
            float dj_dw = 0, dj_db = 0, loss = 0;

            for (int i=0; i<n_train; i++) {
                dj_dw += (pred(f(D_train[i].first, W, B)) - D_train[i].second) * D_train[i].first;
                dj_db += (pred(f(D_train[i].first, W, B)) - D_train[i].second);
            }

            dj_dw /= n_train;
            dj_db /= n_train;

            W -= alpha * dj_dw;
            B -= alpha * dj_db;
        }

        for (int i=0; i<n_test; i++) {
            int p = pred(f(D_test[i].first, W, B));

            if (p == D_test[i].second) correct++;
        }
    });

    char const* fmt = "Final Parameters: W = {1}, B = {2}\n";
    hpx::util::format_to(std::cout, fmt, W, B);

    char const* acc = "Accuracy: {1}\n";
    hpx::util::format_to(std::cout, acc, ((float)correct / n_test));

    // Gradient Descent - Using STL, seq
    hpx::util::perftests_report("Linear Regression GD, STL, seq", "seq  ", 25, [&]{
        correct = 0;
        W = distribution(gen), B = distribution(gen);
        for (int k=0; k<N; k++) {
            auto res = std::transform_reduce(
                D_train.begin(),
                D_train.end(),
                std::make_pair(0.0f, 0.0f),
                [&](auto a, auto b) {
                    a.first = a.first + b.first;
                    a.second = a.second + b.second;
                    return a;
                },
                [&](auto a) {
                    float diff = (pred(f(a.first, W, B)) - a.second) * a.first;
                    return std::make_pair(diff * a.first, diff);
                }
            );

            float dj_dw = res.first, dj_db = res.second;
            dj_dw /= n_train;
            dj_db /= n_train;

            W -= alpha * dj_dw;
            B -= alpha * dj_db;
        }

        for (int i=0; i<n_test; i++) {
            int p = pred(f(D_test[i].first, W, B));

            if (p == D_test[i].second) correct++;
        }
    });

    hpx::util::format_to(std::cout, fmt, W, B);

    hpx::util::format_to(std::cout, acc, ((float)correct / n_test));

    // Gradient Descent - Using STL, par
    hpx::util::perftests_report("Linear Regression, GD, STL, par", "std::execution::par", 25, [&]{
        correct = 0;
        W = distribution(gen), B = distribution(gen);
        for (int k=0; k<N; k++) {
            auto res = std::transform_reduce(
                std::execution::par,
                D_train.begin(),
                D_train.end(),
                std::make_pair(0.0f, 0.0f),
                [&](auto a, auto b) {
                    a.first = a.first + b.first;
                    a.second = a.second + b.second;
                    return a;
                },
                [&](auto a) {
                    float diff = (pred(f(a.first, W, B)) - a.second) * a.first;
                    return std::make_pair(diff * a.first, diff);
                }
            );

            float dj_dw = res.first, dj_db = res.second;

            dj_dw /= n_train;
            dj_db /= n_train;

            W -= alpha * dj_dw;
            B -= alpha * dj_db;
        }

        for (int i=0; i<n_test; i++) {
            int p = pred(f(D_test[i].first, W, B));

            if (p == D_test[i].second) correct++;
        }
    });

    hpx::util::format_to(std::cout, fmt, W, B);
    hpx::util::format_to(std::cout, acc, ((float)correct / n_test));

    // Gradient Descent, using HPX
    hpx::util::perftests_report("Linear Regression, GD, HPX, par", "hpx::execution::par", 25, [&]{
        correct = 0;
        W = distribution(gen), B = distribution(gen);

        for (int k=0; k<N; k++) {
            auto res = hpx::transform_reduce(
                hpx::execution::par,
                D.begin(),
                D.end(),
                std::make_pair(0.0f, 0.0f),
                [&](auto a, auto b) {
                    a.first = a.first + b.first;
                    a.second = a.second + b.second;
                    return a;
                },
                [&](auto a) {
                    float diff = (pred(f(a.first, W, B)) - a.second) * a.first;
                    return std::make_pair(diff * a.first, diff);
                }
            );

            float dj_dw = res.first, dj_db = res.second;

            dj_dw /= n_train;
            dj_db /= n_train;

            W -= alpha * dj_dw;
            B -= alpha * dj_db;
        }

        for (int i=0; i<n_test; i++) {
            int p = pred(f(D_test[i].first, W, B));

            if (p == D_test[i].second) correct++;
        }
    });

    hpx::util::format_to(std::cout, fmt, W, B);
    hpx::util::format_to(std::cout, acc, ((float)correct / n_test));

    hpx::util::perftests_print_times();

    return hpx::local::finalize();
}

int main(int argc, char* argv[]) {
    return hpx::local::init(hpx_main, argc, argv);
}
