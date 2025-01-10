#include <hpx/algorithm.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/testing/performance.hpp>

#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <fstream>
#include <algorithm>
#include <ranges>
#include <execution>

// Linear equation: Y = WX + B
float f(float X, float W, float B) {
    return (W*X + B);
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

    std::vector<std::pair<float, float>> D;
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
                p.first = std::stof(data);
            }
            else {
                p.second = std::stof(data);
            }
        }

        D.push_back(std::move(p));
    }

    // No. of data points
    int n = D.size();

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    float W = distribution(gen), B = distribution(gen);
    float prev_W, prev_B;

    int N = 5000;
    float alpha = 0.00001;

    // Sequential Gradient Descent - Using for loop
    hpx::util::perftests_report("Sequential GD, for-loop", "seq", 10, [&]{
        for (int k=0; k<N; k++) {
            float dj_dw = 0, dj_db = 0;
            for (int i=0; i<n; i++) {
                dj_dw += (f(D[i].first, W, B) - D[i].second) * D[i].first;
                dj_db += (f(D[i].first, W, B) - D[i].second);
            }

            dj_dw /= n;
            dj_db /= n;

            W -= alpha * dj_dw;
            B -= alpha * dj_db;
        }
    });

    char const* fmt = "Final Parameters: W = {1}, B = {2}\n";
    hpx::util::format_to(std::cout, fmt, W, B);

    W = distribution(gen), B = distribution(gen);

    // Gradient Descent - Using STL, seq
    hpx::util::perftests_report("Linear Regression GD, STL, seq", "seq  ", 10, [&]{
        for (int k=0; k<N; k++) {
            auto res = std::transform_reduce(
                D.begin(),
                D.end(),
                std::make_pair(0.0f, 0.0f),
                [&](auto a, auto b) {
                    a.first = a.first + b.first;
                    a.second = a.second + b.second;
                    return a;
                },
                [&](auto a) {
                    a.first += (f(a.first, W, B) - a.second) * a.first;
                    a.second += (f(a.first, W, B) - a.second);
                    return a;
                }
            );

            float dj_dw = res.first, dj_db = res.second;
            dj_dw /= n;
            dj_db /= n;

            W -= alpha * dj_dw;
            B -= alpha * dj_db;
        }
    });

    hpx::util::format_to(std::cout, fmt, W, B);

    W = distribution(gen), B = distribution(gen);

    // Gradient Descent - Using STL, par
    hpx::util::perftests_report("Linear Regression, GD, STL, par", "std::execution::par", 10, [&]{
        for (int k=0; k<N; k++) {
            auto res = std::transform_reduce(
                std::execution::par,
                D.begin(),
                D.end(),
                std::make_pair(0.0f, 0.0f),
                [&](auto a, auto b) {
                    a.first = a.first + b.first;
                    a.second = a.second + b.second;
                    return a;
                },
                [&](auto a) {
                    a.first += (f(a.first, W, B) - a.second) * a.first;
                    a.second += (f(a.first, W, B) - a.second);
                    return a;
                }
            );

            float dj_dw = res.first, dj_db = res.second;

            dj_dw /= n;
            dj_db /= n;

            W -= alpha * dj_dw;
            B -= alpha * dj_db;
        }
    });

    hpx::util::format_to(std::cout, fmt, W, B);

    W = distribution(gen), B = distribution(gen);

    // Gradient Descent, using HPX
    hpx::util::perftests_report("Linear Regression, GD, HPX, par", "hpx::execution::par", 10, [&]{
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
                    a.first += (f(a.first, W, B) - a.second) * a.first;
                    a.second += (f(a.first, W, B) - a.second);
                    return a;
                }
            );

            float dj_dw = res.first, dj_db = res.second;

            dj_dw /= n;
            dj_db /= n;

            W -= alpha * dj_dw;
            B -= alpha * dj_db;
        }
    });

    hpx::util::format_to(std::cout, fmt, W, B);

    hpx::util::perftests_print_times();

    return hpx::local::finalize();
}

int main(int argc, char* argv[]) {
    return hpx::local::init(hpx_main, argc, argv);
}
