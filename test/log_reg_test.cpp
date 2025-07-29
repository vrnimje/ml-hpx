#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <chrono>
#include "LogisticRegression.hpp"

int main(int argc, char* argv[]) {
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

    std::vector<std::pair<double, double>> D;
    std::string line;

    std::getline(in, line); // Skip column names
    int l = 0;

    while (std::getline(in, line)) {
        std::stringstream L(line);
        std::string data;
        std::pair<double, double> p;

        while(getline(L, data, ',')) {
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

    size_t n = D.size();
    size_t n_train = 0.7 * n;
    size_t n_test = (n - n_train);

    std::cout << n_train << ", " << n_test << std::endl;

    auto split_iter = D.begin();
    std::advance(split_iter, n_train);

    std::vector<std::pair<double, int>> D_train(D.begin(), split_iter);
    std::vector<std::pair<double, int>> D_test(split_iter, D.end());

    LogisticRegression logistic_regressor = LogisticRegression(4000, 0.001);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    double train_acc = logistic_regressor.train(D_train);

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    std::cout << "Train accuracy: " << train_acc << std::endl;
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::duration<long double>>(end - start).count() << " ms" << std::endl;

    int correct = 0;

    for (int i=0; i<n_test; i++) {
        int p = logistic_regressor.predict(D_test[i].first);

        if (p == D_test[i].second) correct++;
    }

    std::cout << "Test Accuracy:" << ((double)correct / n_test) << std::endl;

    return 0;
}
