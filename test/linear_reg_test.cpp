#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <chrono>
#include "LinearRegression.hpp"

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

    LinearRegression logistic_regressor = LinearRegression(5000, 1e-5);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    double loss = logistic_regressor.train(D);

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::duration<long double>>(end - start).count() << " ms" << std::endl;

    return 0;
}
