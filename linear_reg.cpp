#include <hpx/algorithm.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/iostream.hpp>

#include <iostream>
#include <random>
#include <vector>
#include <fstream>

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

    std::vector<float> X, Y;
    std::string line;

    std::getline(in, line); // Skip column names
    int l = 0;

    while (std::getline(in, line)) {
        std::stringstream L(line); 
        std::string data;
        while(getline(L, data, ',')) {
            // std::cout << data << "\n";
            l++;
            if (l%2 != 0) {
                X.push_back(std::stof(data));
            }
            else 
                Y.push_back(std::stof(data));
        }
    }

    // No. of data points
    int n = X.size();

    // for (int i=0; i<n; i++) {
    //     std::cout << Y[i] << ", "; 
    // }

    // Linear equation: Y = WX + B

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    float W = distribution(gen), B = distribution(gen);
    float prev_W, prev_B;

    int N = 1500;
    float alpha = 0.00077;

    // Gradient Descent

    for (int k=0; k<N; k++) {
        
        float dj_dw = 0, dj_db = 0;
        float J = 0;
        for (int i=0; i<n; i++) {
            J += (f(X[i], W, B) - Y[i]) * (f(X[i], W, B) - Y[i]) ;
            dj_dw += (f(X[i], W, B) - Y[i]) * X[i];
            dj_db += (f(X[i], W, B) - Y[i]);
        }
        J /= (n * 2);
        dj_dw /= n;
        dj_db /= n;

        std::cout << J << ", ";

        //std::cout << "Derivatives: " << dj_dw << "," << dj_db << "\n";

        prev_W = W;
        prev_B = B;

        W -= alpha * dj_dw;
        B -= alpha * dj_dw;

        // if (prev_W == W && prev_B == B) break;
    }

    std::cout << "\n";

    std::cout << "Final parameters: " << W << "," << B << "\n";

    return hpx::local::finalize();
}

int main(int argc, char* argv[]) {
    return hpx::local::init(hpx_main, argc, argv);
}