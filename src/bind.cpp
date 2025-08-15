#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h> // For std::vector conversion
#include <nanobind/stl/pair.h>   // For std::pair conversion

#include <hpx/hpx_start.hpp>

#include "LinearRegression.hpp"
#include "LogisticRegression.hpp"
#include "KNearestNeighbours.hpp"
#include "KMeansClustering.hpp"
#include "Perceptron.hpp"
#include "SVC.hpp"

namespace nb = nanobind;

NB_MODULE(_ml_hpx_impl, m) {
    m.def("initialize", []() {
        // Check if HPX is already initialized before starting
        if (!hpx::is_running()) {
            hpx::start(nullptr, 0, nullptr);
        }
    }, "Initializes the HPX runtime.");

    m.def("finalize", []() {
        if (hpx::is_running()) {
            hpx::post([]() { hpx::finalize(); });
        }
    }, "Finalizes the HPX runtime.");

    nb::class_<LinearRegression>(m, "LinearRegression")
        .def(nb::init<int, double, unsigned int>(),
            "Initializes the LogisticRegression model.",
            nb::arg("num_epochs") = 5000,
            nb::arg("learning_rate") = 1e-3,
            nb::arg("seed") = 0
        )

        .def("fit", &LinearRegression::fit,
            "Trains the model using input features X and target Y."
        )

        // Bind the vectorized 'predict' methods
        .def("predict", nb::overload_cast<const std::vector<double>&, bool>(&LinearRegression::predict),
            "Predicts Y for a single feature vector X."
        )

        .def("predict", nb::overload_cast<std::vector<std::vector<double>>&>(&LinearRegression::predict),
            "Predicts Y for a batch of feature vectors X."
        );

    nb::class_<LogisticRegression>(m, "LogisticRegression")
        // Constructor binding
        .def(nb::init<int, double, unsigned int>(),
            "Initializes the LogisitcRegression classifier model.",
            nb::arg("num_epochs") = 5000,
            nb::arg("learning_rate") = 1e-3,
            nb::arg("seed") = 0
        )

        .def("fit", &LogisticRegression::fit,
            "Trains the model using input features X and target Y."
        )

        // Overloaded predict method bindings
        .def("predict", nb::overload_cast<const std::vector<double>&, bool>(&LogisticRegression::predict),
            "Predicts Y for a single feature vector X.")

        .def("predict", nb::overload_cast<const std::vector<std::vector<double>>&>(&LogisticRegression::predict),
            "Predicts Y for a batch of feature vectors X.");

    nb::class_<KNearestNeighbours>(m, "KNearestNeighbours")
        // Constructor binding
        .def(nb::init<int>(),
            "Initializes the KNearestNeighbours model.",
            nb::arg("k") = 5
        )

        // Method bindings
        .def("fit", nb::overload_cast<std::vector<std::tuple<double, double, int>>>(&KNearestNeighbours::fit),
            "Fits the model using the provided data D (vector of (input, target) pairs).")
        .def("fit", nb::overload_cast<std::vector<std::pair<double, double>>, std::vector<int>>(&KNearestNeighbours::fit),
            "Fits the model using the X (input) and Y (pairs).")

        .def("predict", nb::overload_cast<std::pair<double, double>>(&KNearestNeighbours::predict),
            "Predicts class Y for a X value.")
        .def("predict", nb::overload_cast<std::vector<std::pair<double, double>>>(&KNearestNeighbours::predict),
            "Predicts classes Y for a vector of X values.");

    nb::class_<KMeansClustering>(m, "KMeansClustering")
        // Constructor binding
        .def(nb::init<int, int, unsigned int>(),
            "Initializes the KMeansClustering model.",
            nb::arg("k") = 8,
            nb::arg("max_iter") = 300,
            nb::arg("seed") = 0
        )

        // Method binding
        .def("fit", nb::overload_cast<std::vector<std::pair<double, double>>>(&KMeansClustering::fit),
            "Fits the model using the provided data D (vector of double pairs).");

    nb::class_<Perceptron>(m, "Perceptron")
        // Constructor binding
        .def(nb::init<double, unsigned int>(),
            "Initializes the Perceptron model.",
            nb::arg("learning_rate") = 1e-3,
            nb::arg("seed") = 0
        )

        // Overloaded predict method bindings
        .def("predict", nb::overload_cast<std::pair<double, double>>(&Perceptron::predict, nb::const_),
            "Predicts Y for a point X.")
        .def("predict", nb::overload_cast<std::vector<std::pair<double, double>>>(&Perceptron::predict, nb::const_),
            "Predicts Y for a vector of X points.")

        // Overloaded fit method bindings
        .def("fit", nb::overload_cast<std::vector<std::tuple<double, double, int>>>(&Perceptron::fit),
            "Trains the model using the provided data D (vector of (input, target) pairs).")
        .def("fit", nb::overload_cast<std::vector<std::pair<double, double>>, std::vector<int>>(&Perceptron::fit),
            "Trains the model using the provided input X and target Y.");

    nb::class_<SVC>(m, "SVC")
        // Constructor binding
        .def(nb::init<int, double, unsigned int>(),
            "Initializes the SVC model.",
            nb::arg("max_iter") = 1000,
            nb::arg("learning_rate") = 1e-3,
            nb::arg("seed") = 0
        )

        // Overloaded predict method bindings
        .def("predict", nb::overload_cast<std::pair<double, double>>(&SVC::predict, nb::const_),
            "Predicts Y for a point X.")
        .def("predict", nb::overload_cast<std::vector<std::pair<double, double>>>(&SVC::predict, nb::const_),
            "Predicts Y for a vector of X points.")

        // Overloaded fit method bindings
        .def("fit", nb::overload_cast<std::vector<std::tuple<double, double, int>>>(&SVC::fit),
            "Trains the model using the provided data D (vector of (input, target) pairs).")
        .def("fit", nb::overload_cast<std::vector<std::pair<double, double>>, std::vector<int>>(&SVC::fit),
            "Trains the model using the provided input X and target Y.");
}
