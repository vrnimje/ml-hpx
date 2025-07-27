#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h> // For std::vector conversion
#include <nanobind/stl/pair.h>   // For std::pair conversion

#include "LinearRegression.hpp"
#include "LogisticRegression.hpp"

namespace nb = nanobind;

NB_MODULE(_ml_hpx_impl, m) {
    nb::class_<LinearRegression>(m, "LinearRegression")
        // Constructor binding
        .def(nb::init<int, float, unsigned int>(), // Update signature
            "Initializes the LinearRegression model.",
            nb::arg("num_epochs") = 5000,
            nb::arg("learning_rate") = 1e-3,
            nb::arg("seed") = 0) // Add default seed argument

        // Method bindings
        .def("f", &LinearRegression::f,
            "Calculates Y = WX + B for a single X.")

        // Overloaded predict method bindings
        .def("predict", nb::overload_cast<float>(&LinearRegression::predict, nb::const_),
            "Predicts Y for a single X value.")
        .def("predict", nb::overload_cast<std::vector<float>>(&LinearRegression::predict, nb::const_),
            "Predicts Y for a vector of X values.")

        .def("train", &LinearRegression::train,
            "Trains the model using the provided data D (vector of (input, target) pairs).");

    nb::class_<LogisticRegression>(m, "LogisticRegression")
        // Constructor binding
        .def(nb::init<int, float>(),
            "Initializes the LogisticRegression model.",
            nb::arg("num_epochs") = 5000,
            nb::arg("learning_rate") = 1e-3)

        // Method bindings
        .def("f", &LogisticRegression::f,
            "Calculates Y = 1 / (1 + exp(-Wx - B)) for a single X.")

        // Overloaded predict method bindings
        .def("predict", nb::overload_cast<float>(&LogisticRegression::predict, nb::const_),
            "Predicts Y for a single X value.")
        .def("predict", nb::overload_cast<std::vector<float>>(&LogisticRegression::predict, nb::const_),
            "Predicts Y for a vector of X values.")

        .def("train", &LogisticRegression::train,
            "Trains the model using the provided data D (vector of (input, target) pairs).");
}
