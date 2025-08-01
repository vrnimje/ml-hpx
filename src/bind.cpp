#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h> // For std::vector conversion
#include <nanobind/stl/pair.h>   // For std::pair conversion

#include <hpx/hpx_start.hpp>

#include "LinearRegression.hpp"
#include "LogisticRegression.hpp"

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
        // Constructor binding
        .def(nb::init<int, double, unsigned int>(),
            "Initializes the LogisticRegression model.",
            nb::arg("num_epochs") = 5000,
            nb::arg("learning_rate") = 1e-3,
            nb::arg("seed") = 0) // default seed argument

        // Method bindings
        .def("f", &LinearRegression::f,
            "Calculates Y = WX + B for a single X.")

        // Overloaded predict method bindings
        .def("predict", nb::overload_cast<double>(&LinearRegression::predict, nb::const_),
            "Predicts Y for a single X value.")
        .def("predict", nb::overload_cast<std::vector<double>>(&LinearRegression::predict, nb::const_),
            "Predicts Y for a vector of X values.")

        .def("fit", nb::overload_cast<std::vector<std::pair<double, double>>>(&LinearRegression::fit),
            "Trains the model using the provided data D (vector of (input, target) pairs).")

        .def("fit", nb::overload_cast<std::vector<double>, std::vector<double>>(&LinearRegression::fit),
            "Trains the model using the provided input X and target Y.");

    nb::class_<LogisticRegression>(m, "LogisticRegression")
        // Constructor binding
        .def(nb::init<int, double, unsigned int>(),
            "Initializes the LinearRegression model.",
            nb::arg("num_epochs") = 5000,
            nb::arg("learning_rate") = 1e-3,
            nb::arg("seed") = 0) // default seed argument

        // Method bindings
        .def("f", &LogisticRegression::f,
            "Calculates Y = 1 / (1 + exp(-Wx - B)) for a single X.")

        // Overloaded predict method bindings
        .def("predict", nb::overload_cast<double>(&LogisticRegression::predict, nb::const_),
            "Predicts Y for a single X value.")
        .def("predict", nb::overload_cast<std::vector<double>>(&LogisticRegression::predict, nb::const_),
            "Predicts Y for a vector of X values.")

        .def("fit", nb::overload_cast<std::vector<std::pair<double, int>>>(&LogisticRegression::fit),
            "Trains the model using the provided data D (vector of (input, target) pairs).")

        .def("fit", nb::overload_cast<std::vector<double>, std::vector<int>>(&LogisticRegression::fit),
            "Trains the model using the provided input X and target Y.");
}
