cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)

project(ml-hpx CXX)
find_package(HPX REQUIRED)

# Linear Regression
add_library(linear_regression src/LinearRegression.cpp)

target_include_directories(linear_regression
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(linear_regression
    PUBLIC
        HPX::hpx
)

# Logistic Regression
add_library(logistic_regression src/LogisticRegression.cpp)

target_include_directories(logistic_regression
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(logistic_regression
    PUBLIC
        HPX::hpx
)

add_executable(linear_reg_test test/linear_reg_test.cpp)
target_link_libraries(linear_reg_test
    linear_regression
)

add_executable(log_reg_test test/log_reg_test.cpp)
target_link_libraries(log_reg_test
    logistic_regression
)

add_executable(linear_reg_benchmark bench/linear_reg.cpp)
target_link_libraries(linear_reg_benchmark
    HPX::hpx
    HPX::wrap_main
    HPX::iostreams_component
)

add_executable(log_reg_benchmark bench/log_reg.cpp)
target_link_libraries(log_reg_benchmark
    HPX::hpx
    HPX::wrap_main
    HPX::iostreams_component
)
