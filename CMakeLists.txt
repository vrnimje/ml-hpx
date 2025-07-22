cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)

project(ml-hpx CXX)
find_package(HPX REQUIRED)

# Nanobind setup
if (CMAKE_VERSION VERSION_LESS 3.29)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# As nanobind is used as a git submodule
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

# Linear Regression
add_library(linear_regression src/LinearRegression.cpp)

target_include_directories(linear_regression
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(linear_regression
    PUBLIC HPX::hpx
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

nanobind_add_module(ml_hpx src/bind.cpp src/LinearRegression.cpp src/LogisticRegression.cpp)

target_include_directories(ml_hpx
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ml_hpx
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
