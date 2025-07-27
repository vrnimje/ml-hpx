cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)

project(ml-hpx CXX)
find_package(HPX REQUIRED)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
    message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

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

nanobind_add_module(_ml_hpx_impl STABLE_ABI src/bind.cpp src/LinearRegression.cpp src/LogisticRegression.cpp)

target_include_directories(_ml_hpx_impl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(_ml_hpx_impl
    PUBLIC
        HPX::hpx
)

install(TARGETS _ml_hpx_impl LIBRARY DESTINATION ml_hpx)

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
