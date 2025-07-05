cmake_minimum_required(VERSION 3.20)
set (CMAKE_CXX_STANDARD 20)

project(ml-hpx CXX)
find_package(HPX REQUIRED)

add_executable(linear_reg linear_reg.cpp)
target_link_libraries(linear_reg HPX::hpx HPX::wrap_main HPX::iostreams_component)

add_executable(log_reg log_reg.cpp)
target_link_libraries(log_reg HPX::hpx HPX::wrap_main HPX::iostreams_component)
