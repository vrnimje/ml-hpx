cmake_minimum_required(VERSION 3.20)

project(ml-hpx CXX)
find_package(HPX REQUIRED)
add_executable(hello_world hello.cpp)
target_link_libraries(hello_world HPX::hpx HPX::wrap_main HPX::iostreams_component)

add_executable(linear_reg linear_reg.cpp)
target_link_libraries(linear_reg HPX::hpx HPX::wrap_main HPX::iostreams_component)