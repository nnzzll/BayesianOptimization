cmake_minimum_required(VERSION 3.16.0)
project(BayesOpt VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_library(BayesOpt INTERFACE)
target_link_libraries(BayesOpt INTERFACE Eigen3::Eigen)
target_include_directories(BayesOpt INTERFACE ${CMAKE_SOURCE_DIR}/BayesOpt)

add_executable(testBayesOpt main.cpp)
target_link_libraries(testBayesOpt PRIVATE BayesOpt)