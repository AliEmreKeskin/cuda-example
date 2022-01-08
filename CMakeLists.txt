cmake_minimum_required(VERSION 3.18)
project(cuda-example VERSION 0.1.0 LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME})

find_package(OpenCV)
find_package(Boost REQUIRED COMPONENTS filesystem)

target_sources(${PROJECT_NAME} PRIVATE main.cpp CudaAlgorithm.cu Algorithm.cpp Timer.cpp CudaAlgorithm2.cu)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} Boost::filesystem)

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")