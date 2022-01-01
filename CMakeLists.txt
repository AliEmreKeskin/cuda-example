cmake_minimum_required(VERSION 3.18)
project(cuda-example VERSION 0.1.0 LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME})

find_package(OpenCV)
find_package(Boost REQUIRED COMPONENTS filesystem)


target_sources(${PROJECT_NAME} PRIVATE main.cpp CudaAlgorithm.cu Algorithm.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} Boost::filesystem)