/**
 * @file main.cpp
 * @author Ali Emre Keskin (aliemrekskn@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-12-28
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include "Algorithm.h"
#include "CudaAlgorithm.h"
#include "Timer.h"
#include "CudaAlgorithm2.h"

int main(int argc, char const *argv[])
{
    if (argc < 2)
    {
        std::runtime_error("Missing arguments.\nUsage: cuda-example <input-image-path> <OPTIONAL-output-image-path>");
    }

    boost::filesystem::path input(argv[1]);
    boost::filesystem::path output = input.parent_path() / "output.png";

    if (argc == 3)
    {
        output = boost::filesystem::path(argv[2]);
    }

    if (!boost::filesystem::exists(input))
    {
        std::cerr << "File not found: " << input << std::endl;
    }

    cv::Mat img = cv::imread(input.string(), cv::ImreadModes::IMREAD_GRAYSCALE);
    cv::Mat img16u;
    img.convertTo(img16u, CV_16UC1);
    cv::Mat result;
    cv::Mat eightBitOutputImage;

    aek::Algorithm cpuAlgorithm;
    aek::CudaAlgorithm cudaAlgorithm;
    aek::CudaAlgorithm2 cudaAlgorithm2;

    aek::Timer timer("main");

    timer.Tic("cpu");
    cpuAlgorithm.Apply(img16u, result);
    timer.Toc();
    result.convertTo(eightBitOutputImage, CV_8UC1);
    cv::imwrite("cpu.png", eightBitOutputImage);

    timer.Tic("cuda");
    cudaAlgorithm.Apply(img16u, result);
    timer.Toc();
    result.convertTo(eightBitOutputImage, CV_8UC1);
    cv::imwrite("cuda.png", eightBitOutputImage);

    timer.Tic("cuda2");
    cudaAlgorithm2.Apply(img16u, result);
    timer.Toc();
    result.convertTo(eightBitOutputImage, CV_8UC1);
    cv::imwrite("cuda2.png", eightBitOutputImage);

    return 0;
}
