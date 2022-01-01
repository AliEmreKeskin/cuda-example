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

int main(int argc, char const *argv[])
{
    boost::filesystem::path input(argv[1]);
    boost::filesystem::path output = input.parent_path() / "output.tiff";

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

    aek::Algorithm algorithm;
    cv::Mat result;
    algorithm.Apply(img16u, result);

    return 0;
}
