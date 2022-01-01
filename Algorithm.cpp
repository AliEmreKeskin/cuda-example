/**
 * @file Algorithm.cpp
 * @author Ali Emre Keskin (aliemrekskn@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-12-31
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include "Algorithm.h"

namespace aek
{
    Algorithm::Algorithm() : windowSize_(5, 5)
    {
    }

    Algorithm::~Algorithm()
    {
    }

    void Algorithm::Apply(cv::InputArray src, cv::OutputArray dst)
    {
        dst.createSameSize(src, CV_16UC1);
        cv::Mat output = dst.getMatRef();
        output = 0;
        cv::Mat input = src.getMat();

        cv::Mat window = cv::Mat(windowSize_, src.type());
        cv::Mat transpose;
        cv::Mat product;

        cv::Point tl;
        for (tl.y = 0; tl.y < src.size().height - windowSize_.height; tl.y++)
        {
            for (tl.x = 0; tl.x < src.size().width - windowSize_.width; tl.x++)
            {
                input(cv::Rect(tl, windowSize_)).copyTo(window);
                Transpose(window, transpose);
                Matmul(window, transpose, product);
                output(cv::Rect(tl, windowSize_)) += product;
            }
        }
    }

    void Algorithm::Transpose(cv::InputArray src, cv::OutputArray dst)
    {
        dst.create(cv::Size(src.size().height, src.size().width), src.type());
        cv::Mat transpose = dst.getMatRef();
        cv::Mat original = src.getMat();
        for (size_t y = 0; y < src.size().height; y++)
        {
            for (size_t x = 0; x < src.size().width; x++)
            {
                transpose.at<uint16_t>(cv::Point(y, x)) = original.at<uint16_t>(cv::Point(x, y));
            }
        }
    }

    void Algorithm::Matmul(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst)
    {
        dst.create(cv::Size(src2.size().width, src1.size().height), src1.type());
        cv::Mat output = dst.getMatRef();
        output = 0;
        cv::Mat input1 = src1.getMat();
        cv::Mat input2 = src2.getMat();

        for (size_t i = 0; i < src2.size().width; i++)
        {
            for (size_t j = 0; j < src1.size().height; j++)
            {
                for (size_t k = 0; k < src1.size().width; k++)
                {
                    uint16_t a, b, c;
                    a = input1.at<uint16_t>(cv::Point(k, j));
                    b = input2.at<uint16_t>(cv::Point(i, k));
                    c = a * b;
                    output.at<uint16_t>(cv::Point(i, j)) += c;
                }
            }
        }
    }
}
