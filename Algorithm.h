#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

/**
 * @file Algorithm.h
 * @author Ali Emre Keskin (aliemrekskn@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-12-31
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <opencv2/opencv.hpp>

namespace aek
{
    class Algorithm
    {
    public:
        Algorithm();
        ~Algorithm();
        virtual void Apply(cv::InputArray src, cv::OutputArray dst);

    protected:
        virtual void Transpose(cv::InputArray src, cv::OutputArray dst);
        virtual void Matmul(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst);
        cv::Size windowSize_;
    };

} // namespace aek

#endif // __ALGORITHM_H__