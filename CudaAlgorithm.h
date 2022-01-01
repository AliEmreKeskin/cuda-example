#ifndef __CUDAALGORITHM_H__
#define __CUDAALGORITHM_H__

/**
 * @file CudaAlgorithm.h
 * @author Ali Emre Keskin (aliemrekskn@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-12-28
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <opencv2/opencv.hpp>

namespace aek
{
    class CudaAlgorithm
    {
    public:
        CudaAlgorithm();
        ~CudaAlgorithm();
        void Apply(cv::InputArray input, cv::OutputArray output);

    private:
    };

} // namespace aek
#endif // __CUDAALGORITHM_H__