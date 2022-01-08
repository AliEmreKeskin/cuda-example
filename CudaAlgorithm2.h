#ifndef __CUDAALGORITHM2_H__
#define __CUDAALGORITHM2_H__

/**
 * @file CudaAlgorithm2.h
 * @author Ali Emre Keskin (aliemrekskn@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-01-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <cuda_runtime.h>
#include "Algorithm.h"

namespace aek
{
    class CudaAlgorithm2 : public aek::Algorithm
    {
    public:
        CudaAlgorithm2();
        ~CudaAlgorithm2();
        virtual void Apply(cv::InputArray src, cv::OutputArray dst) override final;
        int DivCeil(int numerator, int denominator);

    private:
    };

    __global__ void Algorithm2Kernel(u_int16_t *src, u_int16_t *dst, const size_t srcWidth, const size_t srcHeight, const size_t winWidth, u_int16_t *transposes, u_int16_t *products);

} // namespace aek

#endif // __CUDAALGORITHM2_H__