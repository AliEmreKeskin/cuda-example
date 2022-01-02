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
#include <cuda_runtime.h>
#include "Algorithm.h"
namespace aek
{
    class CudaAlgorithm : public aek::Algorithm
    {
    public:
        CudaAlgorithm();
        ~CudaAlgorithm();
        virtual void Apply(cv::InputArray src, cv::OutputArray dst) override final;

    protected:
        virtual void Transpose(cv::InputArray src, cv::OutputArray dst) override final;
        virtual void Matmul(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst) override final;
    };

    __global__ void TransposeKernel(u_int16_t *src, u_int16_t *dst, const size_t srcWidth, const size_t srcHeight);
    __global__ void MatmulKernel(u_int16_t *A, u_int16_t *B, u_int16_t *C, size_t N);

} // namespace aek
#endif // __CUDAALGORITHM_H__