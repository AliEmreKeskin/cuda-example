/**
 * @file CudaAlgorithm2.cu
 * @author Ali Emre Keskin (aliemrekskn@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-01-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "CudaAlgorithm2.h"
#include "Timer.h"

namespace aek
{
    CudaAlgorithm2::CudaAlgorithm2()
    {
    }

    CudaAlgorithm2::~CudaAlgorithm2()
    {
    }

    void CudaAlgorithm2::Apply(cv::InputArray src, cv::OutputArray dst)
    {
        aek::Timer timer("cuda2_apply");

        dst.createSameSize(src, CV_16UC1);
        cv::Mat output = dst.getMatRef();
        // output = 0;
        cv::Mat input = src.getMat();

        timer.Tic("allocation");
        u_int16_t *d_src;
        u_int16_t *d_dst;
        u_int16_t *d_transposes;
        u_int16_t *d_products;
        size_t size = src.size().area() * sizeof(u_int16_t);
        size_t numOfWindows = (src.size().width - windowSize_.width + 1) * (src.size().height - windowSize_.height + 1);
        size_t windowedSize = numOfWindows * windowSize_.area() * sizeof(u_int16_t);
        cudaMalloc((void **)&d_src, size);
        cudaMalloc((void **)&d_dst, size);
        cudaMalloc((void **)&d_transposes, windowedSize);
        cudaMalloc((void **)&d_products, windowedSize);
        timer.Toc();

        timer.Tic("set_to_zero");
        cudaMemset(d_dst, 0, size);
        cudaMemset(d_products, 0, windowedSize);
        timer.Toc();

        timer.Tic("upload");
        cudaMemcpy(d_src, input.data, size, cudaMemcpyHostToDevice);
        timer.Toc();

        dim3 threadsPerBlock(32, 32);
        cv::Size blockCoverage(threadsPerBlock.x, threadsPerBlock.y);
        dim3 blocksPerGrid(aek::CudaAlgorithm2::DivCeil(src.size().width, blockCoverage.width), aek::CudaAlgorithm2::DivCeil(src.size().height, blockCoverage.height));

        // invoke kernel
        timer.Tic("Algorithm2Kernel");
        aek::Algorithm2Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, src.size().width, src.size().height, windowSize_.width, d_transposes, d_products);
        timer.Toc();

        timer.Tic("download");
        cudaMemcpy(output.data,d_dst,size,cudaMemcpyDeviceToHost);
        timer.Toc();
    }

    int CudaAlgorithm2::DivCeil(int numerator, int denominator)
    {
        std::div_t res = std::div(numerator, denominator);
        return res.rem ? (res.quot + 1) : res.quot;
    }

    __global__ void Algorithm2Kernel(u_int16_t *src, u_int16_t *dst, const size_t srcWidth, const size_t srcHeight, const size_t winWidth, u_int16_t *transposes, u_int16_t *products)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= srcWidth || y >= srcHeight)
        {
            return;
        }

        int index = y * srcWidth + x;
        int windowedIndex = index * winWidth * winWidth;

        // transpose
        for (size_t i = 0; i < winWidth; i++)
        {
            for (size_t j = 0; j < winWidth; j++)
            {
                transposes[windowedIndex + j * winWidth + i] = src[index + i * srcWidth + j];
            }
        }

        // matmul
        for (size_t i = 0; i < winWidth; i++)
        {
            for (size_t j = 0; j < winWidth; j++)
            {
                for (size_t k = 0; k < winWidth; k++)
                {
                    products[windowedIndex + j * winWidth + i] += src[index + j * srcWidth + k] * transposes[windowedIndex + k * winWidth + i];
                }
            }
        }

        // elementwise sum
        for (size_t i = 0; i < winWidth; i++)
        {
            for (size_t j = 0; j < winWidth; j++)
            {
                dst[index + i * srcWidth + j] += products[windowedIndex + i * winWidth + j];
            }
        }
    }
}