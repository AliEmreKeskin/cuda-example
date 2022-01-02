/**
 * @file CudaAlgorithm.cu
 * @author Ali Emre Keskin (aliemrekskn@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-12-28
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "CudaAlgorithm.h"

namespace aek
{
    CudaAlgorithm::CudaAlgorithm()
    {
    }

    CudaAlgorithm::~CudaAlgorithm()
    {
    }

    void CudaAlgorithm::Apply(cv::InputArray src, cv::OutputArray dst)
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
                aek::CudaAlgorithm::Transpose(window, transpose);
                aek::CudaAlgorithm::Matmul(window, transpose, product);
                output(cv::Rect(tl, windowSize_)) += product;
            }
        }
    }

    void CudaAlgorithm::Transpose(cv::InputArray src, cv::OutputArray dst)
    {
        dst.createSameSize(src, CV_16UC1);

        u_int16_t *d_src;
        u_int16_t *d_dst;

        size_t size = src.size().width * src.size().height * sizeof(u_int16_t);

        cudaMalloc((void **)&d_src, size);
        cudaMalloc((void **)&d_dst, size);

        cudaMemcpy(d_src, src.getMat().data, size, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(src.size().width, src.size().height);
        dim3 blocksPerGrid(1, 1);
        aek::TransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, src.size().width, src.size().height);

        cudaMemcpy(dst.getMatRef().data, d_dst, size, cudaMemcpyDeviceToHost);

        cudaFree(d_src);
        cudaFree(d_dst);
    }

    void CudaAlgorithm::Matmul(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst)
    {
        dst.createSameSize(src1, CV_16UC1);

        u_int16_t *d_A;
        u_int16_t *d_B;
        u_int16_t *d_C;

        size_t N = src1.size().width;
        size_t size = N * N * sizeof(u_int16_t);

        cudaMalloc((void **)&d_A, size);
        cudaMalloc((void **)&d_B, size);
        cudaMalloc((void **)&d_C, size);

        cudaMemcpy(d_A, src1.getMat().data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, src2.getMat().data, size, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(N, N);
        dim3 blocksPerGrid(1, 1);
        if (N * N > 512)
        {
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
        }

        MatmulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        cudaMemcpy(dst.getMatRef().data, d_C, size, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    __global__ void TransposeKernel(u_int16_t *src, u_int16_t *dst, const size_t srcWidth, const size_t srcHeight)
    {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        dst[x * srcHeight + y] = src[y * srcWidth + x];
    }

    __global__ void MatmulKernel(u_int16_t *A, u_int16_t *B, u_int16_t *C, size_t N)
    {
        int ROW = blockIdx.y * blockDim.y + threadIdx.y;
        int COL = blockIdx.x * blockDim.x + threadIdx.x;

        float tmpSum = 0;

        if (ROW < N && COL < N)
        {
            // each thread computes one element of the block sub-matrix
            for (int i = 0; i < N; i++)
            {
                tmpSum += A[ROW * N + i] * B[i * N + COL];
            }
        }
        C[ROW * N + COL] = tmpSum;
    }

} // namespace aek
