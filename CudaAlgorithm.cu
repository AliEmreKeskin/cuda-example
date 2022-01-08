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
#include "Timer.h"

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

        aek::Timer timer("cuda_apply");

        timer.Tic("allocation");
        cv::cuda::GpuMat d_window = cv::cuda::createContinuous(windowSize_, src.type());
        cv::cuda::GpuMat d_transpose = cv::cuda::createContinuous(windowSize_, src.type());
        cv::cuda::GpuMat d_product = cv::cuda::createContinuous(windowSize_, src.type());
        cv::cuda::GpuMat d_output = cv::cuda::createContinuous(dst.size(), dst.type());
        cv::cuda::GpuMat d_input = cv::cuda::createContinuous(src.size(), src.type());
        timer.Toc();

        timer.Tic("upload");
        d_input.upload(src);
        timer.Toc();

        timer.Tic("set_to_zero");
        d_output.setTo(0);
        timer.Toc();

        timer.Tic("all_windows");
        cv::Point tl;
        for (tl.y = 0; tl.y < src.size().height - windowSize_.height; tl.y++)
        {
            for (tl.x = 0; tl.x < src.size().width - windowSize_.width; tl.x++)
            {
                // timer.Tic("window");
                aek::CudaAlgorithm::CopyRoi(d_input,cv::Rect(tl, windowSize_),d_window);
                aek::CudaAlgorithm::Transpose(d_window, d_transpose);
                aek::CudaAlgorithm::Matmul(d_window, d_transpose, d_product);
                aek::CudaAlgorithm::AddRoi(d_product, d_output, tl);
                // timer.Toc();
            }
        }
        timer.Toc();

        timer.Tic("download");
        d_output.download(output);
        timer.Toc();
    }

    void CudaAlgorithm::Transpose(cv::InputArray src, cv::OutputArray dst)
    {
        if (src.isMat() && dst.isMat())
        {
            aek::CudaAlgorithm::Transpose(src.getMat(), dst.getMatRef());
        }
        else if (src.isGpuMat() && dst.isGpuMat())
        {
            aek::CudaAlgorithm::Transpose(src.getGpuMat(), dst.getGpuMatRef());
        }
    }

    void CudaAlgorithm::Transpose(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst)
    {
        // aek::Timer timer("transpose");
        dst.create(cv::Size(src.size().height, src.size().width), src.type());

        dim3 threadsPerBlock(src.size().width, src.size().height);
        dim3 blocksPerGrid(1, 1);
        aek::TransposeKernel<<<blocksPerGrid, threadsPerBlock>>>((u_int16_t *)src.data, (u_int16_t *)dst.data, src.size().width, src.size().height);
    }

    void CudaAlgorithm::Transpose(const cv::Mat &src, cv::Mat &dst)
    {
        dst.create(cv::Size(src.size().height, src.size().width), src.type());

        u_int16_t *d_src;
        u_int16_t *d_dst;

        size_t size = src.size().width * src.size().height * sizeof(u_int16_t);

        cudaMalloc((void **)&d_src, size);
        cudaMalloc((void **)&d_dst, size);

        cudaMemcpy(d_src, src.data, size, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(src.size().width, src.size().height);
        dim3 blocksPerGrid(1, 1);
        aek::TransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, src.size().width, src.size().height);

        cudaMemcpy(dst.data, d_dst, size, cudaMemcpyDeviceToHost);

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

    void CudaAlgorithm::Matmul(const cv::cuda::GpuMat &src1, const cv::cuda::GpuMat &src2, cv::cuda::GpuMat &dst)
    {
        // aek::Timer timer("matmul");
        dst.create(src1.size(), CV_16UC1);

        dim3 threadsPerBlock(dst.size().width, dst.size().height);
        dim3 blocksPerGrid(1, 1);
        aek::MatmulKernel<<<blocksPerGrid, threadsPerBlock>>>((u_int16_t *)src1.data, (u_int16_t *)src2.data, (u_int16_t *)dst.data, dst.size().width);
    }

    void CudaAlgorithm::AddRoi(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, const cv::Point &tl)
    {
        // aek::Timer timer("add_roi");
        dim3 threadsPerBlock(src.size().width, src.size().height);
        dim3 blocksPerGrid(1, 1);
        aek::AddRoiKernel<<<blocksPerGrid, threadsPerBlock>>>((u_int16_t *)src.data, src.size().width, (u_int16_t *)dst.data, dst.size().width, tl);
    }

    void CudaAlgorithm::CopyRoi(const cv::cuda::GpuMat &src, const cv::Rect &rect, cv::cuda::GpuMat &dst)
    {
        // aek::Timer timer("copy_roi");
        dst.create(src.size(), src.type());
        src(rect).copyTo(dst);
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

    __global__ void AddRoiKernel(u_int16_t *src, const size_t srcWidth, u_int16_t *dst, const size_t dstWidth, const cv::Point tl)
    {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        dst[(y + tl.y) * dstWidth + (tl.x + x)] += src[y * srcWidth + x];
    }

} // namespace aek
