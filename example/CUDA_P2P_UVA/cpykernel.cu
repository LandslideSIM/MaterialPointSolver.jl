#include <cuda_runtime.h>
#include <iostream>

// CUDA核函数，用于复制数据
__global__ void copyKernel(float* dest, const float* src, size_t numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        dest[idx] = src[idx];
    }
}

// 初始化为0的CUDA内核
__global__ void initZero(float* ptr, size_t size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        ptr[idx] = 0.0f;
    }
}

// 初始化为1的CUDA内核
__global__ void initOne(float* ptr, size_t size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        ptr[idx] = 1.0f;
    }
}

// 用于检查CUDA调用的状态
void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}

int main() {
    const size_t Nx = 1024;
    const size_t Ny = 1024;
    const size_t Nz = 1024;
    const size_t totalElements = Nx * Ny * Nz;
    const size_t size = totalElements * sizeof(float);
    
    float *A, *B;
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // 创建事件
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 在GPU 0上分配并初始化数组A
    cudaSetDevice(0);
    checkCudaStatus(cudaMalloc(&A, size));
    initZero<<<(totalElements + 255) / 256, 256>>>(A, totalElements);
    checkCudaStatus(cudaGetLastError());
    checkCudaStatus(cudaDeviceSynchronize());

    // 在GPU 1上分配并初始化数组B
    cudaSetDevice(1);
    checkCudaStatus(cudaMalloc(&B, size));
    initOne<<<(totalElements + 255) / 256, 256>>>(B, totalElements);
    checkCudaStatus(cudaGetLastError());
    checkCudaStatus(cudaDeviceSynchronize());

    // 启用P2P访问
    cudaSetDevice(0);
    checkCudaStatus(cudaDeviceEnablePeerAccess(1, 0));

    // 记录复制开始的时间
    cudaEventRecord(start);

    // 执行复制核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, totalElements);
    checkCudaStatus(cudaGetLastError());
    checkCudaStatus(cudaDeviceSynchronize());

    // 记录复制结束的时间并等待事件完成
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算并打印所用时间
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for manual P2P copy: " << milliseconds / 1000.0f << " seconds." << std::endl;

    // 清理
    cudaSetDevice(0);
    cudaFree(A);
    cudaSetDevice(1);
    cudaFree(B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
