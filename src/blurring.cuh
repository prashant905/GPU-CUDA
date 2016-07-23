#include <cuda_runtime.h>
#include "aux.cuh"

//constant variable
__constant__ float d_const_kernel[(2*20+1)*(2*20+1)]; //r_max = 20



void convolution(float* d_img, float sigma, size_t w, size_t h, size_t nc);


//kernels
__global__ void convolutionGPU_shared_const_Memory(float* imgIn, float* imgOut, size_t w, size_t h, size_t nc, size_t wk, size_t hk, size_t r);

__global__ void convolutionGPU_sharedMemory(float* kernel, float* imgIn, float* imgOut, size_t w, size_t h, size_t nc, size_t wk, size_t hk, size_t r);

