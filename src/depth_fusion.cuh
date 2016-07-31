#include <cuda_runtime.h>
#include <Eigen/Dense>

__constant__ float focal_x = 517.3f;
__constant__ float focal_y = 516.5f;
__constant__ float optcenter_x = 318.6f;
__constant__ float optcenter_y = 255.3f;

__constant__ float focal_length = 1.0f;
__constant__ float baseline = 1.0f;


// store image on texture memory
void loadLRImage(float* d_lrColor, size_t wLR, size_t hLR);

// initialization routines for rotation matrices for SR (key-frame) <-> Groundtruth, LR (new frame, several frames for median) <-> Groundtruth
void init_T_SR(Eigen::Matrix4f rot);

void init_T_inv_SR(Eigen::Matrix4f rot);

void init_T_inv_LR(Eigen::Matrix4f* rot, size_t numFrames = 1);

void init_T_LR(Eigen::Matrix4f* rot, size_t numFrames = 1);

// preprocessing of depth (zero values to INT_MAX) and initialization of weights
void initSRDepthAndWeights(int* d_depth_sr, int* d_weights_sr, size_t wSR, size_t hSR);

// initialization of weights
void initSRWeights(float* d_depth_sr, float* d_weights_sr, size_t wSR, size_t hSR);

// methods for the actual depth and color fusion ...

// ... iterating over LR frame, updating with atomicMin/Max
void fuseDepthAndWeights(int* d_depth_sr, int* d_weights_sr, int* d_depth_lr, float* d_color_sr, float BlurrFactor, size_t s, size_t wSR, size_t hSR);

// ... iterating over key-frame, search along the ray between pixel and optical center, using weighted averaging
void fuseDepthAndWeightsWithRay(float* d_depth_sr, float* d_weights_sr, float* d_depth_lr, float* d_color_sr, float* d_color_lr, float BlurrFactor, size_t s, size_t wSR, size_t hSR);

// ... iterating over key-frame, search along the ray between pixel and optical center, using weighted median (for color fusion)
void fuseDepthAndWeightsWithRayMedian(float* d_depth_sr, float* d_weights_sr, float* d_depth_lr, float* d_color_sr, float* d_color_lr, float* BlurrFactors, size_t numFrames, size_t s, size_t wSR, size_t hSR);

// conversion from float to int and back (as atomicMin/Max doesn't exist for floats)
void transformDepthFloatToInt(float* d_depth, int* d_i_depth, size_t w, size_t h);

void transformDepthIntToFloat(int* d_i_depth, float* d_depth, size_t w, size_t h);

// kernels

__global__ void computeDepthFloatToInt(float* d_depth, int* d_i_depth, size_t w, size_t h);

__global__ void computeDepthIntToFloat( int* d_i_depth, float* d_depth, size_t w, size_t h);

__device__ float2 transform3Dto2D(float x, float y, float z, size_t s = 2);

__device__ float3 transform2Dto3D(float x, float y, float depth_val, size_t s = 2);

__global__ void initAll(float* array, float value, size_t w, size_t h, size_t nc);

__global__ void initAll(int* array, int value, size_t w, size_t h, size_t nc);

__global__ void preprocessDepth(int* depth, size_t w, size_t h);

__global__ void postprocessDepth(int* depth, size_t w, size_t h);

__global__ void calcInitialWeights(int* depthSR, int* weightsSR, size_t wSR, size_t hSR);

__global__ void calcInitialWeights(float* depthSR, float* weightsSR, size_t wSR, size_t hSR);

__device__ int calcWeight(int depth, float f, float b);

__device__ float calcWeight(float depth, float f, float b);

__device__ float3 applyTransformationMatricesSRtoLR(float3 pSR, size_t frame = 0);

__device__ float3 applyTransformationMatricesLRtoSR(float3 pLR, size_t frame = 0);

__global__ void updateStep(int* lrDepth, int* srDepth, int* srWeights, size_t wLR, size_t hLR, size_t wSR, size_t hSR);

__global__ void fuseUpdates(int *srDepth, int *srUpdateDepth, int *srWeights, int *srUpdateWeights, float* srColor, float BlurrFactor, size_t wSR, size_t hSR);

__global__ void fuseFrame(float *srDepth, float *srWeights, float* srColor, float* lrDepth, float* lrColor, float BlurrFactor, size_t wSR, size_t hSR, size_t s);

__global__ void fuseFrameMedian(float *srDepth, float *srWeights, float* srColor, float* lrDepth, float* lrColor, float* BlurrFactors, size_t numFrames, size_t wSR, size_t hSR, size_t s);

