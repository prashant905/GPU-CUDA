#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int initGaussianKernel(float sigma);

int initGaussianKernel(float *d_kernel, float sigma);

float calcBlurFactor(cv::Mat color);

void getIntensity(cv::Mat color, float *imgIntensity, int w, int h);

float h_guassian_kernel(int a, int b, float sigma);

void gaussianKernel(float *kernel, int r, float sigma);

void applyUnsharpMasking(float *imgSharp, float *imgIn, int w, int h, int nc, float weightFactor);

//kernels
__global__ void d_convolve(float *imgOut, float *img, float *kernel, int w, int h, int nc, int r);

__global__ void d_convolve_constMem(float *imgOut, float *img, int w, int h, int nc, int r);

__global__ void d_unsharpFilter(float *imgOut, float *imgIn, float *imgSmoothed, int w, int h, int nc, float k);

__global__ void d_horizontalAbsDifferences(float *imgIn, float *imgBlurred, float *diffHor, float *blurHor, int w, int h);

__global__ void d_verticalAbsDifferences(float *imgIn, float *imgBlurred, float *diffVer, float *blurVer, int w, int h);

