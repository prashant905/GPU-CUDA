#include <cuda_runtime.h>
#include "convolution.cuh"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "aux.cuh"

#include "cublas_v2.h"

__constant__ float constKernel[32 * 32];

int initGaussianKernel(float sigma){

	int r = ceil(3*sigma);
	float *kernel = new float[(2*r + 1) * (2*r + 1)];

	gaussianKernel(kernel, r, sigma); // pre-calculate Gaussian kernel to blur input image

	cudaMemcpyToSymbol(constKernel, kernel, (2*r + 1) * (2*r + 1) * sizeof(float));
	CUDA_CHECK;

	delete[] kernel;

	return r;

}

int initGaussianKernel(float *d_kernel, float sigma){

	int r = ceil(3*sigma);
	float *kernel = new float[(2*r + 1) * (2*r + 1)];

	gaussianKernel(kernel, r, sigma); // pre-calculate Gaussian kernel to blur input image

	cudaMemcpy(d_kernel, kernel, (2*r + 1) * (2*r + 1) * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;

	delete[] kernel;

	return r;

}


float calcBlurFactor(cv::Mat color){
	//cv::cvtColor(color, grayscale, CV_BGR2GRAY);

	size_t w = color.cols;
	size_t h = color.rows;
	size_t nc = color.channels();

	cv::Mat grayscale = cv::Mat(h, w, CV_32FC1);
	
	float sigma = 3.0f;
	int r = initGaussianKernel(sigma);
	//int r = ceil(3*sigma);
	//float *kernel = new float[(2*r + 1) * (2*r + 1)];

	//gaussianKernel(kernel, r, sigma); // pre-calculate Gaussian kernel to blur input image

	float *imgColor = new float[w * h * nc];
	convert_mat_to_layered(imgColor, color);

	float *imgBlurred = new float[w * h * nc]; 
	float *d_imgColor;
	float *d_imgBlurred;
	cudaMalloc(&d_imgColor, w * h * nc * sizeof(float));
	CUDA_CHECK;
	cudaMemcpy(d_imgColor, imgColor, w * h * nc * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc(&d_imgBlurred, w * h * nc * sizeof(float));
	CUDA_CHECK;
	//cudaMemcpyToSymbol(constKernel, kernel, (2*r + 1) * (2*r + 1) * sizeof(float));
	//CUDA_CHECK;

	dim3 block = dim3(32,32,1);
	dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
	d_convolve_constMem<<<grid, block>>>(d_imgBlurred, d_imgColor, w, h, nc, r); //Gaussian convolution kernel creates a blurred image
	cudaMemcpy(imgBlurred, d_imgBlurred, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost); //Copy resulting image to host
	CUDA_CHECK;

	cv::Mat matBlurred = cv::Mat(h, w, CV_32FC3);
	convert_layered_to_mat(matBlurred, imgBlurred);

	//showImage("Blurred Image", matBlurred, 100, 100);
	
	float *imgBlurredIntensity = new float[w * h];
	getIntensity(matBlurred, imgBlurredIntensity, w, h); // Get blurred image intensity matrix

	float *imgIntensity = new float[w * h];
	getIntensity(color, imgIntensity, w, h); // Get original image intensity matrix

	float *d_imgIntensity;
	float *d_imgBlurredIntensity;
	float *d_imgColorHorDiff;
	float *d_imgColorVerDiff;
	float *d_imgBlurredHorDiff;
	float *d_imgBlurredVerDiff;

	cudaMalloc(&d_imgIntensity, w * h * sizeof(float));
	CUDA_CHECK;
	cudaMemcpy(d_imgIntensity, imgIntensity, w * h * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc(&d_imgBlurredIntensity, w * h * sizeof(float));
	CUDA_CHECK;
	cudaMemcpy(d_imgBlurredIntensity, imgBlurredIntensity, w * h * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc(&d_imgColorHorDiff, w * h * sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_imgColorVerDiff, w * h * sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_imgBlurredHorDiff, w * h * sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_imgBlurredVerDiff, w * h * sizeof(float));
	CUDA_CHECK;

	d_horizontalAbsDifferences<<<grid, block>>>(d_imgIntensity, d_imgBlurredIntensity, d_imgColorHorDiff, d_imgBlurredHorDiff, w, h); // calculate absolute horizontal differences 
	d_verticalAbsDifferences<<<grid, block>>>(d_imgIntensity, d_imgBlurredIntensity, d_imgColorVerDiff, d_imgBlurredVerDiff, w, h); // calculate absolute vertical differences 

	cublasHandle_t handle;
	cublasCreate(&handle);

	// Host pointers to store results of sum reduction operations
	float *sumFVer = new float[1]; // Sum of original image vertical absolute differences
	float *sumFHor = new float[1]; // Sum of original image horizontal absolute differences
	float *sumVVer = new float[1]; // Sum of vertical absolute difference between original and blurred image
	float *sumVHor = new float[1]; // Sum of horizontal absolute difference between original and blurred image
	
	
	cublasSasum(handle, w * h, d_imgColorVerDiff, 1, sumFVer); // Calculate sumFVer
	cublasSasum(handle, w * h, d_imgColorHorDiff, 1, sumFHor); // Calculate sumFHor
	cublasSasum(handle, w * h, d_imgBlurredVerDiff, 1, sumVVer); // Calculate sumVVer
	cublasSasum(handle, w * h, d_imgBlurredHorDiff, 1, sumVHor); // Calculate sumVHor

	cublasDestroy(handle);

	//cudaMemcpy(imgIntensity, d_imgBlurredVerDiff, w * h * sizeof(float), cudaMemcpyDeviceToHost);
	//CUDA_CHECK;

	float b_FVer = (sumFVer[0] - sumVVer[0]) / sumFVer[0];
	float b_FHor = (sumFHor[0] - sumVHor[0]) / sumFHor[0];

	float blurFactor = fmax(b_FVer, b_FHor);

	cudaFree(d_imgColor);
	cudaFree(d_imgBlurred);
	cudaFree(d_imgIntensity);
	cudaFree(d_imgBlurredIntensity);
	cudaFree(d_imgColorHorDiff);
	cudaFree(d_imgColorVerDiff);
	cudaFree(d_imgBlurredHorDiff);
	cudaFree(d_imgBlurredVerDiff);
	
	return 1.0f - blurFactor;	

}

void getIntensity(cv::Mat color, float *imgIntensity, int w, int h){

	cv::Mat grayscale = cv::Mat(h, w, CV_32FC1);
	cv::cvtColor(color, grayscale, CV_BGR2GRAY);

	for(int y = 0; y < h; y++){
		for(int x = 0; x < w; x++){
			imgIntensity[x + y * w] = grayscale.at<float>(y, x);
		}
	}

}

void applyUnsharpMasking(float *imgOut, float *imgIn, int w, int h, int nc, float weightFactor){

	float sigma = 3.0f;
	int r = initGaussianKernel(sigma);

	float *d_imgIn;
	float *d_imgSmoothed;
	float *d_imgSharp;
	cudaMalloc(&d_imgIn, w * h * nc * sizeof(float));
	CUDA_CHECK;
	cudaMemcpy(d_imgIn, imgIn, w * h * nc * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc(&d_imgSmoothed, w * h * nc * sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_imgSharp, w * h * nc * sizeof(float));
	CUDA_CHECK;

	dim3 block = dim3(32,32,1);
	dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

	d_convolve_constMem<<<grid, block>>>(d_imgSmoothed, d_imgIn, w, h, nc, r); // Generate blurred image

	d_unsharpFilter<<<grid, block>>>(d_imgSharp, d_imgIn, d_imgSmoothed, w, h, nc, weightFactor); // apply Unsharp Masking

	cudaMemcpy(imgOut, d_imgSharp, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;

	cudaFree(d_imgIn);
	cudaFree(d_imgSmoothed);
	cudaFree(d_imgSharp);	

}

float h_guassian_kernel(int a, int b, float sigma){

	return exp(-(a*a + b*b)/(2*sigma*sigma))/(2*M_PI*sigma*sigma);

}

void gaussianKernel(float *kernel, int r, float sigma){

	int step = 1;

	int a, b;
	float kernel_total = 0;

	// Kernel computation
	for(b = -r; b <= r; b += step)
	{
		for(a = -r; a <= r; a += step)
		{
			if(sqrt(a*a + b*b) <= r)
			{
				kernel_total += h_guassian_kernel(a, b, sigma);
				kernel[(a + r) + (b + r)*(2*r + 1)] = h_guassian_kernel(a, b, sigma);
			}
			else
			{
				kernel[(a + r) + (b + r)*(2*r + 1)] = 0.0;
			}
		}
	}

	float max_kernel = h_guassian_kernel(0, 0, sigma);	
	max_kernel /= kernel_total;

	// Kernel normalization and scaling
	for(int i=0; i < (2*r+1) * (2*r+1); i++)
	{
		kernel[i] /= kernel_total;
	}

}

__global__ void d_convolve_constMem(float *imgOut, float *img, int w, int h, int nc, int r){
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	
	for(int i = 0; i < nc; i++)
	{
		float result = 0;
		if(x < w && y < h)
		{
			for(int b = -r; b < r + 1; b++)
			{	
				for(int a = -r; a < r + 1; a++)
				{	
					float kernel_val = constKernel[(a + r) + (b + r) * (2*r + 1)];
					int coord_x = min(max(x - a, 0), w - 1);
					int coord_y = min(max(y - b, 0), h - 1);
					result += kernel_val * img[coord_x + w * coord_y + i * w * h];
				}
			}
			imgOut[x + w * y + i * w * h] = result;
		}
		
	}

}

__global__ void d_convolve(float *imgOut, float *img, float *kernel, int w, int h, int nc, int r){
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	
	for(int i = 0; i < nc; i++)
	{
		float result = 0;
		if(x < w && y < h)
		{
			for(int b = -r; b < r + 1; b++)
			{	
				for(int a = -r; a < r + 1; a++)
				{	
					float kernel_val = kernel[(a + r) + (b + r) * (2*r + 1)];
					int coord_x = min(max(x - a, 0), w - 1);
					int coord_y = min(max(y - b, 0), h - 1);
					result += kernel_val * img[coord_x + w * coord_y + i * w * h];
				}
			}
			imgOut[x + w * y + i * w * h] = result;
		}
		
	}

}



__global__ void d_unsharpFilter(float *imgOut, float *imgIn, float *imgSmoothed, int w, int h, int nc, float k){
	
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	

	for(int i = 0; i < nc; i++){
		int idx = x + w * y + i * w * h;
		imgOut[idx] = (1 + k) * imgIn[idx] - k * imgSmoothed[idx];
	}

}

__global__ void d_horizontalAbsDifferences(float *imgIn, float *imgBlurred, float *diffHor, float *blurHor, int w, int h){

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if(x < w && y < h){
		int x_l = max(x - 1, 0);
		float d_fHor = fabs(imgIn[x + y * w] - imgIn[x_l + y * w]);
		diffHor[x + y * w] = d_fHor; 
		float d_bHor = fabs(imgBlurred[x + y * w] - imgBlurred[x_l + y * w]);
		blurHor[x + y * w] = fmax(0, d_fHor - d_bHor);	
	}

}

__global__ void d_verticalAbsDifferences(float *imgIn, float *imgBlurred, float *diffVer, float *blurVer, int w, int h){

	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if(x < w && y < h){
		int y_u = max(y - 1, 0);
		float d_fVer = fabs(imgIn[x + y * w] - imgIn[x + y_u * w]);
		diffVer[x + y * w] = d_fVer; 
		float d_bVer = fabs(imgBlurred[x + y * w] - imgBlurred[x + y_u * w]);
		blurVer[x + y * w] = fmax(0, d_fVer - d_bVer);	
	}

}








