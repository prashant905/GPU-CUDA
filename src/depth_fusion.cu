#include "depth_fusion.cuh"
#include "global.cuh"
#include "aux.cuh"
#include <cstdio>
#include <iostream>
#include <climits>


void loadLRImage(float* d_lrColor, size_t wLR, size_t hLR){
	cudaUnbindTexture(&texRefColorR);
	cudaUnbindTexture(&texRefColorG);
	cudaUnbindTexture(&texRefColorB);
    texRefColorR.addressMode[0] = cudaAddressModeClamp; // clamp x to border
    texRefColorR.addressMode[1] = cudaAddressModeClamp; // clamp y to border
    texRefColorR.filterMode = cudaFilterModeLinear; // linear interpolation
    texRefColorR.normalized = true; // access as ((x+0.5f)/w,(y+0.5f)/h), not as (x+0.5f,y+0.5f)
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, &texRefColorR, d_lrColor, &desc, wLR, hLR, wLR * sizeof(d_lrColor[0]));
    CUDA_CHECK;

    texRefColorG.addressMode[0] = cudaAddressModeClamp; // clamp x to border
    texRefColorG.addressMode[1] = cudaAddressModeClamp; // clamp y to border
    texRefColorG.filterMode = cudaFilterModeLinear; // linear interpolation
    texRefColorG.normalized = true; // access as ((x+0.5f)/w,(y+0.5f)/h), not as (x+0.5f,y+0.5f)
    cudaBindTexture2D(NULL, &texRefColorG, d_lrColor + wLR*hLR, &desc, wLR, hLR, wLR * sizeof(d_lrColor[0]));
    CUDA_CHECK;

    texRefColorB.addressMode[0] = cudaAddressModeClamp; // clamp x to border
    texRefColorB.addressMode[1] = cudaAddressModeClamp; // clamp y to border
    texRefColorB.filterMode = cudaFilterModeLinear; // linear interpolation
    texRefColorB.normalized = true; // access as ((x+0.5f)/w,(y+0.5f)/h), not as (x+0.5f,y+0.5f)
    cudaBindTexture2D(NULL, &texRefColorB, d_lrColor + 2*wLR*hLR, &desc, wLR, hLR, wLR * sizeof(float));
    CUDA_CHECK;

}

void init_T_SR(Eigen::Matrix4f rot){
	float data[12];
	for (int y = 0; y < 3; ++y){
		for (int x = 0; x < 4; ++x){
			data[x + y*4] = rot(y, x);
		}
	}
	cudaMemcpyToSymbol(d_T_SR, data, 3 * 4 * sizeof(float));
}

void init_T_inv_SR(Eigen::Matrix4f rot){
	float data[12];

	for (int y = 0; y < 3; ++y){
		for (int x = 0; x < 4; ++x){
			data[x + y*4] = rot(y, x);
		}
	}
	cudaMemcpyToSymbol(d_T_inv_SR, data, 3 * 4 * sizeof(float));
}

void init_T_inv_LR(Eigen::Matrix4f* rot, size_t numFrames){
	if (numFrames > 20){
		std::cout << "Too many Frames (at most 20)!" << std::endl;
	}
	float* data = new float[12 * numFrames];

	for (int frame = 0; frame < numFrames; ++frame){
		for (int y = 0; y < 3; ++y){
			for (int x = 0; x < 4; ++x){
				data[x + y*4 + frame * 12] = rot[frame](y, x);
			}
		}
	}
	cudaMemcpyToSymbol(d_T_inv_LR, data, numFrames * 12 * sizeof(float));
}

void init_T_LR(Eigen::Matrix4f* rot, size_t numFrames){
	if (numFrames > 20){
		std::cout << "Too many Frames (at most 20)!" << std::endl;
	}
	float* data = new float[12 * numFrames];

	for (int frame = 0; frame < numFrames; ++frame){
		for (int y = 0; y < 3; ++y){
			for (int x = 0; x < 4; ++x){
				data[x + y*4 + frame * 12] = rot[frame](y, x);
			}
		}
	}
	cudaMemcpyToSymbol(d_T_LR, data, numFrames * 12 * sizeof(float));
}


void initSRDepthAndWeights(int* d_depth_sr, int* d_weights_sr, size_t wSR, size_t hSR){
    dim3 block = dim3(32,32,1);
	dim3 grid = dim3((wSR + block.x - 1) / block.x, (hSR + block.y - 1) / block.y, 1);

	preprocessDepth <<<grid, block>>> (d_depth_sr, wSR, hSR);

	calcInitialWeights <<<grid, block>>>(d_depth_sr, d_weights_sr, wSR, hSR);

}

void initSRWeights(float* d_depth_sr, float* d_weights_sr, size_t wSR, size_t hSR){
    dim3 block = dim3(32,32,1);
	dim3 grid = dim3((wSR + block.x - 1) / block.x, (hSR + block.y - 1) / block.y, 1);

	calcInitialWeights <<<grid, block>>>(d_depth_sr, d_weights_sr, wSR, hSR);

}



void fuseDepthAndWeights(int* d_depth_sr, int* d_weights_sr, int* d_depth_lr, float* d_color_sr, float BlurrFactor, size_t s, size_t wSR, size_t hSR){
	size_t wLR = wSR/s;
    size_t hLR = hSR/s;

    dim3 block = dim3(64,16,1);
    dim3 grid = dim3((wSR + block.x - 1) / block.x, (hSR + block.y - 1) / block.y, 1);
    dim3 gridColor = dim3((wSR + block.x - 1) / block.x, (hSR + block.y - 1) / block.y, 3);
    dim3 gridLR = dim3((wLR + block.x - 1) / block.x, (hLR + block.y - 1) / block.y, 1);

	int* d_i_update_depth_sr;
	int* d_i_update_weights_sr;
	
	int* i_update_depth_sr = new int[wSR * hSR];
	int* i_update_weights_sr = new int[wSR * hSR];
	
	cudaMalloc(&d_i_update_depth_sr, wSR * hSR * sizeof(int));
	CUDA_CHECK;
	cudaMalloc(&d_i_update_weights_sr,  wSR * hSR * sizeof(int));
	CUDA_CHECK;

	initAll <<<grid, block>>> (d_i_update_depth_sr, INT_MAX, wSR, hSR, 1);
	CUDA_CHECK;
	initAll <<<grid, block>>> (d_i_update_weights_sr, 0, wSR, hSR, 1);
	CUDA_CHECK;

	// overwrite 0 depth values by INT_MAX
	preprocessDepth <<<gridLR, block>>>(d_depth_lr, wLR, hLR);
	CUDA_CHECK;

	initAll <<<grid, block>>> (d_depth_sr, INT_MAX, wSR, hSR, 1);
	initAll <<<grid, block>>> (d_weights_sr, 0, wSR, hSR, 1);

	updateStep <<<gridLR, block>>>(d_depth_lr, d_i_update_depth_sr, d_i_update_weights_sr, wLR, hLR, wSR, hSR);
	CUDA_CHECK;

	initAll<<<gridColor, block>>>(d_color_sr, 0.0f, wSR, hSR, 3);
	CUDA_CHECK;
	fuseUpdates <<<grid, block>>> (d_depth_sr, d_i_update_depth_sr, d_weights_sr, d_i_update_weights_sr, d_color_sr, BlurrFactor, wSR, hSR);
	CUDA_CHECK;

	postprocessDepth <<<grid, block>>> (d_depth_sr, wSR, hSR);

	cudaFree(d_i_update_depth_sr);
	cudaFree(d_i_update_weights_sr);

}

void fuseDepthAndWeightsWithRay(float* d_depth_sr, float* d_weights_sr, float* d_depth_lr, float* d_color_sr, float* d_color_lr, float BlurrFactor, size_t s, size_t wSR, size_t hSR){
	// To see the updates for depth-image
//	initAll(d_depth_sr, 0.0f, wSR, hSR, 1);
//	initAll(d_weights_sr, 0.0f, wSR, hSR, 1);
	// To see the updates for color-image
//	initAll(d_color_sr, 0.0f, wSR, hSR, 3);
    dim3 block = dim3(32,16,1);
    dim3 grid = dim3((wSR + block.x - 1) / block.x, (hSR + block.y - 1) / block.y, 1);
	fuseFrame <<<grid, block>>>(d_depth_sr, d_weights_sr, d_color_sr, d_depth_lr, d_color_lr, BlurrFactor, wSR, hSR, s);
	CUDA_CHECK;
}

void fuseDepthAndWeightsWithRayMedian(float* d_depth_sr, float* d_weights_sr, float* d_depth_lr, float* d_color_sr, float* d_color_lr, float* d_BlurrFactors, size_t numFrames, size_t s, size_t wSR, size_t hSR){
	// To see the updates for depth-image
//	initAll(d_depth_sr, 0.0f, wSR, hSR, 1);
//	initAll(d_weights_sr, 0.0f, wSR, hSR, 1);
	// To see the updates for color-image
//	initAll(d_color_sr, 0.0f, wSR, hSR, 3);
    dim3 block = dim3(32,16,1);
    dim3 grid = dim3((wSR + block.x - 1) / block.x, (hSR + block.y - 1) / block.y, 1);
	fuseFrameMedian<<<grid, block>>>(d_depth_sr, d_weights_sr, d_color_sr, d_depth_lr, d_color_lr, d_BlurrFactors, numFrames, wSR, hSR, s);
	CUDA_CHECK;
}


void transformDepthFloatToInt(float* d_depth, int* d_i_depth, size_t w, size_t h){
    dim3 block = dim3(64,16,1);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    computeDepthFloatToInt <<<grid, block>>> (d_depth, d_i_depth, w, h);
}

void transformDepthIntToFloat(int* d_i_depth, float* d_depth, size_t w, size_t h){
    dim3 block = dim3(64,16,1);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    computeDepthIntToFloat <<<grid, block>>> (d_i_depth, d_depth, w, h);
}


// kernels

__device__ inline float dispErrStdDev(float depth){

	return 1.0f;

}

__global__ void computeDepthFloatToInt(float* d_depth, int* d_i_depth, size_t w, size_t h){
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= w || y >= h){
		return;
	}

	size_t idx = x + w*y;

	d_i_depth[idx] = (int)(d_depth[idx]*scalingIntToFloatFactor);
}

__global__ void computeDepthIntToFloat( int* d_i_depth, float* d_depth, size_t w, size_t h){
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= w || y >= h){
		return;
	}

	size_t idx = x + w*y;

	d_depth[idx] = ((float)d_i_depth[idx]) / scalingIntToFloatFactor;
}

__global__ void initAll(float* array, float value, size_t w, size_t h, size_t nc){
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;
	size_t c = threadIdx.z + blockDim.z * blockIdx.z;

	if (x >= w || y >= h || c >= nc){
		return;
	}

	size_t idx = x + w*y + w*h*c;

	array[idx] = value;
}

__global__ void initAll(int* array, int value, size_t w, size_t h, size_t nc){
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;
	size_t c = threadIdx.z + blockDim.z * blockIdx.z;

	if (x >= w || y >= h || c >= nc){
		return;
	}

	size_t idx = x + w*y + w*h*c;

	array[idx] = value;
}

__global__ void preprocessDepth(int* depth, size_t w, size_t h){
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= w || y >= h){
		return;
	}

	size_t idx = x + w*y;

	if (depth[idx] == 0){
		depth[idx] = INT_MAX;
	}
}

__global__ void postprocessDepth(int* depth, size_t w, size_t h){
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= w || y >= h){
		return;
	}

	size_t idx = x + w*y;

	if (depth[idx] == INT_MAX){
		depth[idx] = 0;
	}
}

__global__ void calcInitialWeights(int* depthSR, int* weightsSR, size_t wSR, size_t hSR){
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;

	size_t idx = x + y * wSR;

	if(x < wSR && y < hSR){
		if(depthSR[idx] == INT_MAX){
			weightsSR[idx] = 0;
		}
		else {
			float depth_val = (float)depthSR[idx];
			depth_val /= scalingIntToFloatFactor;
			float result = (focal_length * baseline / dispErrStdDev(depth_val)) / (depth_val * depth_val);
			weightsSR[idx] = (int)(result*scalingIntToFloatFactor);
		}
	}
}

__global__ void calcInitialWeights(float* depthSR, float* weightsSR, size_t wSR, size_t hSR){
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;

	size_t idx = x + y * wSR;

	if(x < wSR && y < hSR){
		if(depthSR[idx] == 0.0f){
			weightsSR[idx] = 0.0f;
		}
		else {
			float depth_val = (float)depthSR[idx];
			float result = (focal_length * baseline / dispErrStdDev(depth_val)) / (depth_val * depth_val);
			weightsSR[idx] = result;
		}
	}
}

__device__ int calcWeight(int i_depth, float f, float b){
	float depth = ((float)i_depth)/scalingIntToFloatFactor;

	if(depth == 0.0f){
		return 0;
	}
	float depth_val = depth * depth;
	float returnValue =(f * b / dispErrStdDev(depth)) / depth_val;
	return (int)(returnValue * scalingIntToFloatFactor);
}

__device__ float calcWeight(float depth, float f, float b){
	if(depth == 0.0f){
		return 0.0f;
	}
	float depth_val = depth * depth;
	float returnValue =(f * b / dispErrStdDev(depth)) / depth_val;
	return returnValue;
}

__device__ float2 transform3Dto2D(float x, float y, float z, size_t s){

	float2 new_coord;
	new_coord.x = (x / z) * focal_x*s + optcenter_x*s;
	new_coord.y = (y / z) * focal_y*s + optcenter_y*s;

	return new_coord;

}

__device__ float3 transform2Dto3D(float x, float y, float depth_val, size_t s){
	float3 new_coord;

	new_coord.x = ((x - optcenter_x*s) / (focal_x*s)) * depth_val;
	new_coord.y = ((y - optcenter_y*s) / (focal_y*s)) * depth_val;
	new_coord.z = depth_val;

	return new_coord;

}

__device__ float3 applyTransformationMatricesLRtoSR(float3 pLR, size_t frame){
	float3 tmp;
	float3 pSR;

	tmp.x = pLR.x * d_T_inv_LR[0 + frame * 12] + pLR.y * d_T_inv_LR[1 + frame * 12] + pLR.z * d_T_inv_LR[2 + frame * 12] + 1.0f * d_T_inv_LR[3 + frame * 12];
	tmp.y = pLR.x * d_T_inv_LR[4 + frame * 12] + pLR.y * d_T_inv_LR[5 + frame * 12] + pLR.z * d_T_inv_LR[6 + frame * 12] + 1.0f * d_T_inv_LR[7 + frame * 12];
	tmp.z = pLR.x * d_T_inv_LR[8 + frame * 12] + pLR.y * d_T_inv_LR[9 + frame * 12] + pLR.z * d_T_inv_LR[10 + frame * 12] + 1.0f * d_T_inv_LR[11 + frame * 12];

	pSR.x = tmp.x * d_T_SR[0] + tmp.y * d_T_SR[1] + tmp.z * d_T_SR[2] + 1.0f * d_T_SR[3];
	pSR.y = tmp.x * d_T_SR[4] + tmp.y * d_T_SR[5] + tmp.z * d_T_SR[6] + 1.0f * d_T_SR[7];
	pSR.z = tmp.x * d_T_SR[8] + tmp.y * d_T_SR[9] + tmp.z * d_T_SR[10] + 1.0f * d_T_SR[11];

	return pSR;
}

__device__ float3 applyTransformationMatricesSRtoLR(float3 pSR, size_t frame){
	float3 tmp;
	float3 pLR;

	tmp.x = pSR.x * d_T_inv_SR[0] + pSR.y * d_T_inv_SR[1] + pSR.z * d_T_inv_SR[2] + 1.0f * d_T_inv_SR[3];
	tmp.y = pSR.x * d_T_inv_SR[4] + pSR.y * d_T_inv_SR[5] + pSR.z * d_T_inv_SR[6] + 1.0f * d_T_inv_SR[7];
	tmp.z = pSR.x * d_T_inv_SR[8] + pSR.y * d_T_inv_SR[9] + pSR.z * d_T_inv_SR[10] + 1.0f * d_T_inv_SR[11];

	pLR.x = tmp.x * d_T_LR[0 + frame * 12] + tmp.y * d_T_LR[1 + frame * 12] + tmp.z * d_T_LR[2 + frame * 12] + 1.0f * d_T_LR[3 + frame * 12];
	pLR.y = tmp.x * d_T_LR[4 + frame * 12] + tmp.y * d_T_LR[5 + frame * 12] + tmp.z * d_T_LR[6 + frame * 12] + 1.0f * d_T_LR[7 + frame * 12];
	pLR.z = tmp.x * d_T_LR[8 + frame * 12] + tmp.y * d_T_LR[9 + frame * 12] + tmp.z * d_T_LR[10 + frame * 12] + 1.0f * d_T_LR[11 + frame * 12];

	return pLR;
}


__global__ void updateStep(int* lrDepth, int* srDepth, int* srWeights, size_t wLR, size_t hLR, size_t wSR, size_t hSR){
	size_t x = threadIdx.x + blockDim.x * (blockIdx.x);
	size_t y = threadIdx.y + blockDim.y * (blockIdx.y);

	size_t idx = x + y * wLR;


	if (x >= wLR || y >= hLR){
		return;
	}


	float xLR = (float)x + 0.5f;
	float yLR = (float)y + 0.5f;

	int s = wSR / wLR;

	xLR *= s;
	yLR *= s;

	int depthLR = lrDepth[idx];

	if (depthLR == INT_MAX){
		return;
	}

	float3 pLR = transform2Dto3D(xLR, yLR, ((float)depthLR)/scalingIntToFloatFactor, wSR/wLR);

	float3 pSR = applyTransformationMatricesLRtoSR(pLR);

	float2 xySR = transform3Dto2D(pSR.x, pSR.y, pSR.z, wSR/wLR);

	// compute index for 4 neigbouring pixels

	int x0 = (int)(xySR.x - 0.5f);
	int x1 = (int)(xySR.x + 0.5f);
	int y0 = (int)(xySR.y - 0.5f);
	int y1 = (int)(xySR.y + 0.5f);

	int depthLRinKeyFrame = (int)(pSR.z * scalingIntToFloatFactor);

	// update pixel 00
	if (x0 >= 0 && x0 < wSR && y0 >= 0 && y0 < hSR){
		atomicMin(&srDepth[x0 + y0 * wSR], depthLRinKeyFrame);
		atomicMax(&srWeights[x0 + y0 * wSR], calcWeight(depthLR, focal_length, baseline));
	}

	// update pixel 01
	if (x0 >= 0 && x0 < wSR && y1 >= 0 && y1 < hSR){
		atomicMin(&srDepth[x0 + y1 * wSR], depthLRinKeyFrame);
		atomicMax(&srWeights[x0 + y1 * wSR], calcWeight(depthLR, focal_length, baseline));
	}

	// update pixel 10
	if (x1 >= 0 && x1 < wSR && y0 >= 0 && y0 < hSR){
		atomicMin(&srDepth[x1 + y0 * wSR], depthLRinKeyFrame);
		atomicMax(&srWeights[x1 + y0 * wSR], calcWeight(depthLR, focal_length, baseline));
	}

	// update pixel 11
	if (x1 >= 0 && x1 < wSR && y1 >= 0 && y1 < hSR){
		atomicMin(&srDepth[x1 + y1 * wSR], depthLRinKeyFrame);
		atomicMax(&srWeights[x1 + y1 * wSR], calcWeight(depthLR, focal_length, baseline));
	}
}


__global__ void fuseUpdates(int *srDepth, int *srUpdateDepth, int *srWeights, int *srUpdateWeights, float* srColor, float BlurrFactor, size_t wSR, size_t hSR){

	size_t xSR = threadIdx.x + blockDim.x * blockIdx.x;
	size_t ySR = threadIdx.y + blockDim.y * blockIdx.y;

	size_t idxSR = xSR + wSR * ySR;

	int SR_Update = srUpdateDepth[idxSR];

	if(xSR >= wSR || ySR >= hSR) return;

	int SR_W = srWeights[idxSR];
	int weight = srUpdateWeights[idxSR];
	
	if (weight == 0) return;
		//TODO s needed as a parameter (set to 2 implicitly)
		float3 pSR = transform2Dto3D(xSR + 0.5f, ySR + 0.5f, ((float)SR_Update)/scalingIntToFloatFactor);

		float3 pLR = applyTransformationMatricesSRtoLR(pSR);
		//TODO s needed as a parameter (set to 2 implicitly)
		float2 xyLR = transform3Dto2D(pLR.x, pLR.y, pLR.z);

		int tmpx = (int)xyLR.x;
		int tmpy = (int)xyLR.y;
		tmpx /= 2;
		tmpy /= 2;

	if(SR_W == 0){
		// no valid value in keyframe yet -> overwrite
		srDepth[idxSR] = SR_Update;
		srWeights[idxSR] = weight;

		if (weight == 0){
			return;
		}
		//TODO s needed as a parameter (set to 2 implicitly)
		float3 pLR3D = transform2Dto3D(xyLR.x, xyLR.y, pLR.z);

		float3 pSR2 = applyTransformationMatricesLRtoSR(pLR3D);
		//TODO s needed as a parameter (set to 2 implicitly)
		float2 xySR = transform3Dto2D(pSR2.x, pSR2.y, pSR2.z);

		if (xyLR.x < 0.0f || xyLR.x > (float)wSR || xyLR.y < 0.0f || xyLR.y > (float)hSR){
			return;
		}
		
		size_t c=0;
		size_t idxSRColor = idxSR + wSR*hSR*c;
		srColor[idxSRColor] = tex2D(texRefColorR, xyLR.x/(float)wSR, xyLR.y/(float)hSR);
		
		c = 1;
		idxSRColor = idxSR + wSR*hSR*c;
		srColor[idxSRColor] = tex2D(texRefColorG, xyLR.x/(float)wSR, xyLR.y/(float)hSR);
		
		c = 2;
		idxSRColor = idxSR + wSR*hSR*c;
		srColor[idxSRColor] = tex2D(texRefColorB, xyLR.x/(float)wSR, xyLR.y/(float)hSR);
		
		return;
	}

//	Averaging
//
//	int SR_Depth = srDepth[idxSR];
//	int SR_W = srWeights[idxSR];
//	int epsilon = 200; 
//
//	if(abs(SR_Depth - SR_Update) < epsilon){
////		printf("Thread %i averages!\n", idx);
//		if (SR_W + weight == 0){
//			printf("Division by 0 by thread %i\n", idxSR);
//		}
//		float resultDepthFloat =  (((float)SR_W * (float)SR_Depth + (float)weight * (float)SR_Update)) / (((float)SR_W + (float)weight));
////		printf("result depth (float): %f\n", resultDepthFloat);
//		srDepth[idxSR] = (int) ( resultDepthFloat );
////		printf("result depth: %i\n", srDepth[idx]);
//
////		printf("result weight: %i\n", srWeights[idx]);
//
//		// perhaps take resultDepthFloat instead of SR_Update
//		float3 pSR = transform2Dto3D(xSR + 0.5f, ySR + 0.5f, ((float)SR_Update)/scalingIntToFloatFactor);
//
//		// Upsampling taken into account??
//		float3 pLR = applyTransformationMatricesSRtoLR(pSR);
//
//		float2 xyLR = transform3Dto2D(pLR.x, pLR.y, pLR.z);
//
//		size_t c=0;
//		size_t idxSRColor = idxSR + wSR*hSR*c;
//		float lrColor = tex2D(texRefColorR, xyLR.x/(float)wSR, xyLR.y/(float)hSR);
//		float SR_W_f = ((float)SR_W)/scalingIntToFloatFactor;
//		float weight_f = ((float)weight)/scalingIntToFloatFactor;
//		float resultColor =  ( SR_W_f * srColor[idxSRColor] + weight_f * lrColor) / (SR_W_f + weight_f);
//		srColor[idxSRColor] = resultColor;
//		c = 1;
//		idxSRColor = idxSR + wSR*hSR*c;
//		lrColor = tex2D(texRefColorG, xyLR.x/(float)wSR, xyLR.y/(float)hSR);
//		resultColor =  ( SR_W_f * srColor[idxSRColor] + weight_f * lrColor) / (SR_W_f + weight_f);
//		srColor[idxSRColor] = resultColor;
//		c = 2;
//		idxSRColor = idxSR + wSR*hSR*c;
//		lrColor = tex2D(texRefColorB, xyLR.x/(float)wSR, xyLR.y/(float)hSR);
//		resultColor = ( SR_W_f * srColor[idxSRColor] + weight_f * lrColor) / (SR_W_f + weight_f);
//		srColor[idxSRColor] = resultColor;
//
//		srWeights[idxSR] += weight;
//	}
}



__global__ void fuseFrame(float *srDepth, float *srWeights, float* srColor, float* lrDepth, float* lrColor, float BlurrFactor, size_t wSR, size_t hSR, size_t s){

	size_t wLR = wSR / s;
	size_t hLR = hSR / s;

	size_t xSR = threadIdx.x + blockDim.x * blockIdx.x;
	size_t ySR = threadIdx.y + blockDim.y * blockIdx.y;

	if(xSR >= wSR || ySR >= hSR) return;

	// transform the optical center of key frame to coord. system of new frame

	float3 pCameraKF = transform2Dto3D(s*optcenter_x, s*optcenter_y, 0.0f, s);

	float3 pCameraKF_LR = applyTransformationMatricesSRtoLR(pCameraKF);

	float2 xyCameraKF = transform3Dto2D(pCameraKF_LR.x, pCameraKF_LR.y, pCameraKF_LR.z, s);

	size_t idxSR = xSR + wSR * ySR;
	float srDepthKF = srDepth[idxSR];
	float srWeight = srWeights[idxSR];

	if (srWeight == 0.0f){
		// means that depth is also 0.0f, take some "random" depth to compute the ray
		srDepthKF = 1000.0f;
	}

	// transform pixel in key frame to coord. system of new frame

	float3 pSR = transform2Dto3D(xSR + 0.5f, ySR + 0.5f, srDepthKF, s);

	float3 pLR = applyTransformationMatricesSRtoLR(pSR);

	float2 xy_pLR = transform3Dto2D(pLR.x, pLR.y, pLR.z, s);

	// linear function for the ray in new frame - y = mx + n
	float m = (xyCameraKF.y - xy_pLR.y) / (xyCameraKF.x - xy_pLR.x);
	float n = xyCameraKF.y - m * xyCameraKF.x;

	if (fabs(m) <= 1.0f && ((xyCameraKF.x - xy_pLR.x) != 0.0f)){
		// iterate over x

		// compute start and end point of iteration by computing intersection points of ray and extended image edges
		float x_min = (float)wSR;
		float x_max = 0.0f;

		float y_at_xeq0 = m*0.0f + n;
		float y_at_xeqW = m*(float)wSR + n;
		float x_at_yeq0 = -n/m;
		float x_at_yeqH = ((float)hSR - n)/m;

		if (0.0f <= y_at_xeq0 && y_at_xeq0 <= (float)hSR){
			x_min = 0.0f;
		}
		if (0.0f <= y_at_xeqW && y_at_xeqW <= (float)hSR){
			x_max = (float)wSR;
		}
		if (0.0f <= x_at_yeq0 && x_at_yeq0 <= (float)wSR){
			if (x_at_yeq0 < x_min){
				x_min = x_at_yeq0;
			}
			if (x_at_yeq0 > x_max){
				x_max = x_at_yeq0;
			}
		}
		if (0.0f <= x_at_yeqH && x_at_yeqH <= (float)wSR){
			if (x_at_yeqH < x_min){
				x_min = x_at_yeqH;
			}
			if (x_at_yeqH > x_max){
				x_max = x_at_yeqH;
			}
		}

		int x_min_idx = (int)(x_min);
		int x_max_idx = (int)(x_max);

		for(int xIdx = x_min_idx; xIdx < x_max_idx; xIdx++){
			// check if pixel in frame matches and update accordingly
			int yIdx = (int)(m * (float)xIdx + n);
			if (yIdx >= hSR || yIdx < 0) {
				continue;
			}
			float depth = lrDepth[xIdx/s + yIdx/s * wLR];
			if (depth == 0.0f){
				// skip invalid depth values
				continue;
			}
			float3 pFrame = transform2Dto3D(xIdx + 0.5f, yIdx + 0.5f, depth, s);
			float3 pKeyFrame = applyTransformationMatricesLRtoSR(pFrame);
			float2 xy_pKeyFrame = transform3Dto2D(pKeyFrame.x, pKeyFrame.y, pKeyFrame.z, s);
			float depthInKeyFrame = pKeyFrame.z;
			// if x and y position match...
			if (fabs(xy_pKeyFrame.x - ((float)xSR + 0.5f)) <= 0.5f && fabs(xy_pKeyFrame.y - ((float)ySR + 0.5f)) <= 0.5f){
				// if valid depth in key-frame and depth values of new frame "close" ...
				if (srWeight != 0.0f && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < epsilonAveraging){
					// averaging
					bool averageColor = true;
					float sumDiff = 0.0f;
					for (size_t c = 0; c < 3; ++c){
						float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
						float colorKF = srColor[idxSR + wSR * hSR *c];
						float diff = fabs(colorKF - color);
						sumDiff += diff;
						if (diff > maxColorDiffChannel){
							// skip this pixel if color difference too large for some channel
							averageColor = false;
							break;
						}
					}
					if (sumDiff > maxColorDiffSum){
						// skip this pixel if sum of differences in all channels too large
						averageColor = false;
					}
					if (averageColor){
						float weight = calcWeight(depth, focal_length, baseline);
						if (weight == 0.0f){
							continue;
						}
						srDepth[idxSR] = (srWeight * srDepthKF + weight * depthInKeyFrame) / (srWeight + weight);
						for(size_t c = 0; c < 3; ++c){
							float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
							float srColorTmp = srColor[idxSR + wSR * hSR * c];
							srColor[idxSR + wSR * hSR * c] = (srWeight * srColorTmp + BlurrFactor * weight * color) / (srWeight + BlurrFactor * weight);
						}
						srWeights[idxSR] += weight;
					}
				}
				// else if depth of new frame less than depth in key-frame (up to some threshold) or invalid depth value in key-frame ...
				else if ((depthInKeyFrame < srDepthKF && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < epsilonOverwriting) || srWeight == 0.0f){
					// overwriting (for depth), colors are always averaged
					bool overwriteColor = true;
					float sumDiff = 0.0f;
					for (size_t c = 0; c < 3; ++c){
						float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
						float colorKF = srColor[idxSR + wSR * hSR *c];
						float diff = fabs(colorKF - color);
						sumDiff += diff;
						if (diff > maxColorDiffChannel){
							// skip this pixel if color difference too large for some channel
							overwriteColor = false;
							break;
						}
					}
					if (sumDiff > maxColorDiffSum){
						// skip this pixel if sum of differences in all channels too large
						overwriteColor = false;
					}
					if (overwriteColor){
						// overwrite depth and average colors
						srDepth[idxSR] = depthInKeyFrame;
						float weight = calcWeight(depth, focal_length, baseline);
						if (weight == 0.0f){
							continue;
						}
						for (size_t c = 0; c < 3; ++c){
							float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
							float srColorTmp = srColor[idxSR + wSR * hSR * c];
							srColor[idxSR + wSR * hSR * c] = (weight * srColorTmp + BlurrFactor * weight * color) / (weight + BlurrFactor * weight);
						}
						srWeights[idxSR] = weight;
					}
					else if (srWeight == 0.0f){
						// colors differ too much, but no valid depth value in key-frame -> don"t overwrite colors but take depth as "approximation"
						srDepth[idxSR] = depthInKeyFrame;
						float weight = calcWeight(depth, focal_length, baseline);
						srWeights[idxSR] = weight;
					}
				}
				//else skip
			}
		}

	}
	else{
		// iterate over y - recompute m, n s.t. x = my + n
		m = (xyCameraKF.x - xy_pLR.x) / (xyCameraKF.y - xy_pLR.y);
		n = xyCameraKF.x - m * xyCameraKF.y;

		// same as for x with x and y exchanged
		float y_min = (float)hSR;
		float y_max = 0.0f;

		float x_at_yeq0 = m*0.0f + n;
		float x_at_yeqH = m*(float)hSR + n;
		float y_at_xeq0 = -n/m;
		float y_at_xeqW = ((float)wSR - n)/m;

		if (0.0f <= x_at_yeq0 && x_at_yeq0 <= (float)wSR){
			y_min = 0.0f;
		}
		if (0.0f <= x_at_yeqH && x_at_yeqH <= (float)wSR){
			y_max = (float)hSR;
		}
		if (0.0f <= y_at_xeq0 && y_at_xeq0 <= (float)hSR){
			if (y_at_xeq0 < y_min){
				y_min = y_at_xeq0;
			}
			if (y_at_xeq0 > y_max){
				y_max = y_at_xeq0;
			}
		}
		if (0.0f <= y_at_xeqW && y_at_xeqW <= (float)hSR){
			if (y_at_xeqW < y_min){
				y_min = y_at_xeqW;
			}
			if (y_at_xeqW > y_max){
				y_max = y_at_xeqW;
			}
		}

		int y_min_idx = (int)(y_min);
		int y_max_idx = (int)(y_max);

		for(int yIdx = y_min_idx; yIdx < y_max_idx; yIdx++){
			// check if pixel in frame matches and update accordingly

			int xIdx = (int)(m * (float)yIdx + n);

			if (xIdx >= wSR || xIdx < 0) {
				continue;
			}
			float depth = lrDepth[(xIdx/s) + (yIdx/s) * wLR];
			if (depth == 0.0f){
				// skip invalid depth values
				continue;
			}
			float3 pFrame = transform2Dto3D(xIdx + 0.5f, yIdx + 0.5f, depth, s);
			float3 pKeyFrame = applyTransformationMatricesLRtoSR(pFrame);
			float2 xy_pKeyFrame = transform3Dto2D(pKeyFrame.x, pKeyFrame.y, pKeyFrame.z, s);
			float depthInKeyFrame = pKeyFrame.z;
			if (fabs(xy_pKeyFrame.x - ((float)xSR + 0.5f)) <= 0.5f && fabs(xy_pKeyFrame.y - ((float)ySR + 0.5f)) <= 0.5f){

				if (srWeight != 0.0f && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < epsilonAveraging && !( srWeight == 0.0f)){
					// averaging
					bool averageColor = true;
					float sumDiff = 0.0f;
					for (size_t c = 0; c < 3; ++c){
						float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
						float colorKF = srColor[idxSR + wSR * hSR *c];
						float diff = fabs(colorKF - color);
						sumDiff += diff;
						if (diff > maxColorDiffChannel){
							averageColor = false;
							break;
						}
					}
					if (sumDiff > maxColorDiffSum){
						averageColor = false;
					}
					if (averageColor){
						float weight = calcWeight(depth, focal_length, baseline);
						if (weight == 0.0f){
							continue;
						}
						srDepth[idxSR] = (srWeight * srDepthKF + weight * depthInKeyFrame) / (srWeight + weight);
						for(size_t c = 0; c < 3; ++c){
							float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
							float srColorTmp = srColor[idxSR + wSR * hSR * c];
							srColor[idxSR + wSR * hSR * c] = (srWeight * srColorTmp + BlurrFactor * weight * color) / (srWeight + BlurrFactor * weight);
						}
						srWeights[idxSR] += weight;
					}
				}
				else if ((depthInKeyFrame < srDepthKF && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < epsilonOverwriting) || srWeight == 0.0f){
					// overwriting

					bool overwriteColor = true;
					float sumDiff = 0.0f;
					for (size_t c = 0; c < 3; ++c){
						float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
						float colorKF = srColor[idxSR + wSR * hSR *c];
						float diff = fabs(colorKF - color);
						sumDiff += diff;
						if (diff > maxColorDiffChannel){
							overwriteColor = false;
							break;
						}
					}
					if (sumDiff > maxColorDiffSum){
						overwriteColor = false;
					}
					if (overwriteColor){
						srDepth[idxSR] = depthInKeyFrame;
						float weight = calcWeight(depth, focal_length, baseline);
						if (weight == 0.0f){
							continue;
						}
						for (size_t c = 0; c < 3; ++c){
							float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
							float srColorTmp = srColor[idxSR + wSR * hSR * c];
							srColor[idxSR + wSR * hSR * c] = (weight * srColorTmp + BlurrFactor * weight * color) / (weight + BlurrFactor * weight);
						}
						srWeights[idxSR] = weight;
					}
					else if (srWeight == 0.0f){
						srDepth[idxSR] = depthInKeyFrame;
						float weight = calcWeight(depth, focal_length, baseline);
						srWeights[idxSR] = weight;
					}

				}
				//else skip
			}
		}
	}
}



__global__ void fuseFrameMedian(float *srDepth, float *srWeights, float* srColor, float* lrDepth, float* lrColor, float* BlurrFactors, size_t numFrames, size_t wSR, size_t hSR, size_t s){

	size_t wLR = wSR / s;
	size_t hLR = hSR / s;

	size_t xSR = threadIdx.x + blockDim.x * blockIdx.x;
	size_t ySR = threadIdx.y + blockDim.y * blockIdx.y;

	if(xSR >= wSR || ySR >= hSR) return;

	size_t idxSR = xSR + wSR * ySR;

	// update of each frame (weighted average of possibly multiple pixels) for colors and weights are stored here
	float updateColors[(MAX_FRAMES + 1) * 3];
	float updateWeights[MAX_FRAMES + 1];

	// store values of key frame
	updateColors[0] = srColor[idxSR];
	updateColors[1] = srColor[idxSR + wSR * hSR];
	updateColors[2] = srColor[idxSR + 2 * wSR * hSR];
	updateWeights[0] = srWeights[idxSR];

	// initialize for the frames
	for (int j = 1; j < MAX_FRAMES + 1; ++j){
		updateWeights[j] = 0.0f;
		for (int c = 0; c < 3; ++c){
			updateColors[j*3 + c] = 0.0f;
		}
	}

	//iterate over the frames
	for (size_t frame = 1; frame <= numFrames; ++frame, lrColor += wLR*hLR*3, lrDepth += wLR*hLR){
		// the updates for one frame are computed like in the weighted average version
		// especially the depth image is the same

		float srDepthKF = srDepth[idxSR];
		float srWeight = srWeights[idxSR];
		if (srWeight == 0.0f){
			// means that depth is invalid, take some "random" depth to compute the ray
			srDepthKF = 1000.0f;
		}

		float3 pCameraKF = transform2Dto3D(s*optcenter_x, s*optcenter_y, 0.0f, s);

		float3 pCameraKF_LR = applyTransformationMatricesSRtoLR(pCameraKF, frame - 1);

		float2 xyCameraKF = transform3Dto2D(pCameraKF_LR.x, pCameraKF_LR.y, pCameraKF_LR.z, s);

		float3 pSR = transform2Dto3D(xSR + 0.5f, ySR + 0.5f, srDepthKF, s);

		float3 pLR = applyTransformationMatricesSRtoLR(pSR, frame - 1);

		float2 xy_pLR = transform3Dto2D(pLR.x, pLR.y, pLR.z, s);

		// linear function for the ray in new frame - y = mx + n
		float m = (xyCameraKF.y - xy_pLR.y) / (xyCameraKF.x - xy_pLR.x);
		float n = xyCameraKF.y - m * xyCameraKF.x;

		size_t numUpdatesFrame = 0;

		if (fabs(m) <= 1.0f && ((xyCameraKF.x - xy_pLR.x) != 0.0f)){
			// iterate over x

			float x_min = (float)wSR;
			float x_max = 0.0f;

			float y_at_xeq0 = m*0.0f + n;
			float y_at_xeqW = m*(float)wSR + n;
			float x_at_yeq0 = -n/m;
			float x_at_yeqH = ((float)hSR - n)/m;

			if (0.0f <= y_at_xeq0 && y_at_xeq0 <= (float)hSR){
				x_min = 0.0f;
			}
			if (0.0f <= y_at_xeqW && y_at_xeqW <= (float)hSR){
				x_max = (float)wSR;
			}
			if (0.0f <= x_at_yeq0 && x_at_yeq0 <= (float)wSR){
				if (x_at_yeq0 < x_min){
					x_min = x_at_yeq0;
				}
				if (x_at_yeq0 > x_max){
					x_max = x_at_yeq0;
				}
			}
			if (0.0f <= x_at_yeqH && x_at_yeqH <= (float)wSR){
				if (x_at_yeqH < x_min){
					x_min = x_at_yeqH;
				}
				if (x_at_yeqH > x_max){
					x_max = x_at_yeqH;
				}
			}

			int x_min_idx = (int)(x_min);
			int x_max_idx = (int)(x_max);

			for(int xIdx = x_min_idx; xIdx < x_max_idx; xIdx++){
				// check if pixel in frame is more near than epsilon (-> overwrite) or in range (-> averaging) of pixel in Keyframe
				int yIdx = (int)(m * (float)xIdx + n);
				if (yIdx >= hSR || yIdx < 0) {
					continue;
				}
				float depth = lrDepth[xIdx/s + yIdx/s * wLR];
				float3 pFrame = transform2Dto3D(xIdx + 0.5f, yIdx + 0.5f, depth, s);
				float3 pKeyFrame = applyTransformationMatricesLRtoSR(pFrame, frame - 1);
				float2 xy_pKeyFrame = transform3Dto2D(pKeyFrame.x, pKeyFrame.y, pKeyFrame.z, s);
				float depthInKeyFrame = pKeyFrame.z;
				if (fabs(xy_pKeyFrame.x - ((float)xSR + 0.5f)) <= 0.5f && fabs(xy_pKeyFrame.y - ((float)ySR + 0.5f)) <= 0.5f){
					if (srWeight != 0.0f && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < epsilonAveragingMedian){
						// averaging

						bool averageColor = true;
						float sumDiff = 0.0f;
						for (size_t c = 0; c < 3; ++c){
							float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
							float colorKF = srColor[idxSR + wSR * hSR *c];
							float diff = fabs(colorKF - color);
							sumDiff += diff;
							if (diff > maxColorDiffChannelMedian){
								averageColor = false;
								break;
							}
						}
						if (sumDiff > maxColorDiffSumMedian){
							averageColor = false;
						}
						if (averageColor){
							float weight = calcWeight(depth, focal_length, baseline);
							if (weight == 0.0f){
								continue;
							}
							srDepth[idxSR] = (srWeight * srDepthKF + weight * depthInKeyFrame) / (srWeight + weight);
							float currentWeight = updateWeights[frame];
							for(size_t c = 0; c < 3; ++c){
								float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
								updateColors[frame * 3 + c] = (currentWeight * updateColors[frame * 3 + c] + BlurrFactors[frame] * weight * color) / (currentWeight + BlurrFactors[frame] * weight);
							}
							updateWeights[frame] += BlurrFactors[frame] * weight;
							numUpdatesFrame++;
							srWeights[idxSR] += weight;
						}
					}
					else if ((depthInKeyFrame < srDepthKF && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < epsilonOverwritingMedian) || srWeight == 0.0f){
						// overwriting

						bool overwriteColor = true;
						float sumDiff = 0.0f;
						for (size_t c = 0; c < 3; ++c){
							float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
							float colorKF = srColor[idxSR + wSR * hSR *c];
							float diff = fabs(colorKF - color);
							sumDiff += diff;
							if (diff > maxColorDiffChannelMedian){
								overwriteColor = false;
								break;
							}
						}
						if (sumDiff > maxColorDiffSumMedian){
							overwriteColor = false;
						}
						if (overwriteColor){
							srDepth[idxSR] = depthInKeyFrame;
							float weight = calcWeight(depth, focal_length, baseline);
							if (weight == 0.0f){
								continue;
							}
							float currentWeight = updateWeights[frame];
							for (size_t c = 0; c < 3; ++c){
								float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
								updateColors[frame * 3 + c] = (currentWeight * updateColors[frame * 3 + c] + BlurrFactors[frame] * weight * color) / (currentWeight + BlurrFactors[frame] * weight);
							}
							updateWeights[frame] += BlurrFactors[frame] * weight;
							numUpdatesFrame++;
							srWeights[idxSR] = weight;
						}
						else if (srWeight == 0.0f){
							srDepth[idxSR] = depthInKeyFrame;
							float weight = calcWeight(depth, focal_length, baseline);
							srWeights[idxSR] = weight;
						}
					}
					//else skip
				}
			}
		}
		else{
			// iterate over y - recompute m, n s.t. x = my + n
			m = (xyCameraKF.x - xy_pLR.x) / (xyCameraKF.y - xy_pLR.y);
			n = xyCameraKF.x - m * xyCameraKF.y;

			// same as for x with x and y exchanged
			float y_min = (float)hSR;
			float y_max = 0.0f;

			float x_at_yeq0 = m*0.0f + n;
			float x_at_yeqH = m*(float)hSR + n;
			float y_at_xeq0 = -n/m;
			float y_at_xeqW = ((float)wSR - n)/m;

			if (0.0f <= x_at_yeq0 && x_at_yeq0 <= (float)wSR){
				y_min = 0.0f;
			}
			if (0.0f <= x_at_yeqH && x_at_yeqH <= (float)wSR){
				y_max = (float)hSR;
			}
			if (0.0f <= y_at_xeq0 && y_at_xeq0 <= (float)hSR){
				if (y_at_xeq0 < y_min){
					y_min = y_at_xeq0;
				}
				if (y_at_xeq0 > y_max){
					y_max = y_at_xeq0;
				}
			}
			if (0.0f <= y_at_xeqW && y_at_xeqW <= (float)hSR){
				if (y_at_xeqW < y_min){
					y_min = y_at_xeqW;
				}
				if (y_at_xeqW > y_max){
					y_max = y_at_xeqW;
				}
			}

			int y_min_idx = (int)(y_min);
			int y_max_idx = (int)(y_max);

			for(int yIdx = y_min_idx; yIdx < y_max_idx; yIdx++){
				// check if pixel in frame is more near than epsilon (-> overwrite) or in range (-> averaging) of pixel in Keyframe
				int xIdx = (int)(m * (float)yIdx + n);
				if (xIdx >= wSR || xIdx < 0) {
					continue;
				}
				float depth = lrDepth[(xIdx/s) + (yIdx/s) * wLR];
				float3 pFrame = transform2Dto3D(xIdx + 0.5f, yIdx + 0.5f, depth, s);
				float3 pKeyFrame = applyTransformationMatricesLRtoSR(pFrame, frame - 1);
				float2 xy_pKeyFrame = transform3Dto2D(pKeyFrame.x, pKeyFrame.y, pKeyFrame.z, s);
				float depthInKeyFrame = pKeyFrame.z;
				if (fabs(xy_pKeyFrame.x - ((float)xSR + 0.5f)) <= 0.5f && fabs(xy_pKeyFrame.y - ((float)ySR + 0.5f)) <= 0.5f){
					if (srWeight != 0.0f && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < epsilonAveragingMedian && !( srWeight == 0.0f)){
						// averaging
						bool averageColor = true;
						float sumDiff = 0.0f;
						for (size_t c = 0; c < 3; ++c){
							float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
							float colorKF = srColor[idxSR + wSR * hSR *c];
							float diff = fabs(colorKF - color);
							sumDiff += diff;
							if (diff > maxColorDiffChannelMedian){
								averageColor = false;
								break;
							}
						}
						if (sumDiff > maxColorDiffSumMedian){
							averageColor = false;
						}
						if (averageColor){
							float weight = calcWeight(depth, focal_length, baseline);
							if (weight == 0.0f){
								continue;
							}
							srDepth[idxSR] = (srWeight * srDepthKF + weight * depthInKeyFrame) / (srWeight + weight);
							float currentWeight = updateWeights[frame];
							for(size_t c = 0; c < 3; ++c){
								float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
								updateColors[frame * 3 + c] = (currentWeight * updateColors[frame * 3 + c] + BlurrFactors[frame] * weight * color) / (currentWeight + BlurrFactors[frame] * weight);
							}
							updateWeights[frame] += BlurrFactors[frame] * weight;
							numUpdatesFrame++;
							srWeights[idxSR] += weight;
						}


					}
					else if ((depthInKeyFrame < srDepthKF && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < epsilonOverwritingMedian) || srWeight == 0.0f){
						// overwriting

						bool overwriteColor = true;
						float sumDiff = 0.0f;
						for (size_t c = 0; c < 3; ++c){
							float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
							float colorKF = srColor[idxSR + wSR * hSR *c];
							float diff = fabs(colorKF - color);
							sumDiff += diff;
							if (diff > maxColorDiffChannelMedian){
								overwriteColor = false;
								break;
							}
						}
						if (sumDiff > maxColorDiffSumMedian){
							overwriteColor = false;
						}
						if (overwriteColor){
							srDepth[idxSR] = depthInKeyFrame;
							float weight = calcWeight(depth, focal_length, baseline);
							if (weight == 0.0f){
								continue;
							}
							float currentWeight = updateWeights[frame];
							for (size_t c = 0; c < 3; ++c){
								float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
								updateColors[frame * 3 + c] = (currentWeight * updateColors[frame * 3 + c] + BlurrFactors[frame] * weight * color) / (currentWeight + BlurrFactors[frame] * weight);
							}
							updateWeights[frame] += BlurrFactors[frame] * weight;
							numUpdatesFrame++;
							srWeights[idxSR] = weight;
						}
						else if (srWeight == 0.0f){
							srDepth[idxSR] = depthInKeyFrame;
							float weight = calcWeight(depth, focal_length, baseline);
							srWeights[idxSR] = weight;
						}
					}
					//else skip
				}
			}
		}

		if (numUpdatesFrame > 0){
			updateWeights[frame] /= (float)numUpdatesFrame;
		}
	}

	// compute median
	for (size_t c = 0; c < 3; ++c){
		//initialize for keyframe
		size_t currentOptimum = 0;
		float currentOptimumValue = 0.0f;

		float srColorChannelc = updateColors[c];
		for (size_t frame = 1; frame <= numFrames; frame++){
			currentOptimumValue += updateWeights[frame] * fabs(srColorChannelc - updateColors[frame*3 + c]);
		}

		// compute objective function value for each frame
		for (size_t frame = 1; frame <= numFrames; ++frame){
			float fctValue = 0.0f;
			if (updateWeights[frame] == 0.0f) {
				continue;
			}
			float frameColorChannelc = updateColors[frame*3 + c];
			// this includes the key frame
			for (size_t frameCmp = 0; frameCmp <= numFrames; ++frameCmp){
				fctValue += updateWeights[frameCmp] * fabs(frameColorChannelc - updateColors[frameCmp*3 + c]);
			}
			if (fctValue < currentOptimumValue){
				currentOptimumValue = fctValue;
				currentOptimum = frame;
			}
		}
		// write back median
		srColor[idxSR + c * wSR * hSR] = updateColors[currentOptimum*3 + c];
	}
}
