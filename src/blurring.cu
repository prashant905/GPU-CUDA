#include "blurring.cuh"

void convolution(float* d_img, float sigma, size_t w, size_t h, size_t nc){
    dim3 block = dim3(32,32,1);
	dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (nc + block.z - 1) / block.z);


    size_t r = ceilf(3*sigma);
    size_t kernelSize = 2*r + 1;
    size_t numBytesSharedMem = (2*r+block.x)*(2*r+block.y)*nc * sizeof(float);

    float sum = 0.0f;
    float * kernel = new float[kernelSize*kernelSize];
    float * kernelCopy = new float[kernelSize*kernelSize];
    for(size_t i = 0; i < kernelSize; ++i){
    	for(size_t j = 0; j < kernelSize; ++j){
    		float a = (float)i - (float)r;
    		float b = (float)j - (float)r;
    		kernel[i + j*kernelSize] = 1.0f/(2*M_PI*sigma*sigma) * powf(M_E, -(a*a+b*b)/(2*sigma*sigma));
    		sum += kernel[i + j*kernelSize];
    	}
    }
    for(size_t i = 0; i < kernelSize; ++i){
    	for(size_t j = 0; j < kernelSize; ++j){
    		kernel[i + j*kernelSize] /= sum;
    	}
    }

    if (r <= 20){
    	cudaMemcpyToSymbol (d_const_kernel, kernel, kernelSize * kernelSize * sizeof(float));
        convolutionGPU_shared_const_Memory <<<grid, block, numBytesSharedMem>>>(d_img, d_img, w, h, nc, kernelSize, kernelSize, r);
        CUDA_CHECK;
    }
    else {
        float * d_kernel;
        cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
        CUDA_CHECK;
        cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK;
		size_t numBytesSharedMem = (2*r+block.x)*(2*r+block.y)*nc * sizeof(float);
		convolutionGPU_sharedMemory <<<grid, block, numBytesSharedMem>>>(d_kernel, d_img, d_img, w, h, nc, kernelSize, kernelSize, r);
		CUDA_CHECK;
    }
}




__global__ void convolutionGPU_shared_const_Memory(float* imgIn, float* imgOut, size_t w, size_t h, size_t nc, size_t wk, size_t hk, size_t r){
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;
	size_t c = threadIdx.z + blockDim.z * blockIdx.z;

	size_t idx = x + w*y + w*h*c;

	size_t num_x = blockDim.x + 2*r;
	size_t num_y = blockDim.y + 2*r;
	// indices for left upper corner for data needed from imgIn
	int x_l = blockDim.x * blockIdx.x - r;
	int y_u = blockDim.y * blockIdx.y - r;
	// indices for right down corner
	int x_r = x_l + num_x - 1;//blockDim.x * (blockIdx.x + 1) + r - 1;
	int y_d = y_u + num_y - 1;//blockDim.y * (blockIdx.y + 1) + r - 1;

	extern __shared__ float sh_data[];


	int x_copy = x_l + (int)threadIdx.x;
	int y_copy = y_u + (int)threadIdx.y;

	size_t x_steps = (num_x + blockDim.x - 1) / blockDim.x;
	size_t y_steps = (num_y + blockDim.y - 1) / blockDim.y;

	for (size_t ys = 0; ys < y_steps; ++ys, y_copy += blockDim.y){
		if (y_copy > y_d){
			// all copy operations done for this thread
			break;
		}
		x_copy = x_l + (int)threadIdx.x;
		for (size_t xs = 0; xs < x_steps; ++xs, x_copy += blockDim.x){
			if (x_copy > x_r){
				// x out of range
				break;
			}
			size_t idx_sh_data = (size_t)threadIdx.x + xs * (size_t)blockDim.x + ((size_t)threadIdx.y + ys*(size_t)blockDim.y) * num_x + c* num_x * num_y;
			size_t idx_global_x = min(max(x_copy, 0), (int)(w-1));
			size_t idx_global_y = min(max(y_copy, 0), (int)(h-1));
			size_t idx_global = idx_global_x + w * idx_global_y	+ w*h*c;
			sh_data[idx_sh_data] = imgIn[idx_global];
		}
	}

	__syncthreads();

	if (x >= w || y >= h || c >= nc){
		return;
	}
	float result = 0.0f;
	for (size_t xk = 0; xk < wk; xk++){
		for (size_t yk = 0; yk < hk; yk++){
			float imgValue = sh_data[threadIdx.x + xk + (threadIdx.y + yk) * num_x + num_x * num_y * c];

			size_t x_idx = min(max((int)x - (int)r + (int)xk, 0), (int)(w-1));
			size_t y_idx = min(max((int)y - (int)r + (int)yk, 0), (int)(h-1));
			result += d_const_kernel[xk + wk*yk] * imgValue;
		}
	}
	imgOut[idx] = result;
}

__global__ void convolutionGPU_sharedMemory(float* kernel, float* imgIn, float* imgOut, size_t w, size_t h, size_t nc, size_t wk, size_t hk, size_t r){
	size_t x = threadIdx.x + blockDim.x * blockIdx.x;
	size_t y = threadIdx.y + blockDim.y * blockIdx.y;
	size_t c = threadIdx.z + blockDim.z * blockIdx.z;

	size_t idx = x + w*y + w*h*c;

	size_t num_x = blockDim.x + 2*r;
	size_t num_y = blockDim.y + 2*r;
	// indices for left upper corner for data needed from imgIn
	int x_l = blockDim.x * blockIdx.x - r;
	int y_u = blockDim.y * blockIdx.y - r;
	// indices for right down corner
	int x_r = x_l + num_x - 1;//blockDim.x * (blockIdx.x + 1) + r - 1;
	int y_d = y_u + num_y - 1;//blockDim.y * (blockIdx.y + 1) + r - 1;

	extern __shared__ float sh_data[];


	int x_copy = x_l + (int)threadIdx.x;
	int y_copy = y_u + (int)threadIdx.y;

	size_t x_steps = (num_x + blockDim.x - 1) / blockDim.x;
	size_t y_steps = (num_y + blockDim.y - 1) / blockDim.y;

	for (size_t ys = 0; ys < y_steps; ++ys, y_copy += blockDim.y){
		if (y_copy > y_d){
			// all copy operations done for this thread
			break;
		}
		x_copy = x_l + (int)threadIdx.x;
		for (size_t xs = 0; xs < x_steps; ++xs, x_copy += blockDim.x){
			if (x_copy > x_r){
				// x out of range
				break;
			}
			size_t idx_sh_data = (size_t)threadIdx.x + xs * (size_t)blockDim.x + ((size_t)threadIdx.y + ys*(size_t)blockDim.y) * num_x + c* num_x * num_y;
			size_t idx_global_x = min(max(x_copy, 0), (int)(w-1));
			size_t idx_global_y = min(max(y_copy, 0), (int)(h-1));
			size_t idx_global = idx_global_x + w * idx_global_y	+ w*h*c;
			sh_data[idx_sh_data] = imgIn[idx_global];
		}
	}

	__syncthreads();

	if (x >= w || y >= h || c >= nc){
		return;
	}
	float result = 0.0f;
	for (size_t xk = 0; xk < wk; xk++){
		for (size_t yk = 0; yk < hk; yk++){
			float imgValue = sh_data[threadIdx.x + xk + (threadIdx.y + yk) * num_x + num_x * num_y * c];

			size_t x_idx = min(max((int)x - (int)r + (int)xk, 0), (int)w);
			size_t y_idx = min(max((int)y - (int)r + (int)yk, 0), (int)h);
			result += kernel[xk + wk*yk] * imgValue;
		}
	}
	imgOut[idx] = result;
}
