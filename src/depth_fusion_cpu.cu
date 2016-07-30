#include "depth_fusion_cpu.cuh"
#include "global.cuh"
#include "aux.cuh"
#include <cstdio>
#include <iostream>
#include <climits>

namespace cpu{

Eigen::Matrix4f pose_SR;
Eigen::Matrix4f pose_inv_SR;
Eigen::Matrix4f pose_LR;
Eigen::Matrix4f pose_inv_LR;

Eigen::Matrix4f *poses_inv_LR = NULL;
Eigen::Matrix4f *poses_LR = NULL;

float focal_x = 517.3f;
float focal_y = 516.5f;
float optcenter_x = 318.6f;
float optcenter_y = 255.3f;

float focal_length = 1.0f;
float baseline = 1.0f;

float h_maxColorDiffChannelMedian = MAXCOLORDIFFCHANNELMEDIAN;
float h_maxColorDiffSumMedian = MAXCOLORDIFFSUMMEDIAN;
float h_epsilonAveragingMedian = EPSILONAVERAGINGMEDIAN;
float h_epsilonOverwritingMedian = EPSILONOVERWRITINGMEDIAN;

float h_maxColorDiffChannel = MAXCOLORDIFFCHANNEL;
float h_maxColorDiffSum = MAXCOLORDIFFSUM;
float h_epsilonAveraging = EPSILONAVERAGING;
float h_epsilonOverwriting = EPSILONOVERWRITING;

void init_T_SR(Eigen::Matrix4f rot){
	pose_SR = rot;
}

void init_T_inv_SR(Eigen::Matrix4f rot){
	pose_inv_SR = rot;
}

void init_T_inv_LR(Eigen::Matrix4f rot){
	pose_inv_LR = rot;
}

void init_T_inv_LR(Eigen::Matrix4f* rot, size_t numFrames){
	if (poses_inv_LR == NULL){
		poses_inv_LR = new Eigen::Matrix4f[MAX_FRAMES];
	}

	if (numFrames > MAX_FRAMES){
		std::cout << "Too many Frames (at most 20)!" << std::endl;
	}
	for (int i = 0; i < numFrames; ++i ){
		poses_inv_LR[i] = rot[i];
	}
}

void init_T_LR(Eigen::Matrix4f rot){
	pose_LR = rot;
}

void init_T_LR(Eigen::Matrix4f* rot, size_t numFrames){
	if (poses_LR == NULL){
		poses_LR = new Eigen::Matrix4f[MAX_FRAMES];
	}

	if (numFrames > MAX_FRAMES){
		std::cout << "Too many Frames (at most 20)!" << std::endl;
	}
	for (int i = 0; i < numFrames; ++i ){
		poses_LR[i] = rot[i];
	}
}

void initSRDepthAndWeights(float* d_depth_sr, float* d_weights_sr, size_t wSR, size_t hSR){

	calcInitialWeights(d_depth_sr, d_weights_sr, wSR, hSR);
}



void fuseDepthAndWeights(float* d_depth_sr, float* d_weights_sr, float* d_depth_lr, float* d_color_sr, float* d_color_lr, float BlurrFactor, size_t s, size_t wSR, size_t hSR, bool useRay){
	size_t wLR = wSR/s;
    size_t hLR = hSR/s;

	float* update_depth_sr = new float[wSR * hSR];
	float* update_weights_sr = new float[wSR * hSR];
//	float* update_depth_sr = new float[wSR * hSR];
//	float* update_weights_sr = new float[wSR * hSR];
	
	initAll(update_depth_sr, 0.0f, wSR, hSR, 1);
	initAll(update_weights_sr, 0.0f, wSR, hSR, 1);

	// overwrite 0 depth values by INT_MAX
//	preprocessDepth(d_depth_lr, wLR, hLR);


////	 To see the updates for depth-image
//	initAll(d_depth_sr, 0.0f, wSR, hSR, 1);
//	initAll(d_weights_sr, 0.0f, wSR, hSR, 1);
////	 To see the updates for color-image
//	initAll(d_color_sr, 0.0f, wSR, hSR, 3);

	if (useRay){
		fuseFrame(d_depth_sr, d_weights_sr, d_color_sr, d_depth_lr, d_color_lr, BlurrFactor, wSR, hSR, s);
	}
	else{
		updateStep(d_depth_lr, update_depth_sr, update_weights_sr, wLR, hLR, wSR, hSR);

		fuseUpdates(d_depth_sr, update_depth_sr, d_weights_sr, update_weights_sr, d_color_sr, d_color_lr, BlurrFactor, wSR, hSR, s);
	}

//	postprocessDepth <<<grid, block>>> (d_depth_sr, wSR, hSR);
//	postprocessDepth(d_depth_sr, wSR, hSR);

}

void fuseDepthAndWeightsWithRayMedian(float* d_depth_sr, float* d_weights_sr, float* d_depth_lr, float* d_color_sr, float* d_color_lr, float* BlurrFactors, size_t numFrames, size_t s, size_t wSR, size_t hSR){
	float* update_depth_sr = new float[wSR * hSR];
	float* update_weights_sr = new float[wSR * hSR];

	initAll(update_depth_sr, 0.0f, wSR, hSR, 1);
	initAll(update_weights_sr, 0.0f, wSR, hSR, 1);

	fuseFrameMedian(d_depth_sr, d_weights_sr, d_color_sr, d_depth_lr, d_color_lr, BlurrFactors, numFrames, wSR, hSR, s);

}
// kernels

inline float dispErrStdDev(float depth){

	return 1.0f;

}


void initAll(float* array, float value, size_t w, size_t h, size_t nc){
	size_t imgSize = w * h * nc;

	for(size_t idx = 0; idx < imgSize; idx++){
		array[idx] = value;
	}

}


void calcInitialWeights(float* depthSR, float* weightsSR, size_t wSR, size_t hSR){
	size_t imgSize = wSR * hSR;

	for(size_t i = 0; i < imgSize; i++){
		if(depthSR[i] == 0.0f) weightsSR[i] = 0.0f;
		else {
			float depth_val = depthSR[i];
			float result = (focal_length * baseline / dispErrStdDev(depth_val)) / (depth_val * depth_val);
			weightsSR[i] = result;
		}

	}

}

float calcWeight(float depth, float f, float b){
	
	if(depth == 0.0f){
		return 0.0f;
	}
	float depth_val = depth * depth;
	float returnValue =(f * b / dispErrStdDev(depth)) / depth_val;
	return returnValue;
}

float2 transform3Dto2D(Eigen::Vector3f point, size_t s){

	float2 new_coord;
	new_coord.x = (point(0) / point(2)) * focal_x*s + optcenter_x*s;
	new_coord.y = (point(1) / point(2)) * focal_y*s + optcenter_y*s;

	return new_coord;

}

Eigen::Vector3f transform2Dto3D(float x, float y, float depth_val, size_t s){
	Eigen::Vector3f new_coord = Eigen::Vector3f::Ones();

	new_coord(0) = ((x - optcenter_x*s) / (focal_x*s)) * depth_val;
	new_coord(1) = ((y - optcenter_y*s) / (focal_y*s)) * depth_val;
	new_coord(2) = depth_val;

	return new_coord;

}

Eigen::Vector3f applyTransformationMatricesLRtoSR(Eigen::Vector3f pLR){
	Eigen::Vector4f tmp = Eigen::Vector4f::Ones();
	Eigen::Vector3f pSR = Eigen::Vector3f::Ones();
	
	tmp.topLeftCorner(3,1) = pLR;
	tmp(3) = 1.0f;

	tmp = pose_inv_LR * tmp;
	tmp = pose_SR * tmp;

	pSR = tmp.topLeftCorner(3,1);

	return pSR;
}

Eigen::Vector3f applyTransformationMatricesLRtoSR(Eigen::Vector3f pLR, size_t frame){
	Eigen::Vector4f tmp = Eigen::Vector4f::Ones();
	Eigen::Vector3f pSR = Eigen::Vector3f::Ones();

	tmp.topLeftCorner(3,1) = pLR;
	tmp(3) = 1.0f;

	tmp = poses_inv_LR[frame] * tmp;
	tmp = pose_SR * tmp;

	pSR = tmp.topLeftCorner(3,1);

	return pSR;
}

Eigen::Vector3f applyTransformationMatricesSRtoLR(Eigen::Vector3f pSR){
	Eigen::Vector4f tmp = Eigen::Vector4f::Ones();
	Eigen::Vector3f pLR = Eigen::Vector3f::Ones();
	
	tmp.topLeftCorner(3,1) = pSR;
	tmp(3) = 1.0f;

	tmp = pose_inv_SR * tmp;
	tmp = pose_LR * tmp;

	pLR = tmp.topLeftCorner(3,1);

	return pLR;
	
}

Eigen::Vector3f applyTransformationMatricesSRtoLR(Eigen::Vector3f pSR, size_t frame){
	Eigen::Vector4f tmp = Eigen::Vector4f::Ones();
	Eigen::Vector3f pLR = Eigen::Vector3f::Ones();
	
	tmp.topLeftCorner(3,1) = pSR;
	tmp(3) = 1.0f;

	tmp = pose_inv_SR * tmp;
	tmp = poses_LR[frame] * tmp;

	pLR = tmp.topLeftCorner(3,1);
	
	return pLR;
}



void updateStep(float* lrDepth, float* srDepth, float* srWeights, size_t wLR, size_t hLR, size_t wSR, size_t hSR){
	

	for(size_t y = 0; y < hLR; y++){
		for(size_t x = 0; x < wLR; x++){

			size_t idx = x + y * wLR;

			float xLR = (float)x + 0.5f;
			float yLR = (float)y + 0.5f;

			int s = wSR / wLR;

			xLR *= s;
			yLR *= s;

			float depthLR = lrDepth[idx];

			if (depthLR == 0.0f){
				continue;
			}

			Eigen::Vector3f pLR = transform2Dto3D(xLR, yLR, depthLR, s);

			Eigen::Vector3f pSR = applyTransformationMatricesLRtoSR(pLR);

			float2 xySR = transform3Dto2D(pSR, s);

			// compute index for 4 neigbouring pixels

			int x0 = (int)(xySR.x - 0.5f);
			int x1 = (int)(xySR.x + 0.5f);
			int y0 = (int)(xySR.y - 0.5f);
			int y1 = (int)(xySR.y + 0.5f);

			float depthLRinKeyFrame = pSR(2);

			// update pixel 00
			if (x0 >= 0 && x0 < wSR && y0 >= 0 && y0 < hSR){
				if(srDepth[x0 + y0 * wSR] == 0.0f){
					srDepth[x0 + y0 * wSR] = depthLRinKeyFrame;
					srWeights[x0 + y0 * wSR] = calcWeight(depthLR, focal_length, baseline);
				}
				else{
					srDepth[x0 + y0 * wSR] = min(srDepth[x0 + y0 * wSR], depthLRinKeyFrame);
					srWeights[x0 + y0 * wSR] = max(srWeights[x0 + y0 * wSR], calcWeight(depthLR, focal_length, baseline));
				}
			}

			// update pixel 01
			if (x0 >= 0 && x0 < wSR && y1 >= 0 && y1 < hSR){
				if(srDepth[x0 + y1 * wSR] == 0.0f){
					srDepth[x0 + y1 * wSR] = depthLRinKeyFrame;
					srWeights[x0 + y1 * wSR] = calcWeight(depthLR, focal_length, baseline);
				}
				else{
					srDepth[x0 + y1 * wSR] = min(srDepth[x0 + y1 * wSR], depthLRinKeyFrame);
					srWeights[x0 + y1 * wSR] = max(srWeights[x0 + y1 * wSR], calcWeight(depthLR, focal_length, baseline));
				}
			}


			// update pixel 10
			if (x1 >= 0 && x1 < wSR && y0 >= 0 && y0 < hSR){
				if(srDepth[x1 + y0 * wSR] == 0.0f){
					srDepth[x1 + y0 * wSR] = depthLRinKeyFrame;
					srWeights[x1 + y0 * wSR] = calcWeight(depthLR, focal_length, baseline);
				}
				else{
					srDepth[x1 + y0 * wSR] = min(srDepth[x1 + y0 * wSR], depthLRinKeyFrame);
					srWeights[x1 + y0 * wSR] = max(srWeights[x1 + y0 * wSR], calcWeight(depthLR, focal_length, baseline));
				}
			}

			// update pixel 11
			if (x1 >= 0 && x1 < wSR && y1 >= 0 && y1 < hSR){
				if(srDepth[x1 + y1 * wSR] == 0.0f){
					srDepth[x1 + y1 * wSR] = depthLRinKeyFrame;
					srWeights[x1 + y1 * wSR] = calcWeight(depthLR, focal_length, baseline);
				}
				else{
					srDepth[x1 + y1 * wSR] = min(srDepth[x1 + y1 * wSR], depthLRinKeyFrame);
					srWeights[x1 + y1 * wSR] = max(srWeights[x1 + y1 * wSR], calcWeight(depthLR, focal_length, baseline));
				}
			}
		} // loop over x dimension
	} // loop over y dimension

}

void fuseUpdates(float *srDepth, float *srUpdateDepth, float *srWeights, float *srUpdateWeights, float* srColor, float* lrColors, float BlurrFactor, size_t wSR, size_t hSR, size_t s){

	size_t wLR = wSR / s;
	size_t hLR = hSR / s; 
	
	for(size_t ySR = 0; ySR < hSR; ySR++){
		for(size_t xSR = 0; xSR < wSR; xSR++){	


			size_t idxSR = xSR + wSR * ySR;

			float SR_Depth = srDepth[idxSR];
			float SR_Update = srUpdateDepth[idxSR];

			float SR_W = srWeights[idxSR];
			float weight = srUpdateWeights[idxSR];
	
			if (weight == 0.0f) continue;
			
			Eigen::Vector3f pSR = transform2Dto3D((float)xSR + 0.5f, (float)ySR + 0.5f, SR_Update, s);

			Eigen::Vector3f pLR = applyTransformationMatricesSRtoLR(pSR);

			float2 xyLR = transform3Dto2D(pLR, s);

			if(SR_W == 0.0f){
				// no valid value in keyframe yet -> overwrite
				srDepth[idxSR] = SR_Update;
				srWeights[idxSR] = weight;

				//std::cout << "SR_W is not equal to zero. " << std::endl;

				if (weight == 0.0f){
					continue;
				}

				if (xyLR.x < 0.0f || xyLR.x > (float)wSR || xyLR.y < 0.0f || xyLR.y > (float)hSR){
					continue;
				}

			
				size_t c=0;
				size_t idxSRColor = idxSR + wSR*hSR*c;
				int pos_x = (int)(xyLR.x/(float)s);
				int pos_y = (int)(xyLR.y/(float)s);
				if(pos_x < 0 || pos_x >= wLR || pos_y < 0 || pos_y > hLR){
					continue;
				}
				srColor[idxSRColor] = lrColors[pos_x + pos_y * wLR + wLR * hLR * c];
				if (srColor[idxSRColor] == 0.0f){
					printf("color 0.0f\n");
				}
				c = 1;
				idxSRColor = idxSR + wSR*hSR*c;
				srColor[idxSRColor] = lrColors[pos_x + pos_y * wLR + wLR * hLR * c];
				if (srColor[idxSRColor] == 0.0f){
					printf("color 0.0f\n");
				}
				c = 2;
				idxSRColor = idxSR + wSR*hSR*c;
				srColor[idxSRColor] = lrColors[pos_x + pos_y * wLR + wLR * hLR * c];
				if (srColor[idxSRColor] == 0.0f){
					printf("color 0.0f\n");
				}
				continue;
			}




			if(abs(SR_Depth - SR_Update) < 0.02){
				if (SR_W + weight == 0){
					printf("Division by 0 by thread %i\n", idxSR);
				}
				float resultDepthFloat =  (((float)SR_W * (float)SR_Depth + (float)weight * (float)SR_Update)) / (((float)SR_W + (float)weight));
				srDepth[idxSR] = (int) ( resultDepthFloat );

				Eigen::Vector3f  pSR = transform2Dto3D(xSR + 0.5f, ySR + 0.5f, SR_Update, s);

				Eigen::Vector3f  pLR = applyTransformationMatricesSRtoLR(pSR);

				float2 xyLR = transform3Dto2D(pLR, s);

				size_t c=0;
				size_t idxSRColor = idxSR + wSR*hSR*c;
				int xIdx = ((size_t)xyLR.x)/s;
				int yIdx = ((size_t)xyLR.y)/s;
				if(xIdx < 0 || xIdx >= wLR || yIdx < 0 || yIdx > hLR){
					continue;
				}
				float lrColor = lrColors[xIdx + yIdx * wLR];
				float resultColor =  ( SR_W * srColor[idxSRColor] + weight * lrColor) / (SR_W + weight);
				srColor[idxSRColor] = resultColor;
				c = 1;
				idxSRColor = idxSR + wSR*hSR*c;
				lrColor = lrColors[xIdx + yIdx * wLR + c * wLR * hLR];
				resultColor =  ( SR_W * srColor[idxSRColor] + weight * lrColor) / (SR_W + weight);
				srColor[idxSRColor] = resultColor;
				c = 2;
				idxSRColor = idxSR + wSR*hSR*c;
				lrColor = lrColors[xIdx + yIdx * wLR + c * wLR * hLR];
				resultColor = ( SR_W * srColor[idxSRColor] + weight * lrColor) / (SR_W + weight);
				srColor[idxSRColor] = resultColor;

				srWeights[idxSR] += weight;
			}

		}
	}

}

void fuseFrame(float *srDepth, float *srWeights, float* srColor, float* lrDepth, float* lrColor, float BlurrFactor, size_t wSR, size_t hSR, size_t s){

	size_t wLR = wSR / s;
	size_t hLR = hSR / s;

	Eigen::Vector3f pCameraKF = transform2Dto3D(s*optcenter_x, s*optcenter_y, 0.0f, s);

	Eigen::Vector3f pCameraKF_LR = applyTransformationMatricesSRtoLR(pCameraKF);

	float2 xyCameraKF = transform3Dto2D(pCameraKF_LR, s);

	for(size_t ySR = 0; ySR < hSR; ySR++){
		for(size_t xSR = 0; xSR < wSR; xSR++){
			size_t idxSR = xSR + wSR * ySR;
			float srDepthKF = srDepth[idxSR];
			float srWeight = srWeights[idxSR];

			if (srWeight == 0.0f){
				// means that depth is also 0.0f
				srDepthKF = 1000.0f;
			}



			Eigen::Vector3f pSR = transform2Dto3D(xSR + 0.5f, ySR + 0.5f, srDepthKF, s);


			Eigen::Vector3f pLR = applyTransformationMatricesSRtoLR(pSR);


			float2 xy_pLR = transform3Dto2D(pLR, s);

			// linear function for the ray in new frame - y = mx + n
			float m = (xyCameraKF.y - xy_pLR.y) / (xyCameraKF.x - xy_pLR.x);
			float n = xyCameraKF.y - m * xyCameraKF.x;

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
					x_min = std::min(x_min, x_at_yeq0);
					x_max = std::max(x_max, x_at_yeq0);
				}
				if (0.0f <= x_at_yeqH && x_at_yeqH <= (float)wSR){
					x_min = std::min(x_min, x_at_yeqH);
					x_max = std::max(x_max, x_at_yeqH);
				}

				int x_min_idx = (int)(x_min);//(float)s);
				int x_max_idx = (int)(x_max);//(float)s);

				for(int xIdx = x_min_idx; xIdx < x_max_idx; xIdx++){
					// check if pixel in frame is more near than epsilon (-> overwrite) or in range (-> averaging) of pixel in Keyframe
					int yIdx = (int)(m * (float)xIdx + n);
					float depth = lrDepth[xIdx/s + yIdx/s * wLR];
					Eigen::Vector3f pFrame = transform2Dto3D(xIdx + 0.5f, yIdx + 0.5f, depth);
					Eigen::Vector3f pKeyFrame = applyTransformationMatricesLRtoSR(pFrame);
					float depthInKeyFrame = pKeyFrame(2);
					float2 xy_pKeyFrame = transform3Dto2D(pKeyFrame, s);
					if (fabs(xy_pKeyFrame.x - ((float)xSR + 0.5f)) <= 0.5f && fabs(xy_pKeyFrame.y - ((float)ySR + 0.5f)) <= 0.5f){
						if (srWeight != 0.0f && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < h_epsilonAveraging && !( srWeight == 0.0f)){
							// averaging
							bool averageColor = true;
							float sumDiff = 0.0f;
							for (size_t c = 0; c < 3; ++c){
								float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
								float colorKF = srColor[idxSR + wSR * hSR *c];
								float diff = fabs(colorKF - color);
								sumDiff += diff;
								if (diff > h_maxColorDiffChannel){
									averageColor = false;
									break;
								}
							}
							if (sumDiff > h_maxColorDiffSum){
								averageColor = false;
							}
							if (averageColor){
								float weight = calcWeight(depth, focal_length, baseline);
								if (weight == 0.0f){
									continue;
								}
								srDepth[idxSR] = (srWeight * srDepthKF + weight * depth) / (srWeight + weight);
								for(size_t c = 0; c < 3; ++c){
									float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
									float srColorTmp = srColor[idxSR + wSR * hSR * c];
									srColor[idxSR + wSR * hSR * c] = (srWeight * srColorTmp + BlurrFactor * weight * color) / (srWeight + BlurrFactor * weight);
								}
								srWeights[idxSR] += weight;
							}
						}
						else if ((depthInKeyFrame < srDepthKF && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < h_epsilonOverwriting) || srWeight == 0.0f){
							// overwriting

							bool overwriteColor = true;
							float sumDiff = 0.0f;
							for (size_t c = 0; c < 3; ++c){
								float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
								float colorKF = srColor[idxSR + wSR * hSR *c];
								float diff = fabs(colorKF - color);
								sumDiff += diff;
								if (diff > h_maxColorDiffChannel){
									overwriteColor = false;
									break;
								}
							}
							if (sumDiff > h_maxColorDiffSum){
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
							//else skip
						}
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
					y_min = std::min(y_min, y_at_xeq0);
					y_max = std::max(y_max, y_at_xeq0);
				}
				if (0.0f <= y_at_xeqW && y_at_xeqW <= (float)hSR){
					y_min = std::min(y_min, y_at_xeqW);
					y_max = std::max(y_max, y_at_xeqW);
				}

				int y_min_idx = (int)(y_min);//(float)s);
				int y_max_idx = (int)(y_max);//(float)s);

				for(int yIdx = y_min_idx; yIdx < y_max_idx; yIdx++){
					// check if pixel in frame is more near than epsilon (-> overwrite) or in range (-> averaging) of pixel in Keyframe
					int xIdx = (int)(m * (float)yIdx + n);
					float depth = lrDepth[(xIdx/s) + (yIdx/s) * wLR];
					Eigen::Vector3f pFrame = transform2Dto3D(xIdx + 0.5f, yIdx + 0.5f, depth, s);
					Eigen::Vector3f pKeyFrame = applyTransformationMatricesLRtoSR(pFrame);
					float2 xy_pKeyFrame = transform3Dto2D(pKeyFrame, s);
					float depthInKeyFrame = pKeyFrame(2);

					if (fabs(xy_pKeyFrame.x - ((float)xSR + 0.5f)) <= 0.5f && fabs(xy_pKeyFrame.y - ((float)ySR + 0.5f)) <= 0.5f){
						if (srWeight != 0.0f && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < h_epsilonAveraging && !( srWeight == 0.0f)){
							// averaging
							bool averageColor = true;
							float sumDiff = 0.0f;
							for (size_t c = 0; c < 3; ++c){
								float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
								float colorKF = srColor[idxSR + wSR * hSR *c];
								float diff = fabs(colorKF - color);
								sumDiff += diff;
								if (diff > h_maxColorDiffChannel){
									averageColor = false;
									break;
								}
							}
							if (sumDiff > h_maxColorDiffSum){
								averageColor = false;
							}
							if (averageColor){
								float weight = calcWeight(depth, focal_length, baseline);
								if (weight == 0.0f){
									continue;
								}
								srDepth[idxSR] = (srWeight * srDepthKF + weight * depth) / (srWeight + weight);
								for(size_t c = 0; c < 3; ++c){
									float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
									float srColorTmp = srColor[idxSR + wSR * hSR * c];
									srColor[idxSR + wSR * hSR * c] = (srWeight * srColorTmp + BlurrFactor * weight * color) / (srWeight + BlurrFactor * weight);
								}
								srWeights[idxSR] += weight;
							}
						}
						else if ((depthInKeyFrame < srDepthKF && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < h_epsilonOverwriting) || srWeight == 0.0f){
							// overwriting (only if invalid depth in key frame)
							bool overwriteColor = true;
							float sumDiff = 0.0f;
							for (size_t c = 0; c < 3; ++c){
								float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
								float colorKF = srColor[idxSR + wSR * hSR *c];
								float diff = fabs(colorKF - color);
								sumDiff += diff;
								if (diff > h_maxColorDiffChannel){
									overwriteColor = false;
									break;
								}
							}
							if (sumDiff > h_maxColorDiffSum){
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
	}
}


void fuseFrameMedian(float *srDepth, float *srWeights, float* srColor, float* lrDepth, float* lrColor, float* BlurrFactors, size_t numFrames, size_t wSR, size_t hSR, size_t s){

	float* lrDepthCopy = lrDepth;
	float* lrColorCopy = lrColor;

	size_t wLR = wSR / s;
	size_t hLR = hSR / s;

	float *updateColors = new float[(MAX_FRAMES + 1) * 3];
	float *updateWeights = new float[MAX_FRAMES + 1];

	Eigen::Vector3f pCameraKF;

	Eigen::Vector3f pCameraKF_LR;

	float2 xyCameraKF;
	std::cout << "CPU Weighted Median running" << std::endl;
	for(size_t ySR = 0; ySR < hSR; ySR++){
		for(size_t xSR = 0; xSR < wSR; xSR++){
			size_t idxSR = xSR + wSR * ySR;

			// store values of key frame
			updateColors[0] = srColor[idxSR];
			updateColors[1] = srColor[idxSR + wSR * hSR];
			updateColors[2] = srColor[idxSR + 2 * wSR * hSR];
			updateWeights[0] = srWeights[idxSR];

			for (int j = 1; j < MAX_FRAMES + 1; ++j){
				updateWeights[j] = 0.0f;
				for (int c = 0; c < 3; ++c){
					updateColors[j*3 + c] = 0.0f;
				}
			}
			
			for (size_t frame = 1; frame <= numFrames; ++frame, lrColor += wLR*hLR*3, lrDepth += wLR*hLR){
				float srDepthKF = srDepth[idxSR];
				float srWeight = srWeights[idxSR];
				if (srWeight == 0.0f){
					// means that depth is invalid, take some "random" depth to compute the ray
					srDepthKF = 1000.0f;
				}

				pCameraKF = transform2Dto3D(s*optcenter_x, s*optcenter_y, 0.0f, s);

				pCameraKF_LR = applyTransformationMatricesSRtoLR(pCameraKF, frame - 1);

				xyCameraKF = transform3Dto2D(pCameraKF_LR, s);

				Eigen::Vector3f pSR = transform2Dto3D(xSR + 0.5f, ySR + 0.5f, srDepthKF);

				Eigen::Vector3f pLR = applyTransformationMatricesSRtoLR(pSR, frame - 1);

				float2 xy_pLR = transform3Dto2D(pLR, s);

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
						x_min = std::min(x_min, x_at_yeq0);
						x_max = std::max(x_max, x_at_yeq0);
					}
					if (0.0f <= x_at_yeqH && x_at_yeqH <= (float)wSR){
						x_min = std::min(x_min, x_at_yeqH);
						x_max = std::max(x_max, x_at_yeqH);
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
						Eigen::Vector3f pFrame = transform2Dto3D(xIdx + 0.5f, yIdx + 0.5f, depth, s);
						Eigen::Vector3f pKeyFrame = applyTransformationMatricesLRtoSR(pFrame, frame - 1);
						float2 xy_pKeyFrame = transform3Dto2D(pKeyFrame, s);
						float depthInKeyFrame = pKeyFrame(2);
						if (fabs(xy_pKeyFrame.x - ((float)xSR + 0.5f)) <= 0.5f && fabs(xy_pKeyFrame.y - ((float)ySR + 0.5f)) <= 0.5f){
							if (srWeight != 0.0f && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < h_epsilonAveraging){
								// averaging
								bool averageColor = true;
								float sumDiff = 0.0f;
								for (size_t c = 0; c < 3; ++c){
									float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
									float colorKF = srColor[idxSR + wSR * hSR *c];
									float diff = fabs(colorKF - color);
									sumDiff += diff;
									if (diff > h_maxColorDiffChannel){
										averageColor = false;
										break;
									}
								}
								if (sumDiff > h_maxColorDiffSum){
									averageColor = false;
								}
								if (averageColor){
									float weight = calcWeight(depth, focal_length, baseline);
									if (weight == 0.0f){
										continue;
									}
									srDepth[idxSR] = (srWeight * srDepthKF + weight * depth) / (srWeight + weight);
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
							else if ((depthInKeyFrame < srDepthKF && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < h_epsilonOverwriting) || srWeight == 0.0f){
								// overwriting
								bool overwriteColor = true;
								float sumDiff = 0.0f;
								for (size_t c = 0; c < 3; ++c){
									float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
									float colorKF = srColor[idxSR + wSR * hSR *c];
									float diff = fabs(colorKF - color);
									sumDiff += diff;
									if (diff > h_maxColorDiffChannel){
										overwriteColor = false;
										break;
									}
								}
								if (sumDiff > h_maxColorDiffSum){
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
						y_min = std::min(y_min, y_at_xeq0);
						y_max = std::max(y_max, y_at_xeq0);
					}
					if (0.0f <= y_at_xeqW && y_at_xeqW <= (float)hSR){
						y_min = std::min(y_min, y_at_xeq0);
						y_max = std::max(y_max, y_at_xeqW);
					}

					int y_min_idx = (int)(y_min);//(float)s);
					int y_max_idx = (int)(y_max);//(float)s);

					for(int yIdx = y_min_idx; yIdx < y_max_idx; yIdx++){

						// check if pixel in frame is more near than epsilon (-> overwrite) or in range (-> averaging) of pixel in Keyframe
						int xIdx = (int)(m * (float)yIdx + n);
						if (xIdx >= wSR || xIdx < 0) {
							continue;
						}
						float depth = lrDepth[(xIdx/s) + (yIdx/s) * wLR];
						Eigen::Vector3f pFrame = transform2Dto3D(xIdx + 0.5f, yIdx + 0.5f, depth, s);
						Eigen::Vector3f pKeyFrame = applyTransformationMatricesLRtoSR(pFrame, frame - 1);
						float2 xy_pKeyFrame = transform3Dto2D(pKeyFrame, s);
						float depthInKeyFrame = pKeyFrame(2);
						if (fabs(xy_pKeyFrame.x - ((float)xSR + 0.5f)) <= 0.5f && fabs(xy_pKeyFrame.y - ((float)ySR + 0.5f)) <= 0.5f){
							if (srWeight != 0.0f && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < h_epsilonAveraging && !( srWeight == 0.0f)){
								// averaging
								bool averageColor = true;
								float sumDiff = 0.0f;
								for (size_t c = 0; c < 3; ++c){
									float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
									float colorKF = srColor[idxSR + wSR * hSR *c];
									float diff = fabs(colorKF - color);
									sumDiff += diff;
									if (diff > h_maxColorDiffChannel){
										averageColor = false;
										break;
									}
								}
								if (sumDiff > h_maxColorDiffSum){
									averageColor = false;
								}
								if (averageColor){
									float weight = calcWeight(depth, focal_length, baseline);
									if (weight == 0.0f){
										continue;
									}
									srDepth[idxSR] = (srWeight * srDepthKF + weight * depth) / (srWeight + weight);
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
							else if ((depthInKeyFrame < srDepthKF && fabs(depthInKeyFrame - srDepthKF)/depthInKeyFrame < h_epsilonOverwriting) || srWeight == 0.0f){
								// overwriting
								bool overwriteColor = true;
								float sumDiff = 0.0f;
								for (size_t c = 0; c < 3; ++c){
									float color = lrColor[xIdx/s + yIdx/s * wLR + c * wLR * hLR];
									float colorKF = srColor[idxSR + wSR * hSR *c];
									float diff = fabs(colorKF - color);
									sumDiff += diff;
									if (diff > h_maxColorDiffChannel){
										overwriteColor = false;
										break;
									}
								}
								if (sumDiff > h_maxColorDiffSum){
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

			lrDepth = lrDepthCopy;
			lrColor = lrColorCopy;

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
					// this includes key frame
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
	}
}



} // end namespace cpu

