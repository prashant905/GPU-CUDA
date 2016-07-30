#include <cuda_runtime.h>
#include <Eigen/Dense>

namespace cpu{

	void init_T_SR(Eigen::Matrix4f rot);

	void init_T_inv_SR(Eigen::Matrix4f rot);

	void init_T_inv_LR(Eigen::Matrix4f rot);

	void init_T_inv_LR(Eigen::Matrix4f* rot, size_t numFrames);

	void init_T_LR(Eigen::Matrix4f rot);

	void init_T_LR(Eigen::Matrix4f* rot, size_t numFrames);

	void initSRDepthAndWeights(float* d_depth_sr, float* d_weights_sr, size_t wSR, size_t hSR);

	void fuseDepthAndWeights(float* d_depth_sr, float* d_weights_sr, float* d_depth_lr, float* d_color_sr, float* d_color_lr, float BlurrFactor, size_t s, size_t wSR, size_t hSR, bool useRay = false);

	void fuseDepthAndWeightsWithRayMedian(float* d_depth_sr, float* d_weights_sr, float* d_depth_lr, float* d_color_sr, float* d_color_lr, float* BlurrFactors, size_t numFrames, size_t s, size_t wSR, size_t hSR);

	float2 transform3Dto2D(Eigen::Vector3f point, size_t s = 2);

	Eigen::Vector3f transform2Dto3D(float x, float y, float depth_val, size_t s = 2);

	Eigen::Vector3f applyTransformationMatricesLRtoSR(Eigen::Vector3f pLR);

	Eigen::Vector3f applyTransformationMatricesLRtoSR(Eigen::Vector3f pLR, size_t frame);

	Eigen::Vector3f applyTransformationMatricesSRtoLR(Eigen::Vector3f pSR);

	Eigen::Vector3f applyTransformationMatricesSRtoLR(Eigen::Vector3f pSR, size_t frame);

	void initAll(float* array, float value, size_t w, size_t h, size_t nc);

	void calcInitialWeights(float* depthSR, float* weightsSR, size_t wSR, size_t hSR);

	float calcWeight(float depth, float f, float b);

	void updateStep(float* lrDepth, float* srDepth, float* srWeights, size_t wLR, size_t hLR, size_t wSR, size_t hSR);

	void fuseUpdates(float *srDepth, float *srUpdateDepth, float *srWeights, float *srUpdateWeights, float* srColor, float* lrColor, float BlurrFactor, size_t wSR, size_t hSR, size_t s);

	void fuseFrame(float *srDepth, float *srWeights, float* srColor, float* lrDepth, float* lrColor, float BlurrFactor, size_t wSR, size_t hSR, size_t s);

	void fuseFrameMedian(float *srDepth, float *srWeights, float* srColor, float* lrDepth, float* lrColor, float* BlurrFactors, size_t numFrames, size_t wSR, size_t hSR, size_t s);

}
