#ifndef GLOBAL_CUH
#define GLOBAL_CUH


#include <cuda_runtime.h>

#define MAX_FRAMES 20

// global texture variables for color channels
texture<float,2,cudaReadModeElementType> texRefColorR;
texture<float,2,cudaReadModeElementType> texRefColorG;
texture<float,2,cudaReadModeElementType> texRefColorB;
//texture<float,2,cudaReadModeElementType> texRefDepth;
//texture<float,2,cudaReadModeElementType> texRefWeights;

// define parameters for the ray version of depth & color fusion

#define MAXCOLORDIFFCHANNEL 0.05f
#define MAXCOLORDIFFSUM 0.1f
#define EPSILONAVERAGING 0.05f
#define EPSILONOVERWRITING 0.50f

#define MAXCOLORDIFFCHANNELMEDIAN 0.05f
#define MAXCOLORDIFFSUMMEDIAN 0.1f
#define EPSILONAVERAGINGMEDIAN 0.05f
#define EPSILONOVERWRITINGMEDIAN 0.50f

__constant__ float maxColorDiffChannel = MAXCOLORDIFFCHANNEL;
__constant__ float maxColorDiffSum = MAXCOLORDIFFSUM;
__constant__ float epsilonAveraging = EPSILONAVERAGING;
__constant__ float epsilonOverwriting = EPSILONOVERWRITING;

__constant__ float maxColorDiffChannelMedian = MAXCOLORDIFFCHANNELMEDIAN;
__constant__ float maxColorDiffSumMedian = MAXCOLORDIFFSUMMEDIAN;
__constant__ float epsilonAveragingMedian = EPSILONAVERAGINGMEDIAN;
__constant__ float epsilonOverwritingMedian = EPSILONOVERWRITINGMEDIAN;


__constant__ float scalingIntToFloatFactor = 1000000.f;

// rotation matrices
__constant__ float d_T_inv_SR[12];
__constant__ float d_T_SR[12];
__constant__ float d_T_LR[12 * MAX_FRAMES];
__constant__ float d_T_inv_LR[12 * MAX_FRAMES];

#endif //GLOBAL_CUH
