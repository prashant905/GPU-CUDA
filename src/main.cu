#include <iostream>
#include <vector>
#include <strstream>

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif

#include <cstdlib>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "aux.cuh"

#include "tum_benchmark.hpp"


#include "convolution.cuh"
#include "global.cuh"




#define STR1(x)  #x
#define STR(x)  STR1(x)

// comment out to use weighted averaging
#define MEDIAN
// comment in to use cpu version
// #define CPU

typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;


#ifdef MEDIAN
float h_maxColorDiffChannel = MAXCOLORDIFFCHANNELMEDIAN;
float h_maxColorDiffSum = MAXCOLORDIFFSUMMEDIAN;
float h_epsilonAveraging = EPSILONAVERAGINGMEDIAN;
float h_epsilonOverwriting = EPSILONOVERWRITINGMEDIAN;
#else
float h_maxColorDiffChannel = MAXCOLORDIFFCHANNEL;
float h_maxColorDiffSum = MAXCOLORDIFFSUM;
float h_epsilonAveraging = EPSILONAVERAGING;
float h_epsilonOverwriting = EPSILONOVERWRITING;
#endif




int main(int argc, char *argv[])
{
	Timer timer;

#ifdef MEDIAN
	bool median = true;
#else
	bool median = false;
#endif

    std::string dataFolder = std::string(STR(FUSION_CUDA_SOURCE_DIR)) + "/data/rgbd_dataset_freiburg1_xyz/";
//    std::string dataFolder = std::string(STR(FUSION_CUDA_SOURCE_DIR)) + "/data/rgbd_dataset_freiburg3_sitting_xyz_validation/";

    // load file names
    // generate assoc file according to scheme:
    // ./associate.py ~/data/rgbd_dataset_freiburg1_xyz/groundtruth.txt ~/data/rgbd_dataset_freiburg1_xyz/depth.txt > tmp.txt
    // ./associate.py tmp.txt ~/data/rgbd_dataset_freiburg1_xyz/rgb.txt > ~/data/rgbd_dataset_freiburg1_xyz/rgbd_assoc_poses.txt
    std::string assocFile = dataFolder + "rgbd_assoc_poses.txt";

    // get parameters

    // scaling factor per dimension
    size_t s = 2;
    getParam("s", s, argc, argv);

	// scaling factor per dimension
    bool extrapolate = false;
    getParam("ex", extrapolate, argc, argv);

	// scaling factor per dimension
    bool sharpening = false;
    getParam("um", sharpening, argc, argv);


    // initialize intrinsic matrix
    Eigen::Matrix3f K;
   K <<    517.3, 0.0, 318.6,
            0.0, 516.5, 255.3,
            0.0, 0.0, 1.0;

    std::vector<Eigen::Matrix4f> poses;
    std::vector<std::string> filesDepth;
    std::vector<std::string> filesColor;
    if (!loadAssoc(assocFile, poses, filesDepth, filesColor))
    {
        std::cout << "Assoc file could not be loaded!" << std::endl;
        return 1;
    }
    int numFrames = filesDepth.size();

    // initialize cuda context
    cudaDeviceSynchronize(); CUDA_CHECK;

	// Save results for each keyfram
	std::stringstream fileName;
	/*
    fileName << h_maxColorDiffChannel;
    fileName << "_";
    fileName << h_maxColorDiffSum;
    fileName << "_";
    fileName << h_epsilonAveraging;
    fileName << "_";
    fileName << h_epsilonOverwriting;
    fileName << ".png";
	*/

	std::string resultFolder = "data/Results/s_eq_2/";
    if (s == 1){
    	resultFolder = "data/Results/s_eq_1/";
    }
    else if (s == 4){
    	resultFolder = "data/Results/s_eq_4/";
    }

    // first iteration outside to init super-resolution image

    // load input frame
    std::string fileColor = filesColor[0];
    std::string fileDepth = filesDepth[0];
    std::cout << "File 0" << ": " << fileColor << ", " << fileDepth << std::endl;

    cv::Mat color = loadColor(dataFolder + fileColor);
    cv::Mat depth = loadDepth(dataFolder + fileDepth);

/*
    // get pose
   	Eigen::Matrix4f pose1 = poses[0];

    cv::Mat vertexMap;
    depthToVertexMap(K, depth, vertexMap);

    cv::Mat color2;
    color.convertTo(color2, CV_8UC3, 255.0);
    savePlyFile("Keyframe0.ply", color2, vertexMap);

    Eigen::Matrix3f R = pose1.topLeftCorner(3,3);
    Eigen::Vector3f t = pose1.topRightCorner(3,1);
    transformVertexMap(R, t, vertexMap);

    savePlyFile("Keyframe.ply", color2, vertexMap);
    std::cout << "pose:\n" << std::endl << pose1 << std::endl;

*/

	// get pose
    Eigen::Matrix4f pose = poses[0];

    Eigen::Matrix4f pose_inv = pose.inverse();

#ifdef CPU
    cpu::init_T_SR(pose_inv);
    cpu::init_T_inv_SR(pose);
#else
	init_T_SR(pose_inv);
    init_T_inv_SR(pose);
#endif


    // init super-resolution keyframe
    size_t wLR = color.cols;
    size_t hLR = color.rows;
    size_t nc = color.channels();
    size_t wSR = wLR * s;
    size_t hSR = hLR * s;

    cv::Mat srColor = cv::Mat(hSR, wSR, CV_32FC3);
    cv::Mat srDepth = cv::Mat(hSR, wSR, CV_32FC1);
    float* srDataColor = new float[wSR*hSR*nc];
    float* srDataDepth = new float[wSR*hSR];
#if defined(MEDIAN) && defined(CPU)
	float* lrDataColor = new float[MAX_FRAMES * wLR * hLR * nc];
	float* lrDataDepth = new float[MAX_FRAMES * wLR*hLR];
#else
	float* lrDataColor = new float[wLR*hLR*nc];
	float* lrDataDepth = new float[wLR*hLR];
#endif

//#ifdef CPU
	float* h_data_colorSR = new float[wSR * hSR * nc];
//#endif



#ifndef CPU
    float* d_color_data;
    float* d_depth_data;
//	int* d_i_weights_sr;

    if(median){
		cudaMalloc(&d_color_data, MAX_FRAMES * wLR * hLR * nc * sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&d_depth_data, MAX_FRAMES * wLR * hLR * sizeof(float));
		CUDA_CHECK;
    }
    else{
		cudaMalloc(&d_color_data, wLR * hLR * nc * sizeof(float));
		CUDA_CHECK;
		cudaMalloc(&d_depth_data, wLR * hLR * sizeof(float));
		CUDA_CHECK;
    }
#endif

//	int* iLRDataDepth = new int[wLR * hLR];
//    cudaMalloc(&d_i_weights_sr, wSR * hSR * sizeof(int));
//    CUDA_CHECK;
//	int* d_iDepth_data;
//	cudaMalloc(&d_iDepth_data, wLR * hLR * sizeof(int));
//    CUDA_CHECK;
//    int* d_i_depth_dataSR;
//    cudaMalloc(&d_i_depth_dataSR, wSR*hSR*sizeof(int));
//    CUDA_CHECK;

    
#ifdef CPU
	float* h_depth_dataSR = new float[wSR * hSR];
	float* h_weights_SR = new float[wSR * hSR];
#else
	float* d_color_dataSR;
    float* d_depth_dataSR;
    float* d_weightsSR;
    cudaMalloc(&d_color_dataSR, wSR * hSR * nc * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_depth_dataSR, wSR * hSR * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_weightsSR, wSR * hSR * sizeof(float));
    CUDA_CHECK;
#endif


	// Depth Map Visualization
	
	std::string outputFolder = "data/depthToMesh/";
    std::stringstream ss;
	
    ss << outputFolder << "mesh_" << std::setw(5) << std::setfill('0') << 0 << ".ply";
	cv::Mat vertexMap;
    depthToVertexMap(K, srDepth, vertexMap);

    cv::Mat color2;
    srColor.convertTo(color2, CV_8UC3, 255.0);
    savePlyFile("Keyframe0.ply", color2, vertexMap);

//	ss.str("");
//	ss << outputFolder << "mesh_" << std::setw(5) << std::setfill('0') << 0 << ".ply";
//	depthToVertexMap(K, srDepth, vertexMap);
//	srColor.convertTo(color2, CV_8UC3, 255.0);
//	savePlyFile(ss.str(), color2, vertexMap);
//
//	Eigen::Matrix3f K2 = Eigen::Matrix3f(K);
//	K2(0,0) *= s;
//	K2(1,1) *= s;
//	K2(0,2) *= s;
//	K2(1,2) *= s;
//
//    saveRgbdFrameAsMesh(K2, srDepth, ss.str());
	
    // process frames
    for (size_t i = 0; i < numFrames; i += MAX_FRAMES + 1)
    {
		// Re-initialize the keyframe every MAX_FRAMES
		if(i % (MAX_FRAMES + 1) == 0){
			
			// Save fused color and depth image
			if(i > 0){
				fileName.str("");
				if(median) fileName << "wMedian_";
				else fileName << "wAverage_";
				if(extrapolate) fileName << "ex_";
				if(sharpening) fileName << "um_";
				fileName << i;
				fileName << "_";
				fileName << h_maxColorDiffChannel;
				fileName << "_";
				fileName << h_maxColorDiffSum;
				fileName << "_";
				fileName << h_epsilonAveraging;
				fileName << "_";
				fileName << h_epsilonOverwriting;
				fileName << ".png";

				cv::imwrite(resultFolder + "fusedColor_" + fileName.str(), srColor*255.f);
				cv::imwrite(resultFolder + "fusedDepth_" + fileName.str(), srDepth*255.f);
			}
			

			// load input frame
	        fileColor = filesColor[i];
	        fileDepth = filesDepth[i];
	        std::cout << "File " << i << ": " << fileColor << ", " << fileDepth << std::endl;

	        color = loadColor(dataFolder + fileColor);
	        depth = loadDepth(dataFolder + fileDepth);

			convert_mat_to_layered(lrDataColor, color);
	        convert_mat_to_layered(lrDataDepth, depth);

			// get pose
	        pose = poses[i];
			pose_inv = pose.inverse();

			// for visualization of input images
			cpu::upsampleColor(h_data_colorSR, lrDataColor, wSR, hSR, nc, s);
			convert_layered_to_mat(srColor, h_data_colorSR);

			if(sharpening){			
				applyUnsharpMasking(lrDataColor, lrDataColor, wLR, hLR, nc, 0.8f);
			}


			// Copy LR Color image to the device
			#ifndef CPU
	        cudaMemcpy(d_color_data, lrDataColor, wLR * hLR * nc * sizeof(float), cudaMemcpyHostToDevice);
	        CUDA_CHECK;
			#endif

			#ifdef CPU
    		cpu::init_T_SR(pose_inv);
    		cpu::init_T_inv_SR(pose);
			#endif

			if(extrapolate){
				extrapolateDepth(lrDataDepth, wLR, hLR);
			}

			#ifndef CPU
			// Copy LR Depth image to the device
	        cudaMemcpy(d_depth_data, lrDataDepth, wLR * hLR * sizeof(float), cudaMemcpyHostToDevice);
	        CUDA_CHECK;
			#endif



			cv::waitKey(0);

			init_T_SR(pose_inv);
    		init_T_inv_SR(pose);

			#ifndef CPU
			// upsampling via textures for color image
			upsampleTexture(d_color_data, wLR, hLR);
			CUDA_CHECK;
			initKeyFrameColor(d_color_dataSR, wSR, hSR);
			CUDA_CHECK;
			upsampleSimple(d_depth_data, d_depth_dataSR, s, wSR, hSR);
			CUDA_CHECK;
			initSRWeights(d_depth_dataSR, d_weightsSR, wSR, hSR);
			CUDA_CHECK;
			#endif




			#ifdef CPU
			memcpy(srDataDepth, h_depth_dataSR, wSR * hSR * sizeof(float));
			#endif

    	
//
//			Eigen::Matrix3f K2 = Eigen::Matrix3f(K);
//			K2(0,0) *= s;
//			K2(1,1) *= s;
//			K2(0,2) *= s;
//			K2(1,2) *= s;
//
//			saveRgbdFrameAsMesh(K2, srDepth, ss.str());
//
//			// R = pose.topLeftCorner(3,3);
//			// t = pose.topRightCorner(3,1);
//			//transformVertexMap(R, t, vertexMap);
//
//			// savePlyFile("Keyframe.ply", color2, vertexMap);
//		}

		if (median){
			size_t counter = 0;
			float* BlurrFactors = new float[MAX_FRAMES];
			Eigen::Matrix4f* pose_lr = new Eigen::Matrix4f[MAX_FRAMES];
			Eigen::Matrix4f* pose_inv_lr = new Eigen::Matrix4f[MAX_FRAMES];
			
			// Define timer (mem tranfers)
			timer.start();
			for (size_t frame = i + 1; frame < i + MAX_FRAMES + 1; ++frame, ++counter){
				// load input frame
				fileColor = filesColor[frame];
				fileDepth = filesDepth[frame];
				std::cout << "File " << frame << ": " << fileColor << ", " << fileDepth << std::endl;

				color = loadColor(dataFolder + fileColor);
				depth = loadDepth(dataFolder + fileDepth);

#if defined(MEDIAN) && defined(CPU)
				convert_mat_to_layered(lrDataColor + counter * wLR * hLR * nc, color);
				convert_mat_to_layered(lrDataDepth + counter * wLR * hLR, depth);
#else
				convert_mat_to_layered(lrDataColor, color);
				convert_mat_to_layered(lrDataDepth, depth);
#endif

				// get pose
				pose_lr[counter] = poses[frame];
				pose_inv_lr[counter] = pose_lr[counter].inverse();

				if(sharpening){
#if defined(MEDIAN) && defined(CPU)
					applyUnsharpMasking(lrDataColor + counter * wLR * hLR * nc, lrDataColor + counter * wLR * hLR * nc, wLR, hLR, nc, 0.8f);
#else
					applyUnsharpMasking(lrDataColor, lrDataColor, wLR, hLR, nc, 0.8f);
#endif
				}

				// Start timer (mem transfers)
				#ifndef CPU
				// Copy LR Color image to the device
				cudaMemcpy(d_color_data + counter * wLR * hLR * nc, lrDataColor, wLR * hLR * nc * sizeof(float), cudaMemcpyHostToDevice);
				CUDA_CHECK;
				// Copy LR Depth image to the device
				cudaMemcpy(d_depth_data + counter * wLR * hLR, lrDataDepth, wLR * hLR * sizeof(float), cudaMemcpyHostToDevice);
				CUDA_CHECK;
				cudaDeviceSynchronize();
				#endif

				BlurrFactors[counter] = calcBlurFactor(color);
				// Stop timer (mem tranfers)

			}
			timer.end();
			std::cout << "Time for memory transfers: " << timer.get() << std::endl;
			#ifdef CPU
			cpu::init_T_inv_LR(pose_lr, MAX_FRAMES);
			cpu::init_T_LR(pose_inv_lr, MAX_FRAMES);
			

			
			cudaMemcpy(srDataColor, d_color_dataSR, wSR * hSR * nc * sizeof(float), cudaMemcpyDeviceToHost);
			CUDA_CHECK;
			cudaMemcpy(srDataDepth, d_depth_dataSR, wSR * hSR * sizeof(float), cudaMemcpyDeviceToHost);
			CUDA_CHECK;
			convert_layered_to_mat(srColor, srDataColor);
			#endif
			//end timer mem tranfers

			for (int j=0; j<wSR*hSR; ++j){
				srDataDepth[j] /= 5.0f;
			}
			convert_layered_to_mat(srDepth, srDataDepth);
			showImage("SR-Color-Ray", srColor, 100, 100+hLR+40);
			showImage("SR-Depth-Ray", srDepth, 100+wLR+40, 100+hLR+40);
		}
		else { // use averaging
			float timeAverage = 0.0f;
			for (size_t frame = i + 1; frame < i + MAX_FRAMES + 1; ++frame){
				// load input frame
				fileColor = filesColor[frame];
				fileDepth = filesDepth[frame];
				std::cout << "File " << frame << ": " << fileColor << ", " << fileDepth << std::endl;

				color = loadColor(dataFolder + fileColor);
				depth = loadDepth(dataFolder + fileDepth);

				convert_mat_to_layered(lrDataColor, color);
				convert_mat_to_layered(lrDataDepth, depth);

				// get pose
				pose = poses[frame];
				pose_inv = pose.inverse();

				// Start timer (memory transfers)

				if(sharpening){
					applyUnsharpMasking(lrDataColor, lrDataColor, wLR, hLR, nc, 0.8f);
				}

				#ifndef CPU
				// Copy LR Color image to the device
				cudaMemcpy(d_color_data, lrDataColor, wLR * hLR * nc * sizeof(float), cudaMemcpyHostToDevice);
				CUDA_CHECK;
				// Copy LR Depth image to the device
				cudaMemcpy(d_depth_data, lrDataDepth, wLR * hLR * sizeof(float), cudaMemcpyHostToDevice);
				CUDA_CHECK;
				#endif

				//depthToVertexMap(K, depth, vertexMap);
				/*
				cv::Mat color2;
				color.convertTo(color2, CV_8UC3, 255.0);
				std::stringstream str;
				str << "Frame";
				str << i;
				str << ".ply";
				*/

		//        savePlyFile(str.str() , color2, vertexMap);


				//R = pose.topLeftCorner(3,3);
				//t = pose.topRightCorner(3,1);
				//transformVertexMap(R, t, vertexMap);

				//savePlyFile(str.str(), color2, vertexMap);

				#ifdef CPU
				cpu::init_T_LR(pose_inv);
				cpu::init_T_inv_LR(pose);
				#else
				init_T_LR(&pose_inv);
				CUDA_CHECK;
				init_T_inv_LR(&pose);
				CUDA_CHECK;
				#endif
				
				cudaDeviceSynchronize(); // Always invoke this 
				


				float BlurrFactor = calcBlurFactor(color);

				#ifdef CPU
				bool useRay = false;
				// Start timer CPU
				timer.start();
				cpu::fuseDepthAndWeights(h_depth_dataSR, h_weights_SR, lrDataDepth, h_data_colorSR, lrDataColor, BlurrFactor, s, wSR, hSR, useRay);
				timer.end();
				timeAverage += timer.get();
				std::cout << "CPU-Time (fuseDepthAndWeights with ray, averaging): " << timer.get() << std::endl;
				// End timer CPU
				memcpy(srDataDepth, h_depth_dataSR, wSR * hSR * sizeof(float));
				convert_layered_to_mat(srColor, h_data_colorSR);
				#else
				// Start timer GPU
				timer.start();
				fuseDepthAndWeightsWithRay(d_depth_dataSR, d_weightsSR, d_depth_data, d_color_dataSR, d_color_data, BlurrFactor, s, wSR, hSR);
				CUDA_CHECK;
				timer.end();
				timeAverage += timer.get();
				// End timer GPU
//        		transformDepthFloatToInt(d_depth_data, d_iDepth_data, wLR, hLR);
//        		CUDA_CHECK;
//        		loadLRImage(d_color_data, wLR, hLR);
//		        fuseDepthAndWeights(d_i_depth_dataSR, d_i_weights_sr, d_iDepth_data, d_color_dataSR, BlurrFactor, s, wSR, hSR);
//				CUDA_CHECK;
				cudaMemcpy(srDataColor, d_color_dataSR, wSR * hSR * nc * sizeof(float), cudaMemcpyDeviceToHost);
				CUDA_CHECK;
				cudaMemcpy(srDataDepth, d_depth_dataSR, wSR * hSR * sizeof(float), cudaMemcpyDeviceToHost);
				CUDA_CHECK;
				convert_layered_to_mat(srColor, srDataColor);
				#endif

				// End timer (memory transfers)

				for (int j=0; j<wSR*hSR; ++j){
					srDataDepth[j] /= 5.0f;
				}
				
				convert_layered_to_mat(srDepth, srDataDepth);
				showImage("SR-Color-Ray", srColor, 100, 100+hLR+40);
				showImage("SR-Depth-Ray", srDepth, 100+wLR+40, 100+hLR+40);
			}
#ifdef CPU
			std::cout << "CPU-Time (fuseDepthAndWeightsWithRay, averaging): " << timeAverage << std::endl;
#else
			std::cout << "GPU-Time (fuseDepthAndWeightsWithRay, averaging): " << timeAverage << std::endl;
#endif

		}
    }


    fileName << h_maxColorDiffChannel;
    fileName << "_";
    fileName << h_maxColorDiffSum;
    fileName << "_";
    fileName << h_epsilonAveraging;
    fileName << "_";
    fileName << h_epsilonOverwriting;
    fileName << ".png";

	resultFolder = "Results/s_eq_2/";
    if (s == 1){
    	resultFolder = "Results/s_eq_1/";
    }

    cv::imwrite(resultFolder + "fusedColor_" + fileName.str(), srColor*255.f);
    cv::imwrite(resultFolder + "fusedDepth_" + fileName.str(), srDepth*255.f);


#ifndef CPU
	cudaFree(d_color_data);
	cudaFree(d_depth_data);
	cudaFree(d_weightsSR);
	cudaFree(d_color_dataSR);
	cudaFree(d_depth_dataSR);
//	cudaFree(d_i_weights_sr);
//	cudaFree(d_iDepth_data);
#endif

    cv::destroyAllWindows();
    cv::waitKey(0);
    return 0;
}
