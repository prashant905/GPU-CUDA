# Copyright (c) 2015 Robert Maier
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(RgbdKeyframeFusion C CXX)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE RelWithDebInfo FORCE)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#==============================================================================
# Dependencies
#==============================================================================

# Eigen
SET(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/third_party/include/)
FIND_PACKAGE(Eigen REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch sm_35 -gencode=arch=compute_35,code=sm_35 --ptxas-options=-v --use_fast_math)


# OpenCV
FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS opencv_core opencv_highgui opencv_imgproc REQUIRED)

#==============================================================================
# Application
#==============================================================================

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -funroll-loops" )

# preprocessor define: source directory
ADD_DEFINITIONS(-DFUSION_CUDA_SOURCE_DIR=${PROJECT_SOURCE_DIR})

# applications
FILE(GLOB incs "src/*.hpp")
FILE(GLOB srcs "src/*.cpp")
FILE(GLOB cuda_incs "src/*.cuh")
FILE(GLOB cuda_srcs "src/*.cu")

CUDA_ADD_EXECUTABLE(rgbd_kf_fusion_cuda ${incs} ${srcs} ${cuda_incs} ${cuda_srcs} )
TARGET_LINK_LIBRARIES(rgbd_kf_fusion_cuda ${OpenCV_LIBS} -lcublas)


