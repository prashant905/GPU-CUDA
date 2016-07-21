#ifndef TUM_BENCHMARK_H
#define TUM_BENCHMARK_H

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


bool loadAssoc(const std::string &assocFile, std::vector<Eigen::Matrix4f> &poses, std::vector<std::string> &filesDepth, std::vector<std::string> &filesColor)
{
    if (assocFile.empty())
        return false;

    //load transformations from CVPR RGBD datasets benchmark
    std::ifstream assocIn;
    assocIn.open(assocFile.c_str());
    if (!assocIn.is_open())
        return false;

    // first load all groundtruth timestamps and poses
    std::string line;
    while (std::getline(assocIn, line))
    {
        if (line.empty() || line.compare(0, 1, "#") == 0)
            continue;
        std::istringstream iss(line);
        double timestampPose, timestampDepth, timestampColor;
        double tx, ty, tz;
        double qx, qy, qz, qw;
        std::string fileDepth, fileColor;
        if (!(iss >> timestampPose >> tx >> ty >> tz >> qx >> qy >> qz >> qw >> timestampDepth >> fileDepth >> timestampColor >> fileColor))
            break;

        filesDepth.push_back(fileDepth);
        filesColor.push_back(fileColor);

        //std::cout << "qx=" << qx << ", qy=" << qy << ", qz=" << qz << ", qw=" << qw << std::endl;
        //std::cout << "tx=" << tx << ", ty=" << ty << ", tz=" << tz << std::endl;
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        Eigen::Vector3f tVec(tx, ty, tz);
        pose.topRightCorner(3,1) = tVec;
        Eigen::Quaternionf quat(qw, qx, qy, qz);
        pose.topLeftCorner(3,3) = quat.toRotationMatrix();
        poses.push_back(pose);
    }
    assocIn.close();

    return true;
}

cv::Mat loadColor(const std::string &filename)
{
    cv::Mat imgColor = cv::imread(filename);
    // convert gray to float
    cv::Mat color;
    imgColor.convertTo(color, CV_32FC3, 1.0f / 255.0f);
    return color;
}


cv::Mat loadDepth(const std::string &filename)
{
    //fill/read 16 bit depth image
    cv::Mat imgDepthIn = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    cv::Mat imgDepth;
    imgDepthIn.convertTo(imgDepth, CV_32FC1, (1.0 / 5000.0));
    return imgDepth;
}

#endif
