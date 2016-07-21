#include <sstream>
#include <iomanip>
#include <fstream>
#include <vector>

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
/*
bool depthToVertexMap(const Eigen::Matrix3f &K, const cv::Mat &depth, cv::Mat &vertexMap)
{
    int w = depth.cols;
    int h = depth.rows;
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxInv = 1.0f / K(0, 0);
    float fyInv = 1.0f / K(1, 1);

    vertexMap = cv::Mat::zeros(h, w, CV_32FC3);
    float* ptrVert = (float*)vertexMap.data;
    const float* ptrDepth = (const float*)depth.data;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float depthVal = ptrDepth[y*w + x];
            float x0 = (float(x) - cx) * fxInv;
            float y0 = (float(y) - cy) * fyInv;
            float scale = std::sqrt(x0*x0 + y0*y0 + 1.0f);
            scale = 1.0;
            depthVal = depthVal * scale;

            size_t off = (y*w + x) * 3;
            ptrVert[off] = x0 * depthVal;
            ptrVert[off+1] = y0 * depthVal;
            ptrVert[off+2] = depthVal;
        }
    }

    return true;
}
*/

void transformVertexMap(const Eigen::Matrix3f &R, const Eigen::Vector3f &t, cv::Mat &vertexMap)
{
    int w = vertexMap.cols;
    int h = vertexMap.rows;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            cv::Vec3f pt = vertexMap.at<cv::Vec3f>(y, x);
            if (pt.val[2] == 0.0 || std::isnan(pt.val[2]))
                continue;
            Eigen::Vector3f ptTf(pt.val[0], pt.val[1], pt.val[2]);
            ptTf = R * ptTf + t;
            vertexMap.at<cv::Vec3f>(y, x) = cv::Vec3f(ptTf[0], ptTf[1], ptTf[2]);
        }
    }
}


bool savePlyFile(const std::string &filename, const std::vector<Eigen::Vector3f> &pts, const std::vector<Eigen::Vector3f> &colors)
{
    if (pts.empty())
        return false;

    std::ofstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open())
        return false;

    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << pts.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "property uchar red" << std::endl;
    plyFile << "property uchar green" << std::endl;
    plyFile << "property uchar blue" << std::endl;
    plyFile << "element face 0" << std::endl;
    plyFile << "property list uchar int vertex_indices" << std::endl;
    plyFile << "end_header" << std::endl;

    for (size_t i = 0; i < pts.size(); i++)
    {
        plyFile << pts[i][0] << " " << pts[i][1] << " " << pts[i][2];
        plyFile << " " << (int)colors[i][0] << " " << (int)colors[i][1] << " " << (int)colors[i][2];
        plyFile << std::endl;
    }
    plyFile.close();

    return true;
}


bool savePlyFile(const std::string &filename, const cv::Mat &color, const cv::Mat &vertexMap)
{
    // convert frame to points vector and colors vector
    std::vector<Eigen::Vector3f> pts;
    std::vector<Eigen::Vector3f> colors;
    for (int y = 0; y < vertexMap.rows; ++y)
    {
        for (int x = 0; x < vertexMap.cols; ++x)
        {
            cv::Vec3f pt = vertexMap.at<cv::Vec3f>(y, x);
            if (pt.val[2] == 0.0 || std::isnan(pt.val[2]))
                continue;
            pts.push_back(Eigen::Vector3f(pt.val[0], pt.val[1], pt.val[2]));

            cv::Vec3b c = color.at<cv::Vec3b>(y, x);
            colors.push_back(Eigen::Vector3f(c.val[2], c.val[1], c.val[0]));
        }
    }

    return savePlyFile(filename, pts, colors);
}
