#ifndef CONVERTER_H
#define CONVERTER_H

#include <opencv2/core/core.hpp>

#include <Eigen/Dense>
namespace ORB_SLAM2
{
class Converter
{
public:
    static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);
    static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);
    static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);
    static std::vector<float> toQuaternion(const cv::Mat &M);
};

} // namespace ORB_SLAM2

#endif // CONVERTER_H
