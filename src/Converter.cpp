#include "Converter.h"

namespace ORB_SLAM2
{
//将cv::Mat转化为Eigen类型

Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double, 3, 1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double, 3, 1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double, 3, 3> Converter::toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double, 3, 3> M;

    M << cvMat3.at<float>(0, 0), cvMat3.at<float>(0, 1), cvMat3.at<float>(0, 2),
        cvMat3.at<float>(1, 0), cvMat3.at<float>(1, 1), cvMat3.at<float>(1, 2),
        cvMat3.at<float>(2, 0), cvMat3.at<float>(2, 1), cvMat3.at<float>(2, 2);

    return M;
}

std::vector<float> Converter::toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double, 3, 3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

} // namespace ORB_SLAM2
