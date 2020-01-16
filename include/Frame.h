#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>
using namespace std;

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class Frame
{
public:
    // 默认构造函数
    Frame();
    // 复制构造函数
    Frame(const Frame &frame);

    // 构造函数.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, cv::Mat &K, cv::Mat &distCoef);

    // 提取的关键点存放在mvKeys和mDescriptors中
    void ExtractORB(int flag, const cv::Mat &im);

    //设置相机的位姿态mTcw
    void SetPose(cv::Mat Tcw);

    //获得R与t
    cv::Mat GetPose_r();
    cv::Mat GetPose_t();

    // 根据mTcw来获得mRcw, mtcw
    void UpdatePoseMatrices();

    // 找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点
    vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel = -1, const int maxLevel = -1) const;

    // 计算一个特征点对应的格子
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

public:
    // 特征点提取器
    ORBextractor *mpORBextractorLeft;

    // 帧的时间戳
    double mTimeStamp;

    // 相机参数
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // 特征点数量
    int N;

    // mvKeysUn:校正mvKeys后的特征点
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysUn;

    // 图像对应的描述子
    cv::Mat mDescriptors;

    // 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    // 每个格子分配的特征点，将图像分成格子，保证提取的特征点比较均匀
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // 世界坐标系到相机的坐标系变换矩阵
    cv::Mat mTcw;

    // 帧ID
    long unsigned int mnId;
    static long unsigned int nNextId;

    // 金字塔信息
    int mnScaleLevels;      //图像提金字塔的层数
    float mfScaleFactor;    //图像提金字塔的尺度因子
    float mfLogScaleFactor; //
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    // 用于确定画格子时的边界
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

private:
    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // 旋转,平移
    cv::Mat mRcw;
    cv::Mat mtcw;
    cv::Mat mRwc;
    cv::Mat mtwc;

    static bool mbInitialComputations;
};

} // namespace ORB_SLAM2

#endif // FRAME_H
