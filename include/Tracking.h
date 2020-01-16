#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Frame.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "System.h"

namespace ORB_SLAM2
{
class System;

class Tracking
{

public:
    Tracking(System *pSys, const string &strSettingPath);

    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

public:
    // 跟踪状态量
    enum eTrackingState
    {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };

    eTrackingState mState;

    // 当前帧
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // 初始化时前两帧相关变量
    std::vector<int> mvIniMatches; // 跟踪初始化时前两帧之间的匹配
    // 存放前一帧的特征点
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

protected:
    // Main tracking function.
    void Track();

    // initialization for monocular
    int MonocularInitialization();

    //bool TrackLastFrame();

    //ORB
    // 单目在初始化的时候使用mpIniORBextractor而不是mpORBextractorLeft，
    // mpIniORBextractor属性中提取的特征点个数是mpORBextractorLeft的两倍
    ORBextractor *mpORBextractorLeft;
    ORBextractor *mpIniORBextractor;

    // Initalization (only for monocular)
    // 单目初始器
    Initializer *mpInitializer;

    // System
    System *mpSystem;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame
    Frame mLastFrame;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;
};

} // namespace ORB_SLAM2

#endif // TRACKING_H
