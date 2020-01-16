#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "Converter.h"
#include "Initializer.h"

#include <iostream>
#include <cmath>

using namespace std;

// 程序中变量名的第一个字母如果为"m"则表示为类中的成员变量，member
// 第一个、第二个字母:
// "p"表示指针数据类型
// "n"表示int类型
// "b"表示bool类型
// "s"表示set类型
// "v"表示vector数据类型
// 'l'表示list数据类型
// "KF"表示KeyPoint数据类型

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, const string &strSettingPath) : mState(NO_IMAGES_YET), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys)
{
    // Load camera parameters from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    // 相机的内参mK
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];

    float cy = fSettings["Camera.cy"];

    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    // 图像矫正系数mDistCoef
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    cout << endl
         << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if (DistCoef.rows == 5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;

    // 1:RGB 0:BGR
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    // 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    // tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    // 在单目初始化的时候，会用mpIniORBextractor来作为特征点提取器
    mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl
         << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
}

// 输入左目RGB或RGBA图像
// 1、将图像转为mImGray并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    // 步骤1：将RGB或RGBA图像转为灰度图像
    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    // 步骤2：构造Frame
    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET) // 没有成功初始化的前一个状态就是NO_IMAGES_YET
        mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mK, mDistCoef);
    else
        mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mK, mDistCoef);

    // 步骤3：跟踪
    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    // SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
    // 如果图像第一次运行，则为NO_IMAGE_YET状态
    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    if (mState == NOT_INITIALIZED)
    {
        if (MonocularInitialization())
            mState = OK;
        else
        {
            return;
        }
    }
    else
    {
        cout << "初始化完成！！！" << endl;
    }
}

/**
 * @brief 单目初始化
 *
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态
 */
int Tracking::MonocularInitialization()
{
    // 如果单目初始器还没有被创建，则创建单目初始器
    if (!mpInitializer)
    {
        // Set Reference Frame
        // 单目初始帧提取的特征点数必须大于100，否则放弃该帧图像
        if (mCurrentFrame.mvKeys.size() > 100)
        {
            // 步骤1：得到用于初始化的第一帧，初始化需要两帧
            mInitialFrame = Frame(mCurrentFrame);
            // 记录最近的一帧
            mLastFrame = Frame(mCurrentFrame);
            // mvbPrevMatched最大的情况就是所有特征点都被跟踪上
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            cout << "构造初始化器。。。" << endl;
            // 由当前帧构造初始器 sigma:1.0 iterations:200
            mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
        }
        else
        {

            cout << "特征点小于100，不能构造初始化器！！！" << endl;
        }

        return 0;
    }
    else
    {
        // Try to initialize
        // 步骤2：如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
        // 如果当前帧特征点太少，重新构造初始器
        // 因此只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
        if ((int)mCurrentFrame.mvKeys.size() <= 100)
        {
            delete mpInitializer;
            cout << "当前帧特征点数量小于100，删除初始化器！！！" << endl;
            mpInitializer = static_cast<Initializer *>(NULL);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return 0;
        }

        // Find correspondences
        // 步骤3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
        // mvbPrevMatched为前一帧的特征点
        // mvIniMatches存储mInitialFrame,mCurrentFrame之间匹配的特征点号
        ORBmatcher matcher(0.9, true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        // 步骤4：如果初始化的两帧之间的匹配点太少，重新初始化
        if (nmatches < 100)
        {
            delete mpInitializer;
            cout << "初始化两帧之间的匹配点太少，删除初始化器，重新初始化！！！" << endl;
            mpInitializer = static_cast<Initializer *>(NULL);
            return 0;
        }

        cv::Mat Rcw;                 // Current Camera Rotation
        cv::Mat tcw;                 // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        // 步骤5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            // 步骤6：删除那些无法进行三角化的匹配点
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                }
            }

            // Set Frame Poses
            // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到该帧的坐标系变换矩阵
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);
            cout << "旋转：" << endl
                 << Converter::toMatrix3d(mCurrentFrame.GetPose_r()) << endl;
            cout << "平移:" << endl
                 << Converter::toVector3d(mCurrentFrame.GetPose_t()) << endl;
            return 1;
        }
        else
        {
            delete mpInitializer;
            cout << "H模型与F模型失败!!!" << endl;
            mpInitializer = static_cast<Initializer *>(NULL);
            return 0;
        }
    }
}

} // namespace ORB_SLAM2
