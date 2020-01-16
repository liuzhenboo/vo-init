#include "Frame.h"
#include "ORBmatcher.h"
#include <thread>
using namespace std;

namespace ORB_SLAM2
{
long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{
}
// 复制构造函数, mLastFrame = Frame(mCurrentFrame)
Frame::Frame(const Frame &frame)
    : mpORBextractorLeft(frame.mpORBextractorLeft),
      mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()), N(frame.N),
      mvKeys(frame.mvKeys), mvKeysUn(frame.mvKeysUn), mDescriptors(frame.mDescriptors.clone()),
      mnId(frame.mnId),
      mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor), mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for (int i = 0; i < FRAME_GRID_COLS; i++)
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j] = frame.mGrid[i][j];

    if (!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

// 单目初始化
// mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mK, mDistCoef);
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *extractor, cv::Mat &K, cv::Mat &distCoef)
    : mpORBextractorLeft(extractor), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone())
{
    // ORB金字塔参数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB特征提取函数
    ExtractORB(0, imGray);

    // 特征点的数量
    N = mvKeys.size();

    if (mvKeys.empty())
        return;

    // 调用OpenCV的矫正函数矫正提取的ORB特征点
    UndistortKeyPoints();

    // 仅仅在第一帧的时候执行
    if (mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    AssignFeaturesToGrid();
}

// 将特征点分配到FRAME_GRID_COLS*FRAME_GRID_ROWS个格子中去
void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(nReserve);

    // 在mGrid中记录了各特征点
    for (int i = 0; i < N; i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
}

/**
 * @brief Set the camera pose.
 * 
 * 设置相机姿态，随后会调用 UpdatePoseMatrices() 来改变mRcw,mRwc等变量的值
 * @param Tcw Transformation from world to camera
 */
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

cv::Mat Frame::GetPose_r()
{
    return mRwc;
}

cv::Mat Frame::GetPose_t()
{
    return mtwc;
}

/**
 * @brief Computes rotation, translation and camera center matrices from the camera pose.
 *
 * 根据Tcw计算mRcw、mtcw和mRwc、mOw
 */
void Frame::UpdatePoseMatrices()
{
    // mtcw, 即相机坐标系下相机坐标系到世界坐标系间的向量, 向量方向由相机坐标系指向世界坐标系
    // mtwc, 即世界坐标系下世界坐标系到相机坐标系间的向量, 向量方向由世界坐标系指向相机坐标系
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mtcw = mTcw.rowRange(0, 3).col(3);
    mRwc = mRcw.t();
    mtwc = -mRcw.t() * mtcw;
}

/**
 * @brief 找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点
 * @param x        图像坐标u
 * @param y        图像坐标v
 * @param r        边长
 * @param minLevel 最小尺度
 * @param maxLevel 最大尺度
 * @return         满足条件的特征点的序号
 */
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    const int nMinCellY = max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if (vCell.empty())
                continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if (bCheckLevels)
                {
                    if (kpUn.octave < minLevel)
                        continue;
                    if (maxLevel >= 0)
                        if (kpUn.octave > maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}

// 调用OpenCV的矫正函数矫正orb提取的特征点
void Frame::UndistortKeyPoints()
{
    // 如果没有图像是矫正过的，没有失真
    if (mDistCoef.at<float>(0) == 0.0)
    {
        mvKeysUn = mvKeys;
        return;
    }

    // Fill matrix with points
    // N为提取的特征点数量，将N个特征点保存在N*2的mat中
    cv::Mat mat(N, 2, CV_32F);
    for (int i = 0; i < N; i++)
    {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    // 调整mat的通道为2，矩阵的行列形状不变
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK); // 用cv的函数进行失真校正
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    // 存储校正后的特征点
    mvKeysUn.resize(N);
    for (int i = 0; i < N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if (mDistCoef.at<float>(0) != 0.0)
    {
        // 矫正前四个边界点：(0,0) (cols,0) (0,rows) (cols,rows)
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0; //左上
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = imLeft.cols; //右上
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0; //左下
        mat.at<float>(2, 1) = imLeft.rows;
        mat.at<float>(3, 0) = imLeft.cols; //右下
        mat.at<float>(3, 1) = imLeft.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0)); //左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0)); //右上和右下横坐标最大的
        mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1)); //左上和右上纵坐标最小的
        mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1)); //左下和右下纵坐标最小的
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

} // namespace ORB_SLAM2