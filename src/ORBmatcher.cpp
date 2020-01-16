#include "ORBmatcher.h"

#include <limits.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <stdint.h>

using namespace std;

namespace ORB_SLAM2
{
const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

// 初始化时假设F1和F2图像变化不大，在windowSize范围进行匹配，外部调用中windowSize = 100
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches = 0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = HISTO_LENGTH / 360.0f;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(), -1);

    for (size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if (level1 > 0)
            continue;

        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, level1, level1);

        if (vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for (vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1, d2);

            if (vMatchedDistance[i2] <= dist)
                continue;

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)函数步骤4
        if (bestDist <= TH_LOW)
        {
            if (bestDist < (float)bestDist2 * mfNNratio)
            {
                if (vnMatches21[bestIdx2] >= 0)
                {
                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    nmatches--;
                }
                vnMatches12[i1] = bestIdx2;
                vnMatches21[bestIdx2] = i1;
                vMatchedDistance[bestIdx2] = bestDist;
                nmatches++;

                if (mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;
                    if (rot < 0.0)
                        rot += 360.0f;
                    int bin = round(rot * factor);
                    if (bin == HISTO_LENGTH)
                        bin = 0;
                    assert(bin >= 0 && bin < HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                int idx1 = rotHist[i][j];
                if (vnMatches12[idx1] >= 0)
                {
                    vnMatches12[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }

    //Update prev matched
    for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
        if (vnMatches12[i1] >= 0)
            vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

// 取出直方图中值最大的三个index
void ORBmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++)
    {
        const int s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
} // namespace ORB_SLAM2