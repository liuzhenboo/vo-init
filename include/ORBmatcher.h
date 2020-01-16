
#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Frame.h"

namespace ORB_SLAM2
{

class ORBmatcher
{
public:
    ORBmatcher(float nnratio = 0.6, bool checkOri = true);

    // Matching for the Map Initialization (only used in the monocular case)
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize = 10);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    bool mbCheckOrientation;
};

} // namespace ORB_SLAM2

#endif // ORBMATCHER_H
