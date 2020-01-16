#include "System.h"
#include <thread>
#include <iostream> // std::cout, std::fixed
#include <iomanip>  // std::setprecision

using namespace std;
namespace ORB_SLAM2
{

System::System(const string &strSettingsFile)
{
    cout << endl
         << "ORB-MVO(F2F) start !!!" << std::endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }

    //Initialize the Tracking thread
    mpTracker = new Tracking(this, strSettingsFile);
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{
    return mpTracker->GrabImageMonocular(im, timestamp);
}

void System::SaveTrajectoryTUM(const string &filename)
{
}

} // namespace ORB_SLAM2
