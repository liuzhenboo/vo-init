#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include <opencv2/core/core.hpp>

#include <System.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << endl
             << "Usage: ./orb-mvo path_to_settings path_to_dataset" << endl;
        return 1;
    }

    // argv[2]为tum数据集地址
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[2]) + "/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // 创建ORB-MVO系统.
    ORB_SLAM2::System SLAM(argv[1]);

    // 统计跟踪时间
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

    // 主循环
    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++)
    {
        im = cv::imread(string(argv[2]) + "/" + vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (im.empty())
        {
            cerr << endl
                 << "Failed to load image at: "
                 << string(argv[2]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // 将图片传送给ORB-MVO系统
        SLAM.TrackMonocular(im, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
        // 记录花费时间
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // 等待读取下一张图片
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // 对每次循环跟踪花费的时间进行排序
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl
         << endl;
    cout << "耗时中位数: " << vTimesTrack[nImages / 2] << endl;
    cout << "耗时平均数: " << totaltime / nImages << endl;

    // 保存位姿
    SLAM.SaveTrajectoryTUM("Trajectory.txt");

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // 跳过前三行
    string s0;
    getline(f, s0);
    getline(f, s0);
    getline(f, s0);

    while (!f.eof())
    {
        string s;
        getline(f, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB); // rgb/xxx.png
        }
    }
}
