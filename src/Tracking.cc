/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "Tracking.h"
#include <iomanip>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"
#include "Kernels/CudaUtils.h"
#include "Stats/TrackingStats.h"
#include "CalculationLogger.h"
#include <opencv2/video/tracking.hpp>

#include <iostream>

#include "Kernels/KernelController.h"
#include <mutex>
#include <random>
#include <deque>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstdlib> // For std::exit


using namespace std;


namespace ORB_SLAM3
{

ORBextractor *Tracking::mpORBextractorLeft, *Tracking::mpORBextractorRight;

void Tracking::signalHandler(int signum) {
    // Release resources here
    KernelController::shutdownKernels();
    if (mpORBextractorLeft) {
        delete mpORBextractorLeft;
    }
    if (mpORBextractorRight) {
        delete mpORBextractorRight;
    }
    
    // Exit the program
    std::exit(signum);
}

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL))
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("tracking.construct sensor=%d sequence=%s", sensor, _nameSeq.c_str());

    // FastTrack's handler calls std::exit (no Shutdown, no map save). keep_host_sigint leaves SIGINT to the host.
    if(sbInstallSigintHandler)
        std::signal(SIGINT, Tracking::signalHandler);
    else
        FT_CALC_LOG("tracking.sigint handler=host");

    // Load camera parameters from settings file
    mSettingsPath = strSettingPath;
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            FT_CALC_LOG("tracking.config.error section=camera");
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            FT_CALC_LOG("tracking.config.error section=orb");
        }

        bool b_parse_imu = true;
        if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO || sensor==System::IMU_RGBD)
        {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if(!b_parse_imu)
            {
                FT_CALC_LOG("tracking.config.error section=imu");
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            FT_CALC_LOG("tracking.config.error format=invalid");
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    initID = 0; lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    FT_CALC_LOG("tracking.cameras.count=%llu", static_cast<unsigned long long>(vpCams.size()));
    for(GeometricCamera* pCam : vpCams)
    {
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            FT_CALC_LOG("tracking.camera id=%llu type=pinhole",
                        static_cast<unsigned long long>(pCam->GetId()));
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            FT_CALC_LOG("tracking.camera id=%llu type=fisheye",
                        static_cast<unsigned long long>(pCam->GetId()));
        }
        else
        {
            FT_CALC_LOG("tracking.camera id=%llu type=unknown",
                        static_cast<unsigned long long>(pCam->GetId()));
        }
    }

    FT_CALC_LOG("tracking.kernels.initialize");
    KernelController::initializeKernels();


#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    auto logStats = [](const char *metric, double mean, double deviation) {
        FT_CALC_LOG("tracking.stats metric=%s mean_ms=%f deviation=%f", metric, mean, deviation);
    };
    //Report the mean and std of each one
    FT_CALC_LOG("tracking.stats.begin unit=ms");
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    FT_CALC_LOG("tracking.stats.opencv version=%s", CV_VERSION);
    f << "OpenCV version: " << CV_VERSION << endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        logStats("stereo_rectification", average, deviation);
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        logStats("image_resize", average, deviation);
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    logStats("orb_extraction", average, deviation);
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        logStats("stereo_matching", average, deviation);
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        logStats("imu_preintegration", average, deviation);
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    logStats("pose_prediction", average, deviation);
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    logStats("local_map_tracking", average, deviation);
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    logStats("new_keyframe_decision", average, deviation);
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    logStats("tracking_total", average, deviation);
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    FT_CALC_LOG("tracking.stats.section=local_mapping");
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    logStats("keyframe_insertion", average, deviation);
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    logStats("mappoint_culling", average, deviation);
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    logStats("mappoint_creation", average, deviation);
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    logStats("local_bundle_adjustment", average, deviation);
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    logStats("keyframe_culling", average, deviation);
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    logStats("local_mapping_total", average, deviation);
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    FT_CALC_LOG("tracking.stats.section=lba_complexity");
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    logStats("lba_edges", average, deviation);
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    logStats("lba_keyframes_optimized", average, deviation);
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    logStats("lba_keyframes_fixed", average, deviation);
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    logStats("lba_mappoints", average, deviation);
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    FT_CALC_LOG("tracking.stats lba_executions=%d lba_aborts=%d",
                mpLocalMapper->nLBA_exec, mpLocalMapper->nLBA_abort);
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    FT_CALC_LOG("tracking.stats.section=map_complexity keyframes=%llu mappoints=%llu",
                static_cast<unsigned long long>(mpAtlas->GetAllKeyFrames().size()),
                static_cast<unsigned long long>(mpAtlas->GetAllMapPoints().size()));
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    FT_CALC_LOG("tracking.stats.section=place_recognition");
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    logStats("database_query", average, deviation);
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    logStats("se3_estimation", average, deviation);
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    logStats("place_recognition_total", average, deviation);

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    FT_CALC_LOG("tracking.stats.section=loop_closing");
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    logStats("loop_fusion", average, deviation);
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    logStats("essential_graph", average, deviation);
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    logStats("loop_closing_total", average, deviation);

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    FT_CALC_LOG("tracking.stats loop_executions=%d", mpLoopClosing->nLoop);
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    logStats("loop_keyframes", average, deviation);

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    FT_CALC_LOG("tracking.stats.section=map_merging");
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    logStats("merge_maps", average, deviation);
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    logStats("welding_bundle_adjustment", average, deviation);
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    logStats("merge_essential_graph", average, deviation);
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    logStats("map_merging_total", average, deviation);

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    FT_CALC_LOG("tracking.stats map_merge_executions=%d", mpLoopClosing->nMerges);
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    logStats("merge_keyframes", average, deviation);
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    logStats("merge_mappoints", average, deviation);

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    FT_CALC_LOG("tracking.stats.section=full_gba");
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    logStats("global_bundle_adjustment", average, deviation);
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    logStats("map_update", average, deviation);
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    logStats("full_gba_total", average, deviation);

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    FT_CALC_LOG("tracking.stats full_gba_executions=%d", mpLoopClosing->nFGBA_exec);
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    FT_CALC_LOG("tracking.stats full_gba_aborts=%d", mpLoopClosing->nFGBA_abort);
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    logStats("gba_keyframes", average, deviation);
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    logStats("gba_mappoints", average, deviation);

    f.close();

}

#endif

Tracking::~Tracking()
{
    //f_track_stats.close();
}

void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if(settings->needToUndistort()){
        mDistCoef = settings->camera1DistortionCoef();
    }
    else{
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    //TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    if((mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD) &&
        settings->cameraType() == Settings::KannalaBrandt){
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr = settings->Tlr();

        mpFrameDrawer->both = true;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD ){
        mbf = settings->bf();
        mThDepth = settings->b() * settings->thDepth();
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD){
        mDepthMapFactor = settings->depthMapFactor();
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    //ORB parameters
    int nFeatures = settings->nFeatures();
    int nLevels = settings->nLevels();
    int fIniThFAST = settings->initThFAST();
    int fMinThFAST = settings->minThFAST();
    float fScaleFactor = settings->scaleFactor();
    bool cameraIsFisheye = (settings->cameraType() == Settings::KannalaBrandt);

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST, settings->newImSize().width, settings->newImSize().height);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO) {
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST, settings->newImSize().width, settings->newImSize().height);
        CudaUtils::loadSetting(nFeatures, nLevels, false, fScaleFactor, settings->newImSize().width, settings->newImSize().height, cameraIsFisheye);
        KernelController::cuda_utils_loaded_setting = true;
    }
    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR){
    int iniFactor = 5;  // Init.IniFeaturesFactor (research, default 5 = upstream)
    {
        cv::FileStorage fF(mSettingsPath, cv::FileStorage::READ);
        if(fF.isOpened()) { cv::FileNode n = fF["Init.IniFeaturesFactor"]; if(!n.empty() && n.isInt() && (int)n >= 1 && (int)n <= 10) iniFactor = (int)n; }
    }
        std::cerr << "FastTrack init extractor: " << iniFactor << " x " << nFeatures << " features" << std::endl;
        mpIniORBextractor = new ORBextractor(iniFactor*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST, settings->newImSize().width, settings->newImSize().height);
        CudaUtils::loadSetting(iniFactor*nFeatures, nLevels, true, fScaleFactor, settings->newImSize().width, settings->newImSize().height, cameraIsFisheye);
        KernelController::cuda_utils_loaded_setting = true;
    }
    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    mImuPer = 0.001; //1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
    // The Settings path never assigned this (only the cv::FileStorage branch of the constructor did), leaving the
    // post-relocalization IMU windows on an indeterminate value. mMaxFrames = fps is set above.
    mnFramesToResetIMU = mMaxFrames;
    FT_CALC_LOG("tracking.config.frames_to_reset_imu value=%d", mnFramesToResetIMU);
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);

    // ZUPT (Zero Velocity Update) parameters from settings file
    mbZuptEnabled = false;
    mZuptWindow = (int)mImuFreq;
    mZuptGyroVarTh = 1e-3;
    mZuptAccelVarTh = 1e-1;
    mZuptHystEnter = 3;
    mZuptHystExit = 3;
    mZuptExitRatio = 3.0;
    mZuptGyroMeanGate = 0.03;
    mZuptAccelNormGate = 0.3;
    mZuptVelGate = 0.05;
    mZuptVelExitTh = 0.15;
    mZuptVelHyst = 3;
    mbZuptHoldRotation = false;
    mbZuptStrongRestEnabled = false;
    mZuptStrongGyroVarTh = 1e-5;
    mZuptStrongAccelVarTh = 2.5e-4;
    mZuptStrongMinS = 2.0;
    mZuptStrongSince = -1.0;
    mbZuptStrongRest = false;
    mZuptVelExitCount = 0;
    mZuptVelEma = 0.0;
    mbZuptActive = false;
    mZuptEnterCount = 0;
    mZuptExitCount = 0;
    mlZuptImuWindow.clear();

    cv::FileStorage fZupt(mSettingsPath, cv::FileStorage::READ);
    if(fZupt.isOpened())
    {
        cv::FileNode node = fZupt["IMU.ZUPT.Enabled"];
        mbZuptEnabled = !node.empty() && node.isInt() && static_cast<int>(node) != 0;
        node = fZupt["IMU.ZUPT.Window"];
        if(!node.empty() && node.isInt() && node.operator int() > 0)
            mZuptWindow = node.operator int();
        node = fZupt["IMU.ZUPT.GyroVarTh"];
        if(!node.empty() && node.isReal() && node.real() > 0)
            mZuptGyroVarTh = node.real();
        node = fZupt["IMU.ZUPT.AccelVarTh"];
        if(!node.empty() && node.isReal() && node.real() > 0)
            mZuptAccelVarTh = node.real();
        node = fZupt["IMU.ZUPT.HystEnter"];
        if(!node.empty() && node.isInt() && node.operator int() > 0)
            mZuptHystEnter = node.operator int();
        node = fZupt["IMU.ZUPT.HystExit"];
        if(!node.empty() && node.isInt() && node.operator int() > 0)
            mZuptHystExit = node.operator int();
        node = fZupt["IMU.ZUPT.ExitRatio"];
        if(!node.empty() && node.isReal() && node.real() >= 1.0)
            mZuptExitRatio = node.real();
        node = fZupt["IMU.ZUPT.GyroMeanGate"];
        if(!node.empty() && node.isReal() && node.real() > 0)
            mZuptGyroMeanGate = node.real();
        node = fZupt["IMU.ZUPT.AccelNormGate"];
        if(!node.empty() && node.isReal() && node.real() > 0)
            mZuptAccelNormGate = node.real();
        node = fZupt["IMU.ZUPT.VelGate"];
        if(!node.empty() && node.isReal() && node.real() > 0)
            mZuptVelGate = node.real();
        node = fZupt["IMU.ZUPT.VelExitTh"];
        if(!node.empty() && node.isReal() && node.real() > 0)
            mZuptVelExitTh = node.real();
        node = fZupt["IMU.ZUPT.VelHyst"];
        if(!node.empty() && node.isInt() && node.operator int() > 0)
            mZuptVelHyst = node.operator int();
        // At true rest the rotation is zero; integrating (gyro - estimated bias) only adds the bias error,
        // so the held output keeps the attitude as well (2026-10-21: attitude drifted 0.2-2 deg/s at rest).
        node = fZupt["IMU.ZUPT.HoldRotation"];
        mbZuptHoldRotation = !node.empty() && node.isInt() && static_cast<int>(node) != 0;
        node = fZupt["IMU.ZUPT.StrongRest.Enabled"];
        mbZuptStrongRestEnabled = !node.empty() && node.isInt() && static_cast<int>(node) != 0;
        node = fZupt["Init.Metric.Enabled"];
        mbMetricInitEnabled = !node.empty() && node.isInt() && static_cast<int>(node) != 0;
        node = fZupt["Init.Metric.PadCameraHeight"]; if(!node.empty() && node.isReal() && node.real() >= 0) mMetricPadCamHeight = node.real();
        node = fZupt["Init.Metric.MinHeight"]; if(!node.empty() && node.isReal() && node.real() > 0) mMetricMinHeight = node.real();
        node = fZupt["Metric.DeferClimbRate"]; if(!node.empty() && node.isReal() && node.real() >= 0) mDeferClimbRate = node.real();
        node = fZupt["Metric.ClimbMinParallaxDeg"]; if(!node.empty() && node.isReal() && node.real() >= 0 && node.real() < 30) mClimbMinParallaxDeg = node.real();
        node = fZupt["Metric.ClimbParallaxRate"]; if(!node.empty() && node.isReal() && node.real() > 0) mClimbParallaxRate = node.real();
        node = fZupt["Metric.BaroVz.Enabled"]; if(!node.empty() && node.isInt()) mbBaroVz = node.operator int() != 0;
        node = fZupt["Metric.ClimbFullTailLink"]; if(!node.empty() && node.isInt()) mbClimbFullTail = node.operator int() != 0;
        node = fZupt["Metric.BaroVz.Sigma"]; if(!node.empty() && node.isReal() && node.real() > 0) mBaroVzSigma = node.real();
        node = fZupt["Metric.BaroVz.Window"]; if(!node.empty() && node.isReal() && node.real() >= 0.2 && node.real() <= 10) mBaroVzWindow = node.real();
        node = fZupt["Metric.AccBiasObs.Enabled"]; if(!node.empty() && node.isInt()) mbAccBiasObs = node.operator int() != 0;  // C7
        node = fZupt["Metric.VertKF.Enabled"]; if(!node.empty() && node.isInt()) mbVertKF = node.operator int() != 0;  // C7c
        node = fZupt["Metric.TiltCorr.Enabled"]; if(!node.empty() && node.isInt()) mbTiltCorr = node.operator int() != 0;  // C8
        node = fZupt["Metric.TiltCorr.MinHeight"]; if(!node.empty() && node.isReal() && node.real() >= 0) mTiltCorrMinHeight = node.real();
        node = fZupt["Metric.TiltCorr.Deadband"]; if(!node.empty() && node.isReal() && node.real() >= 0) mTiltCorrDeadband = node.real();
        node = fZupt["Metric.TiltCorr.Gain"]; if(!node.empty() && node.isReal() && node.real() > 0 && node.real() <= 1) mTiltCorrGain = node.real();
        node = fZupt["Metric.TiltCorr.MaxStep"]; if(!node.empty() && node.isReal() && node.real() > 0) mTiltCorrMaxStep = node.real();
        node = fZupt["Metric.TiltCorr.MaxTotal"]; if(!node.empty() && node.isReal() && node.real() > 0) mTiltCorrMaxTotal = node.real();
        node = fZupt["Metric.TiltCorr.MinInlier"]; if(!node.empty() && node.isReal() && node.real() > 0) mTiltCorrMinInlier = node.real();
        node = fZupt["Metric.TiltCorr.Window"]; if(!node.empty() && node.isInt() && (int)node >= 3) mTiltCorrWindow = (int)node;
        node = fZupt["Metric.TiltCorr.Period"]; if(!node.empty() && node.isInt() && (int)node >= 1) mTiltCorrPeriod = (int)node;
        node = fZupt["Metric.TiltCorr.Source"]; if(!node.empty() && node.isInt() && ((int)node == 0 || (int)node == 1)) mTiltCorrSource = (int)node;  // C8f
        node = fZupt["Metric.TiltCorr.TurnRate"]; if(!node.empty() && node.isReal() && node.real() > 0) mTiltCorrTurnRate = node.real();  // C8f v2
        node = fZupt["Metric.TiltCorr.TurnHold"]; if(!node.empty() && node.isReal() && node.real() >= 0) mTiltCorrTurnHold = node.real();
        node = fZupt["Metric.TiltCorr.SnapMax"]; if(!node.empty() && node.isReal() && node.real() >= 0 && node.real() <= 30) mTiltCorrSnapMax = node.real();  // C8s
        node = fZupt["Metric.TiltCorr.SnapMinObs"]; if(!node.empty() && node.isInt() && (int)node >= 1) mTiltCorrSnapMinObs = (int)node;
        node = fZupt["Metric.TiltCorr.SnapResetDeg"]; if(!node.empty() && node.isReal() && node.real() >= 0 && node.real() <= 90) mTiltCorrSnapResetDeg = node.real();  // C8r
        std::cerr << "FastTrack C8 settings: enabled=" << mbTiltCorr << " minHeight=" << mTiltCorrMinHeight << " deadband="
                  << mTiltCorrDeadband << " gain=" << mTiltCorrGain << " maxStep=" << mTiltCorrMaxStep << " maxTotal="
                  << mTiltCorrMaxTotal << " window=" << mTiltCorrWindow << " period=" << mTiltCorrPeriod << " source=" << mTiltCorrSource << " turnRate=" << mTiltCorrTurnRate << " turnHold=" << mTiltCorrTurnHold << " snapMax=" << mTiltCorrSnapMax
                  << " snapMinObs=" << mTiltCorrSnapMinObs << " snapResetDeg=" << mTiltCorrSnapResetDeg << std::endl;
        node = fZupt["Metric.VertKF.AccNoise"]; if(!node.empty() && node.isReal() && node.real() > 0) mVkfAccNoise = node.real();
        node = fZupt["Metric.VertKF.UseFcAttitude"]; if(!node.empty() && node.isInt()) mbVkfUseFc = node.operator int() != 0;
        node = fZupt["Metric.VertKF.BiasRW"]; if(!node.empty() && node.isReal() && node.real() > 0) mVkfBiasRW = node.real();
        node = fZupt["Metric.VertKF.BaroNoise"]; if(!node.empty() && node.isReal() && node.real() > 0) mVkfBaroNoise = node.real();
        node = fZupt["Metric.VertKF.MinHeight"]; if(!node.empty() && node.isReal() && node.real() >= 0) mVkfMinHeight = node.real();
        node = fZupt["Metric.VertKF.MaxBias"]; if(!node.empty() && node.isReal() && node.real() > 0) mVkfMaxBias = node.real();
        node = fZupt["Metric.VertKF.MaxStep"]; if(!node.empty() && node.isReal() && node.real() > 0) mVkfMaxStep = node.real();
        node = fZupt["Metric.VertKF.MaxSd"]; if(!node.empty() && node.isReal() && node.real() > 0) mVkfMaxSd = node.real();
        node = fZupt["Metric.VertKF.ChainKFs"]; if(!node.empty() && node.isInt() && (int)node >= 1) mVkfChainKFs = (int)node;
        std::cerr << "FastTrack C7c settings: enabled=" << mbVertKF << " accNoise=" << mVkfAccNoise << " biasRW=" << mVkfBiasRW
                  << " baroNoise=" << mVkfBaroNoise << " maxSd=" << mVkfMaxSd << " maxStep=" << mVkfMaxStep
                  << " useFcAttitude=" << mbVkfUseFc << std::endl;
        node = fZupt["Metric.AccBiasObs.Window"]; if(!node.empty() && node.isReal() && node.real() >= 2 && node.real() <= 60) mAccBiasObsWindow = node.real();
        node = fZupt["Metric.AccBiasObs.Sigma"]; if(!node.empty() && node.isReal() && node.real() > 0) mAccBiasObsSigma = node.real();
        node = fZupt["Metric.AccBiasObs.MinKFs"]; if(!node.empty() && node.isInt() && (int)node >= 5) mAccBiasObsMinKFs = (int)node;
        node = fZupt["Metric.AccBiasObs.MaxStep"]; if(!node.empty() && node.isReal() && node.real() > 0) mAccBiasObsMaxStep = node.real();
        node = fZupt["Metric.AccBiasObs.MaxBias"]; if(!node.empty() && node.isReal() && node.real() > 0) mAccBiasObsMaxBias = node.real();
        node = fZupt["Metric.AccBiasObs.MinHeight"]; if(!node.empty() && node.isReal() && node.real() >= 0) mAccBiasObsMinHeight = node.real();
        node = fZupt["Metric.Cruise.Enabled"]; if(!node.empty() && node.isInt()) mbCruiseScale = node.operator int() != 0;
        node = fZupt["Metric.Cruise.Window"]; if(!node.empty() && node.isInt() && node.operator int() >= 3) mCruiseWindow = node.operator int();
        node = fZupt["Metric.Cruise.Period"]; if(!node.empty() && node.isInt() && node.operator int() >= 1) mCruisePeriod = node.operator int();
        node = fZupt["Metric.Cruise.Gain"]; if(!node.empty() && node.isReal() && node.real() > 0 && node.real() <= 1) mCruiseGain = node.real();
        node = fZupt["Metric.Cruise.Deadband"]; if(!node.empty() && node.isReal() && node.real() >= 0) mCruiseDeadband = node.real();
        node = fZupt["Metric.Cruise.MaxStep"]; if(!node.empty() && node.isReal() && node.real() > 0 && node.real() < 0.5) mCruiseMaxStep = node.real();
        node = fZupt["Metric.Cruise.MinHeight"]; if(!node.empty() && node.isReal() && node.real() > 0) mCruiseMinHeight = node.real();
        node = fZupt["Metric.Cruise.MaxTiltDeg"]; if(!node.empty() && node.isReal() && node.real() > 0) mCruiseMaxTilt = node.real();
        node = fZupt["Metric.Cruise.MinInlier"]; if(!node.empty() && node.isReal() && node.real() >= 0 && node.real() <= 1) mCruiseMinInlier = node.real();
        node = fZupt["Features.MaxRadiusPx"]; if(!node.empty() && node.isReal() && node.real() >= 0) Frame::msMaxKeypointRadius = (float)node.real();
        node = fZupt["Init.MinAxisAngleDeg"]; if(!node.empty() && node.isReal() && node.real() >= 0 && node.real() < 90) mInitMinAxisAngleDeg = node.real();
        node = fZupt["Klt.Enabled"];
        mbKltEnabled = !node.empty() && node.isInt() && static_cast<int>(node) != 0;
        node = fZupt["Klt.WinSize"]; if(!node.empty() && node.isInt() && (int)node >= 5) mKltWin = (int)node;
        node = fZupt["Klt.Levels"]; if(!node.empty() && node.isInt() && (int)node >= 0) mKltLevels = (int)node;
        node = fZupt["Klt.MaxFbError"]; if(!node.empty() && node.isReal() && node.real() > 0) mKltMaxFb = node.real();
        node = fZupt["Klt.Radius"]; if(!node.empty() && node.isReal() && node.real() > 0) mKltRadius = node.real();
        node = fZupt["Klt.MaxHamming"]; if(!node.empty() && node.isInt() && (int)node > 0) mKltMaxHamming = (int)node;
        node = fZupt["Klt.Persistent"]; if(!node.empty() && node.isInt()) mbKltPersistent = (int)node != 0;
        node = fZupt["Klt.MaxAppend"]; if(!node.empty() && node.isInt() && (int)node >= 0) mKltMaxAppend = (int)node;
        node = fZupt["IMU.RestObs.Enabled"];
        mbRestObsEnabled = !node.empty() && node.isInt() && static_cast<int>(node) != 0;
        node = fZupt["IMU.RestObs.GyroSigma"];
        if(!node.empty() && node.isReal() && node.real() > 0) mRestObsGyroSigma = node.real();
        node = fZupt["IMU.RestObs.AccSigma"];
        if(!node.empty() && node.isReal() && node.real() > 0) mRestObsAccSigma = node.real();
        node = fZupt["IMU.ZUPT.StrongRest.GyroVarTh"];
        if(!node.empty() && node.isReal() && node.real() > 0)
            mZuptStrongGyroVarTh = node.real();
        node = fZupt["IMU.ZUPT.StrongRest.AccelVarTh"];
        if(!node.empty() && node.isReal() && node.real() > 0)
            mZuptStrongAccelVarTh = node.real();
        node = fZupt["IMU.ZUPT.StrongRest.MinS"];
        if(!node.empty() && node.isReal() && node.real() >= 0)
            mZuptStrongMinS = node.real();
    }

    if(mbZuptEnabled)
    {
        cout << "ZUPT enabled (window=" << mZuptWindow << ", gyroVarTh=" << mZuptGyroVarTh
             << ", accelVarTh=" << mZuptAccelVarTh << ", hystEnter=" << mZuptHystEnter
             << ", hystExit=" << mZuptHystExit << ", exitRatio=" << mZuptExitRatio
             << ", gyroMeanGate=" << mZuptGyroMeanGate << ", accelNormGate=" << mZuptAccelNormGate
             << ", velGate=" << mZuptVelGate << ", velExitTh=" << mZuptVelExitTh
             << ", velHyst=" << mZuptVelHyst << ", holdRotation=" << (mbZuptHoldRotation ? 1 : 0)
             << ", strongRest=" << (mbZuptStrongRestEnabled ? 1 : 0) << " (gyroVar<" << mZuptStrongGyroVarTh
             << ", accelVar<" << mZuptStrongAccelVarTh << ", " << mZuptStrongMinS << " s))" << endl;
    }
    else
    {
        cout << "ZUPT disabled (set IMU.ZUPT.Enabled=1 to enable)" << endl;
    }

    if(mbZuptEnabled)
    {
        cout << endl << "ZUPT loaded from settings file" << endl;
    }

    // External absolute reference (compass heading + barometric altitude) parameters.
    mbExtRefHasYaw = false;
    mbExtRefHasAlt = false;
    mExtRefYawDeg = 0.0;
    mExtRefAltM = 0.0;
    mExtRefLatched = false;
    mExtRefLatchInitKF = 0;
    mExtMagYawOffsetRad = 0.0;
    mExtBaroAltOffsetM = 0.0;
    mExtMagYawSigmaDeg = 8.0;   // default trust: 8 deg heading prior
    mExtBaroAltSigmaM = 3.0;    // default trust: 3 m altitude prior
    mbExtRefPrinted = false;
    // nf1 (2026-10-15): the wrapper's heading is a compass heading (clockwise from north) while the
    // SLAM world yaw is counter-clockwise about z, so the prior target is offset - heading.
    // 0 reproduces the previous target heading + offset, which was mirrored: after a turn by d the
    // prior pulled the yaw by 2d (Oct 9 2026 flight: ORB yaw + compass constant, SD 2.3 deg).
    mbExtMagYawClockwise = true;

    cv::FileStorage fExt(mSettingsPath, cv::FileStorage::READ);
    if(fExt.isOpened())
    {
        cv::FileNode node = fExt["IMU.EXT.MagYawSigmaDeg"];
        if(!node.empty() && node.isReal() && node.real() > 0)
            mExtMagYawSigmaDeg = node.real();
        node = fExt["IMU.EXT.BaroAltSigmaM"];
        if(!node.empty() && node.isReal() && node.real() > 0)
            mExtBaroAltSigmaM = node.real();
        node = fExt["IMU.EXT.MagYawClockwise"];
        if(!node.empty() && node.isInt())
            mbExtMagYawClockwise = node.operator int() != 0;
    }

    cout << "External ref (compass+baro) ready: magYawSigmaDeg=" << mExtMagYawSigmaDeg
         << ", baroAltSigmaM=" << mExtBaroAltSigmaM
         << ", magYawClockwise=" << (mbExtMagYawClockwise ? 1 : 0)
         << " (waiting for wrapper data; enable via launch param mag_baro_enable=true)" << endl;
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    FT_CALC_SCOPE();
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.fx type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.fy type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.cx type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.cy type=real");
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.k1 type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.k2 type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.p1 type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.p2 type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        if(mImageScale != 1.f)
        {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        FT_CALC_LOG("tracking.camera.pinhole scale=%f fx=%f fy=%f cx=%f cy=%f k1=%f k2=%f p1=%f p2=%f k3=%f",
                    mImageScale, fx, fy, cx, cy, mDistCoef.at<float>(0), mDistCoef.at<float>(1),
                    mDistCoef.at<float>(2), mDistCoef.at<float>(3),
                    mDistCoef.rows == 5 ? mDistCoef.at<float>(4) : 0.0f);

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

        mK_.setIdentity();
        mK_(0,0) = fx;
        mK_(1,1) = fy;
        mK_(0,2) = cx;
        mK_(1,2) = cy;
    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.fx type=real");
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.fy type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.cx type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.cy type=real");
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.k1 type=real");
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.k2 type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.k3 type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.k4 type=real");
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(!b_miss_params)
        {
            if(mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            FT_CALC_LOG("tracking.camera.fisheye scale=%f fx=%f fy=%f cx=%f cy=%f k1=%f k2=%f k3=%f k4=%f",
                        mImageScale, fx, fy, cx, cy, k1, k2, k3, k4);

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;

            mK_.setIdentity();
            mK_(0,0) = fx;
            mK_(1,1) = fy;
            mK_(0,2) = cx;
            mK_(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                FT_CALC_LOG("tracking.config.missing name=Camera2.fx type=real");
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                FT_CALC_LOG("tracking.config.missing name=Camera2.fy type=real");
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                FT_CALC_LOG("tracking.config.missing name=Camera2.cx type=real");
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                FT_CALC_LOG("tracking.config.missing name=Camera2.cy type=real");
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                FT_CALC_LOG("tracking.config.missing name=Camera2.k1 type=real");
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                FT_CALC_LOG("tracking.config.missing name=Camera2.k2 type=real");
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                FT_CALC_LOG("tracking.config.missing name=Camera2.k3 type=real");
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                FT_CALC_LOG("tracking.config.missing name=Camera2.k4 type=real");
                b_miss_params = true;
            }


            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                FT_CALC_LOG("tracking.config.warning name=Camera.lappingBegin invalid=1");
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                FT_CALC_LOG("tracking.config.warning name=Camera.lappingEnd invalid=1");
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                FT_CALC_LOG("tracking.config.warning name=Camera2.lappingBegin invalid=1");
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                FT_CALC_LOG("tracking.config.warning name=Camera2.lappingEnd invalid=1");
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if(!node.empty())
            {
                cvTlr = node.mat();
                if(cvTlr.rows != 3 || cvTlr.cols != 4)
                {
                    FT_CALC_LOG("tracking.config.missing name=Tlr type=3x4_matrix");
                    b_miss_params = true;
                }
            }
            else
            {
                FT_CALC_LOG("tracking.config.missing name=Tlr type=matrix");
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                if(mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                FT_CALC_LOG("tracking.camera.fisheye_right scale=%f fx=%f fy=%f cx=%f cy=%f k1=%f k2=%f k3=%f k4=%f left_lap_begin=%d left_lap_end=%d right_lap_begin=%d right_lap_end=%d tlr_rows=%d tlr_cols=%d",
                            mImageScale, fx, fy, cx, cy, k1, k2, k3, k4, leftLappingBegin,
                            leftLappingEnd, rightLappingBegin, rightLappingEnd, cvTlr.rows, cvTlr.cols);
            }
        }

        if(b_miss_params)
        {
            return false;
        }

    }
    else
    {
        FT_CALC_LOG("tracking.config.error camera_sensor=unsupported");
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD )
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
            if(mImageScale != 1.f)
            {
                mbf *= mImageScale;
            }
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=Camera.bf type=real");
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    FT_CALC_LOG("tracking.config.camera fps=%f", fps);


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        FT_CALC_LOG("tracking.config.camera color_order=RGB");
    else
        FT_CALC_LOG("tracking.config.camera color_order=BGR");

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            FT_CALC_LOG("tracking.config.camera depth_threshold=%f", mThDepth);
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=ThDepth type=real");
            b_miss_params = true;
        }


    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            FT_CALC_LOG("tracking.config.missing name=DepthMapFactor type=real");
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    FT_CALC_SCOPE();
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST, imageWidth, imageHeight;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=ORBextractor.nFeatures type=integer");
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=ORBextractor.scaleFactor type=real");
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=ORBextractor.nLevels type=integer");
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=ORBextractor.iniThFAST type=integer");
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=ORBextractor.minThFAST type=integer");
        b_miss_params = true;
    }

    node = fSettings["Camera.width"];
    if(!node.empty() && node.isInt())
    {
        imageWidth = node.operator int();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=Camera.width type=integer");
        b_miss_params = true;
    }

    node = fSettings["Camera.height"];
    if(!node.empty() && node.isInt())
    {
        imageHeight = node.operator int();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=Camera.height type=integer");
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST,imageWidth,imageHeight);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST,imageWidth,imageHeight);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
    {
        int iniFactor = 5;  // Init.IniFeaturesFactor (research, default 5 = upstream)
        cv::FileNode n = fSettings["Init.IniFeaturesFactor"]; if(!n.empty() && n.isInt() && (int)n >= 1 && (int)n <= 10) iniFactor = (int)n;
        mpIniORBextractor = new ORBextractor(iniFactor*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST,imageWidth,imageHeight);
    }

    FT_CALC_LOG("tracking.config.orb features=%d levels=%d scale_factor=%f initial_fast=%d minimum_fast=%d",
                nFeatures, nLevels, fScaleFactor, fIniThFAST, fMinThFAST);

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    FT_CALC_SCOPE();
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            FT_CALC_LOG("tracking.config.missing name=Tbc type=4x4_matrix");
            b_miss_params = true;
        }
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=Tbc type=matrix");
        b_miss_params = true;
    }
    FT_CALC_LOG("tracking.config.imu.tbc rows=%d cols=%d", cvTbc.rows, cvTbc.cols);
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        FT_CALC_LOG("tracking.config.imu insert_keyframes_when_lost=0");



    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=IMU.Frequency type=integer");
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=IMU.NoiseGyro type=real");
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=IMU.NoiseAcc type=real");
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=IMU.GyroWalk type=real");
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        FT_CALC_LOG("tracking.config.missing name=IMU.AccWalk type=real");
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    node = fSettings["IMU.ZUPT.Enabled"];
    mbZuptEnabled = (node.isInt() && static_cast<int>(node) != 0);
    node = fSettings["IMU.ZUPT.Window"];
    mZuptWindow = (node.isInt() && node.operator int() > 0) ? node.operator int() : (int)mImuFreq;
    node = fSettings["IMU.ZUPT.GyroVarTh"];
    mZuptGyroVarTh = (node.isReal() && node.real() > 0) ? node.real() : 1e-3;
    node = fSettings["IMU.ZUPT.AccelVarTh"];
    mZuptAccelVarTh = (node.isReal() && node.real() > 0) ? node.real() : 1e-1;
    node = fSettings["IMU.ZUPT.HystEnter"];
    mZuptHystEnter = (node.isInt() && node.operator int() > 0) ? node.operator int() : 3;
    node = fSettings["IMU.ZUPT.HystExit"];
    mZuptHystExit = (node.isInt() && node.operator int() > 0) ? node.operator int() : 3;
    node = fSettings["IMU.ZUPT.ExitRatio"];
    mZuptExitRatio = (node.isReal() && node.real() >= 1.0) ? node.real() : 3.0;
    node = fSettings["IMU.ZUPT.GyroMeanGate"];
    mZuptGyroMeanGate = (node.isReal() && node.real() > 0) ? node.real() : 0.03;
    node = fSettings["IMU.ZUPT.AccelNormGate"];
    mZuptAccelNormGate = (node.isReal() && node.real() > 0) ? node.real() : 0.3;
    node = fSettings["IMU.ZUPT.VelGate"];
    mZuptVelGate = (node.isReal() && node.real() > 0) ? node.real() : 0.05;
    node = fSettings["IMU.ZUPT.VelExitTh"];
    mZuptVelExitTh = (node.isReal() && node.real() > 0) ? node.real() : 0.15;
    node = fSettings["IMU.ZUPT.VelHyst"];
    mZuptVelHyst = (node.isInt() && node.operator int() > 0) ? node.operator int() : 3;
    node = fSettings["IMU.ZUPT.HoldRotation"];
    mbZuptHoldRotation = (node.isInt() && static_cast<int>(node) != 0);
    node = fSettings["IMU.ZUPT.StrongRest.Enabled"];
    mbZuptStrongRestEnabled = (node.isInt() && static_cast<int>(node) != 0);
    node = fSettings["Init.Metric.Enabled"];
    mbMetricInitEnabled = (node.isInt() && static_cast<int>(node) != 0);
    node = fSettings["Init.Metric.PadCameraHeight"]; if(node.isReal() && node.real() >= 0) mMetricPadCamHeight = node.real();
    node = fSettings["Init.Metric.MinHeight"]; if(node.isReal() && node.real() > 0) mMetricMinHeight = node.real();
    node = fSettings["Metric.DeferClimbRate"]; if(node.isReal() && node.real() >= 0) mDeferClimbRate = node.real();
    node = fSettings["Metric.ClimbMinParallaxDeg"]; if(node.isReal() && node.real() >= 0 && node.real() < 30) mClimbMinParallaxDeg = node.real();
    node = fSettings["Metric.ClimbParallaxRate"]; if(node.isReal() && node.real() > 0) mClimbParallaxRate = node.real();
    node = fSettings["Metric.BaroVz.Enabled"]; if(node.isInt()) mbBaroVz = node.operator int() != 0;
    node = fSettings["Metric.ClimbFullTailLink"]; if(node.isInt()) mbClimbFullTail = node.operator int() != 0;
    cout << "- Metric.ClimbFullTailLink: " << mbClimbFullTail << endl;
    node = fSettings["Metric.BaroVz.Sigma"]; if(node.isReal() && node.real() > 0) mBaroVzSigma = node.real();
    node = fSettings["Metric.BaroVz.Window"]; if(node.isReal() && node.real() >= 0.2 && node.real() <= 10) mBaroVzWindow = node.real();
    cout << "- Metric.BaroVz: enabled=" << mbBaroVz << " sigma=" << mBaroVzSigma << " m/s window=" << mBaroVzWindow << " s" << endl;
    node = fSettings["Metric.AccBiasObs.Enabled"]; if(node.isInt()) mbAccBiasObs = node.operator int() != 0;  // C7
    node = fSettings["Metric.VertKF.Enabled"]; if(node.isInt()) mbVertKF = node.operator int() != 0;  // C7c
    node = fSettings["Metric.TiltCorr.Enabled"]; if(node.isInt()) mbTiltCorr = node.operator int() != 0;  // C8
    node = fSettings["Metric.VertKF.AccNoise"]; if(node.isReal() && node.real() > 0) mVkfAccNoise = node.real();
    node = fSettings["Metric.VertKF.UseFcAttitude"]; if(node.isInt()) mbVkfUseFc = node.operator int() != 0;
    node = fSettings["Metric.VertKF.BiasRW"]; if(node.isReal() && node.real() > 0) mVkfBiasRW = node.real();
    node = fSettings["Metric.VertKF.BaroNoise"]; if(node.isReal() && node.real() > 0) mVkfBaroNoise = node.real();
    node = fSettings["Metric.VertKF.MinHeight"]; if(node.isReal() && node.real() >= 0) mVkfMinHeight = node.real();
    node = fSettings["Metric.VertKF.MaxBias"]; if(node.isReal() && node.real() > 0) mVkfMaxBias = node.real();
    node = fSettings["Metric.VertKF.MaxStep"]; if(node.isReal() && node.real() > 0) mVkfMaxStep = node.real();
    node = fSettings["Metric.VertKF.MaxSd"]; if(node.isReal() && node.real() > 0) mVkfMaxSd = node.real();
    node = fSettings["Metric.VertKF.ChainKFs"]; if(node.isInt() && (int)node >= 1) mVkfChainKFs = (int)node;
    cout << "- Metric.VertKF: enabled=" << mbVertKF << " accNoise=" << mVkfAccNoise << " biasRW=" << mVkfBiasRW
         << " baroNoise=" << mVkfBaroNoise << " maxBias=" << mVkfMaxBias << " maxStep=" << mVkfMaxStep << endl;
    node = fSettings["Metric.AccBiasObs.Window"]; if(node.isReal() && node.real() >= 2 && node.real() <= 60) mAccBiasObsWindow = node.real();
    node = fSettings["Metric.AccBiasObs.Sigma"]; if(node.isReal() && node.real() > 0) mAccBiasObsSigma = node.real();
    node = fSettings["Metric.AccBiasObs.MinKFs"]; if(node.isInt() && (int)node >= 5) mAccBiasObsMinKFs = (int)node;
    node = fSettings["Metric.AccBiasObs.MaxStep"]; if(node.isReal() && node.real() > 0) mAccBiasObsMaxStep = node.real();
    node = fSettings["Metric.AccBiasObs.MaxBias"]; if(node.isReal() && node.real() > 0) mAccBiasObsMaxBias = node.real();
    node = fSettings["Metric.AccBiasObs.MinHeight"]; if(node.isReal() && node.real() >= 0) mAccBiasObsMinHeight = node.real();
    cout << "- Metric.AccBiasObs: enabled=" << mbAccBiasObs << " window=" << mAccBiasObsWindow << " s sigma=" << mAccBiasObsSigma
         << " m/s^2 minKFs=" << mAccBiasObsMinKFs << " maxStep=" << mAccBiasObsMaxStep << " maxBias=" << mAccBiasObsMaxBias
         << " minHeight=" << mAccBiasObsMinHeight << " m" << endl;
    cout << "- Metric.ClimbMinParallaxDeg: " << mClimbMinParallaxDeg << " deg above " << mClimbParallaxRate << " m/s climb (0 = off)" << endl;
    cout << "- Metric.DeferClimbRate: " << mDeferClimbRate << " m/s (0 = off)" << endl;
    node = fSettings["Metric.Cruise.Enabled"]; if(node.isInt()) mbCruiseScale = node.operator int() != 0;
    node = fSettings["Metric.Cruise.Window"]; if(node.isInt() && node.operator int() >= 3) mCruiseWindow = node.operator int();
    node = fSettings["Metric.Cruise.Period"]; if(node.isInt() && node.operator int() >= 1) mCruisePeriod = node.operator int();
    node = fSettings["Metric.Cruise.Gain"]; if(node.isReal() && node.real() > 0 && node.real() <= 1) mCruiseGain = node.real();
    node = fSettings["Metric.Cruise.Deadband"]; if(node.isReal() && node.real() >= 0) mCruiseDeadband = node.real();
    node = fSettings["Metric.Cruise.MaxStep"]; if(node.isReal() && node.real() > 0 && node.real() < 0.5) mCruiseMaxStep = node.real();
    node = fSettings["Metric.Cruise.MinHeight"]; if(node.isReal() && node.real() > 0) mCruiseMinHeight = node.real();
    node = fSettings["Metric.Cruise.MaxTiltDeg"]; if(node.isReal() && node.real() > 0) mCruiseMaxTilt = node.real();
    node = fSettings["Metric.Cruise.MinInlier"]; if(node.isReal() && node.real() >= 0 && node.real() <= 1) mCruiseMinInlier = node.real();
    cout << "- Metric.Cruise: enabled=" << mbCruiseScale << " window=" << mCruiseWindow << " period=" << mCruisePeriod << " gain=" << mCruiseGain << " deadband=" << mCruiseDeadband << " maxStep=" << mCruiseMaxStep << endl;
    node = fSettings["Features.MaxRadiusPx"]; if(node.isReal() && node.real() >= 0) Frame::msMaxKeypointRadius = (float)node.real();
    cout << "- Features.MaxRadiusPx: " << Frame::msMaxKeypointRadius << endl;
    node = fSettings["Init.MinAxisAngleDeg"]; if(node.isReal() && node.real() >= 0 && node.real() < 90) mInitMinAxisAngleDeg = node.real();
    cout << "- Init.MinAxisAngleDeg: " << mInitMinAxisAngleDeg << endl;
    cout << "- Init.Metric: enabled=" << (mbMetricInitEnabled ? 1 : 0) << " padCameraHeight=" << mMetricPadCamHeight
         << " minHeight=" << mMetricMinHeight << endl;
    node = fSettings["Klt.Enabled"];
    mbKltEnabled = (node.isInt() && static_cast<int>(node) != 0);
    node = fSettings["Klt.WinSize"]; if(node.isInt() && (int)node >= 5) mKltWin = (int)node;
    node = fSettings["Klt.Levels"]; if(node.isInt() && (int)node >= 0) mKltLevels = (int)node;
    node = fSettings["Klt.MaxFbError"]; if(node.isReal() && node.real() > 0) mKltMaxFb = node.real();
    node = fSettings["Klt.Radius"]; if(node.isReal() && node.real() > 0) mKltRadius = node.real();
    node = fSettings["Klt.MaxHamming"]; if(node.isInt() && (int)node > 0) mKltMaxHamming = (int)node;
    node = fSettings["Klt.Persistent"]; if(node.isInt()) mbKltPersistent = (int)node != 0;
    node = fSettings["Klt.MaxAppend"]; if(node.isInt() && (int)node >= 0) mKltMaxAppend = (int)node;
    cout << "- Klt: enabled=" << (mbKltEnabled ? 1 : 0) << " win=" << mKltWin << " levels=" << mKltLevels
         << " maxFb=" << mKltMaxFb << " radius=" << mKltRadius << " maxHamming=" << mKltMaxHamming
         << " persistent=" << mbKltPersistent << " maxAppend=" << mKltMaxAppend << endl;
    node = fSettings["IMU.RestObs.Enabled"];
    mbRestObsEnabled = (node.isInt() && static_cast<int>(node) != 0);
    node = fSettings["IMU.RestObs.GyroSigma"];
    if(node.isReal() && node.real() > 0) mRestObsGyroSigma = node.real();
    node = fSettings["IMU.RestObs.AccSigma"];
    if(node.isReal() && node.real() > 0) mRestObsAccSigma = node.real();
    cout << "- IMU.RestObs: enabled=" << (mbRestObsEnabled ? 1 : 0) << " gyroSigma=" << mRestObsGyroSigma
         << " accSigma=" << mRestObsAccSigma << endl;
    node = fSettings["IMU.ZUPT.StrongRest.GyroVarTh"];
    mZuptStrongGyroVarTh = (node.isReal() && node.real() > 0) ? node.real() : 1e-5;
    node = fSettings["IMU.ZUPT.StrongRest.AccelVarTh"];
    mZuptStrongAccelVarTh = (node.isReal() && node.real() > 0) ? node.real() : 2.5e-4;
    node = fSettings["IMU.ZUPT.StrongRest.MinS"];
    mZuptStrongMinS = (node.isReal() && node.real() >= 0) ? node.real() : 2.0;
    mZuptStrongSince = -1.0;
    mbZuptStrongRest = false;

    mbZuptActive = false;
    mZuptEnterCount = 0;
    mZuptExitCount = 0;
    mZuptVelExitCount = 0;
    mZuptVelEma = 0.0;

    if(mbZuptEnabled)
    {
        cout << "ZUPT enabled (window=" << mZuptWindow << ", gyroVarTh=" << mZuptGyroVarTh
             << ", accelVarTh=" << mZuptAccelVarTh << ", hystEnter=" << mZuptHystEnter
             << ", hystExit=" << mZuptHystExit << ", exitRatio=" << mZuptExitRatio
             << ", gyroMeanGate=" << mZuptGyroMeanGate << ", accelNormGate=" << mZuptAccelNormGate
             << ", velGate=" << mZuptVelGate << ", velExitTh=" << mZuptVelExitTh
             << ", velHyst=" << mZuptVelHyst << ", holdRotation=" << (mbZuptHoldRotation ? 1 : 0)
             << ", strongRest=" << (mbZuptStrongRestEnabled ? 1 : 0) << " (gyroVar<" << mZuptStrongGyroVarTh
             << ", accelVar<" << mZuptStrongAccelVarTh << ", " << mZuptStrongMinS << " s))" << endl;
    }
    else
    {
        cout << "ZUPT disabled (set IMU.ZUPT.Enabled=1 to enable)" << endl;
    }

    if(mFastInit)
        FT_CALC_LOG("tracking.config.imu fast_initialization=1");

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    FT_CALC_LOG("tracking.config.imu frequency=%d gyro_noise=%f gyro_walk=%f acc_noise=%f acc_walk=%f",
                mImuFreq, Ng, Ngw, Na, Naw);

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}



Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("tracking.input.stereo timestamp=%f left=%dx%d right=%dx%d", timestamp,
                imRectLeft.cols, imRectLeft.rows, imRectRight.cols, imRectRight.rows);
    //cout << "GrabImageStereo" << endl;

    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        //cout << "Image with 3 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        //cout << "Image with 4 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    //cout << "Incoming frame creation" << endl;

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    // frame parsa
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    // frame akhari
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    //cout << "Incoming frame ended" << endl;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    //cout << "Tracking start" << endl;
    Track();
    //cout << "Tracking end" << endl;

    FT_CALC_LOG("tracking.output.stereo frame=%llu state=%d keypoints=%d",
                static_cast<unsigned long long>(mCurrentFrame.mnId), static_cast<int>(mState),
                mCurrentFrame.N);

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("tracking.input.rgbd timestamp=%f rgb=%dx%d depth=%dx%d depth_type=%d", timestamp,
                imRGB.cols, imRGB.rows, imD.cols, imD.rows, imD.type());
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    if (mSensor == System::RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::IMU_RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);






    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    Track();

    FT_CALC_LOG("tracking.output.rgbd frame=%llu state=%d keypoints=%d",
                static_cast<unsigned long long>(mCurrentFrame.mnId), static_cast<int>(mState),
                mCurrentFrame.N);

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("tracking.input.monocular timestamp=%f image=%dx%d channels=%d", timestamp, im.cols,
                im.rows, im.channels());
    mImGray = im;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if (mSensor == System::MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
    }
    else if(mSensor == System::IMU_MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }

    if (mState==NO_IMAGES_YET)
        t0=timestamp;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId;
    Track();
    if(mbKltEnabled)
        mKltPrevImg = mImGray.clone();

    FT_CALC_LOG("tracking.output.monocular frame=%llu state=%d keypoints=%d",
                static_cast<unsigned long long>(mCurrentFrame.mnId), static_cast<int>(mState),
                mCurrentFrame.N);

    return mCurrentFrame.GetPose();
}

// ---- Pipelined monocular input --------------------------------------------------------------
// Same extractor decision GrabImageMonocular() makes, evaluated against the current tracking state.
bool Tracking::WantIniExtractorMonocular() const
{
    if (mSensor == System::MONOCULAR)
        return mState==NOT_INITIALIZED || mState==NO_IMAGES_YET || (lastID - initID) < mMaxFrames;
    return mState==NOT_INITIALIZED || mState==NO_IMAGES_YET;
}

// Frame construction exactly as in GrabImageMonocular(), except that no previous frame is passed:
// the tracking thread may be rewriting mLastFrame concurrently. TrackPreparedMonocular() applies
// the previous-frame pointer and velocity at track time instead. Call with mMutexExtraction held.
Frame Tracking::BuildMonocularFrame(const cv::Mat &imGray, const double &timestamp, const bool useIni)
{
    ORBextractor* extractor = useIni ? mpIniORBextractor : mpORBextractorLeft;
    if (mSensor == System::IMU_MONOCULAR)
        return Frame(imGray,timestamp,extractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,static_cast<Frame*>(nullptr),*mpImuCalib);
    return Frame(imGray,timestamp,extractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
}

std::shared_ptr<Tracking::PreparedMonocular> Tracking::PrepareMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    auto prepared = std::make_shared<PreparedMonocular>();
    cv::Mat imGray = im;
    if(imGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(imGray,imGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(imGray,imGray,cv::COLOR_BGR2GRAY);
    }
    else if(imGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(imGray,imGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(imGray,imGray,cv::COLOR_BGRA2GRAY);
    }
    prepared->imGray = imGray;
    prepared->timestamp = timestamp;
    prepared->filename = filename;
    // Snapshot of the tracking state; re-checked in TrackPreparedMonocular() and re-extracted if stale.
    prepared->usedIniExtractor = WantIniExtractorMonocular();
    {
        unique_lock<mutex> lock(mMutexExtraction);
        prepared->frame = BuildMonocularFrame(imGray, timestamp, prepared->usedIniExtractor);
    }
    FT_CALC_LOG("tracking.prepare.monocular frame=%llu ini=%d features=%d",
                static_cast<unsigned long long>(prepared->frame.mnId),
                prepared->usedIniExtractor ? 1 : 0, prepared->frame.N);
    return prepared;
}

Sophus::SE3f Tracking::TrackPreparedMonocular(const std::shared_ptr<PreparedMonocular> &prepared)
{
    const double timestamp = prepared->timestamp;
    mImGray = prepared->imGray;

    const bool wantIni = WantIniExtractorMonocular();
    const bool staleExtractor = (wantIni != prepared->usedIniExtractor);
    const bool staleOrder = mbHaveTrackedPrepared && prepared->frame.mnId <= mnLastTrackedPreparedId;
    if (staleExtractor || staleOrder)
    {
        ++mnPipelineReextractions;
        FT_CALC_LOG("tracking.prepare.reextract frame=%llu stale_extractor=%d stale_order=%d",
                    static_cast<unsigned long long>(prepared->frame.mnId),
                    staleExtractor ? 1 : 0, staleOrder ? 1 : 0);
        unique_lock<mutex> lock(mMutexExtraction);
        mCurrentFrame = BuildMonocularFrame(mImGray, timestamp, wantIni);
    }
    else
    {
        mCurrentFrame = prepared->frame;
    }

    // What the serial Frame constructor takes from the previous frame (Frame.cc, IMU monocular ctor).
    if (mSensor == System::IMU_MONOCULAR)
    {
        mCurrentFrame.mpPrevFrame = &mLastFrame;
        if (mLastFrame.HasVelocity())
            mCurrentFrame.SetVelocity(mLastFrame.GetVelocity());
    }
    mbHaveTrackedPrepared = true;
    mnLastTrackedPreparedId = mCurrentFrame.mnId;

    // ---- identical to the tail of GrabImageMonocular() ----
    if (mState==NO_IMAGES_YET)
        t0=timestamp;

    mCurrentFrame.mNameFile = prepared->filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId;
    Track();
    // 2026-11-03: GrabImageMonocular() keeps the previous image for KLT since the pipelining patch (23 Oct); same here.
    if(mbKltEnabled)
        mKltPrevImg = mImGray.clone();

    FT_CALC_LOG("tracking.output.monocular frame=%llu state=%d keypoints=%d",
                static_cast<unsigned long long>(mCurrentFrame.mnId), static_cast<int>(mState),
                mCurrentFrame.N);

    return mCurrentFrame.GetPose();
}


void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    FT_CALC_LOG("imu.input t=%f ax=%f ay=%f az=%f wx=%f wy=%f wz=%f", imuMeasurement.t,
                imuMeasurement.a.x(), imuMeasurement.a.y(), imuMeasurement.a.z(),
                imuMeasurement.w.x(), imuMeasurement.w.y(), imuMeasurement.w.z());
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);

    if(mbZuptEnabled)
    {
        mlZuptImuWindow.push_back(imuMeasurement);
        while ((int)mlZuptImuWindow.size() > mZuptWindow)
            mlZuptImuWindow.pop_front();
    }
    else if(!mlZuptImuWindow.empty())
    {
        mlZuptImuWindow.clear();
    }
}

void Tracking::PreintegrateIMU()
{
    FT_CALC_SCOPE();
    size_t imuQueueSize = 0;
    {
        unique_lock<mutex> queueLock(mMutexImuQueue);
        imuQueueSize = mlQueueImuData.size();
    }
    FT_CALC_LOG("imu.preintegrate.begin frame=%llu queue=%llu",
                static_cast<unsigned long long>(mCurrentFrame.mnId),
                static_cast<unsigned long long>(imuQueueSize));

    // FTS: a default-constructed previous frame (start-up, after a reset) has no timestamp.
    if(!mCurrentFrame.mpPrevFrame || !std::isfinite(mCurrentFrame.mpPrevFrame->mTimeStamp))
    {
        FT_CALC_LOG("imu.preintegrate.skip reason=no_previous_frame");
        mCurrentFrame.setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(imuQueueSize);
    if(imuQueueSize == 0)
    {
        FT_CALC_LOG("imu.preintegrate.skip reason=empty_queue");
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mCurrentFrame.mTimeStamp-mImuPer)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }

    const int n = mvImuFromLastFrame.size()-1;
    if(n<=0){
        FT_CALC_LOG("imu.preintegrate.empty samples=%d", n);
        mCurrentFrame.setIntegrated();
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);
    // C7c: active on a metric map after inertial BA2 (gravity-aligned world); (re)initialised per map.
    bool vkfOn = false;
    Eigen::Matrix3f vkfR = Eigen::Matrix3f::Identity();
    if(mbVertKF)
    {
        Map* pM = mpAtlas->GetCurrentMap();
        {   // C7c diag (log only)
            static int c7cDiag = 0;
            if(++c7cDiag % 300 == 1)
            {
                double dh = 0, dhp = 0, dtb = 0;
                const bool alt = LatestAltitude(mCurrentFrame.mTimeStamp, dtb, dh, dhp);
                std::cerr << std::fixed << std::setprecision(3) << "FastTrack C7c diag t=" << mCurrentFrame.mTimeStamp
                          << " map=" << (pM ? 1 : 0) << " imuinit=" << (pM && pM->isImuInitialized() ? 1 : 0)
                          << " ba2=" << (pM && pM->GetIniertialBA2() ? 1 : 0)
                          << " lastpreint=" << (mLastFrame.mpImuPreintegrated ? 1 : 0)
                          << " alt=" << (alt ? 1 : 0) << " alt_age=" << (alt ? mCurrentFrame.mTimeStamp - dtb : -1.0)
                          << " active=" << (mVkfActive ? 1 : 0) << " b=" << mVkfX[2] << " sdb=" << std::sqrt(std::max(0.0, mVkfP[2][2]))
                          << " h=" << mVkfX[0] << " vz=" << mVkfX[1]
                          << std::defaultfloat << std::setprecision(6) << std::endl;
            }
        }
        if(pM && pM->isImuInitialized() && pM->GetIniertialBA2() && mLastFrame.mpImuPreintegrated)
        {
            bool fcOk = true;  // C7c v3
            if(mbVkfUseFc)
            {
                Eigen::Matrix3f Rfc;
                fcOk = mpLocalMapper && mpLocalMapper->FcRotationAt(mCurrentFrame.mTimeStamp, Rfc);
                if(fcOk) vkfR = Rfc;
            }
            else
                vkfR = mLastFrame.GetImuRotation();
            unique_lock<mutex> lk(mMutexVkf);
            if(!mVkfActive || mVkfMapId != pM->GetId())
            {
                double h = 0, hPad = 0, tb = 0;
                if(LatestAltitude(mCurrentFrame.mTimeStamp, tb, h, hPad))
                {
                    mVkfX[0] = h - hPad; mVkfX[1] = 0.0; mVkfX[2] = 0.0;
                    for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) mVkfP[i][j] = 0.0;
                    mVkfP[0][0] = 1.0; mVkfP[1][1] = 1.0; mVkfP[2][2] = 0.09;
                    mVkfActive = true; mVkfMapId = pM->GetId(); mVkfLastBaroT = mCurrentFrame.mTimeStamp;
                }
            }
            vkfOn = mVkfActive && fcOk;
        }
        else
        {
            unique_lock<mutex> lk(mMutexVkf);
            mVkfActive = false;
        }
    }

    for(int i=0; i<n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        FT_CALC_LOG("imu.preintegrate.step index=%d dt=%f ax=%f ay=%f az=%f wx=%f wy=%f wz=%f", i,
                    tstep, acc.x(), acc.y(), acc.z(), angVel.x(), angVel.y(), angVel.z());
        if (!mpImuPreintegratedFromLastKF)
        {
            FT_CALC_LOG("imu.preintegrate.abort reason=missing_keyframe_integrator");
            mCurrentFrame.setIntegrated();
            delete pImuPreintegratedFromLastFrame;
            return;
        }
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
        if(vkfOn)  // C7c: vertical specific force in the gravity-aligned world
            VertKFPredict(double((vkfR * acc)(2)) - IMU::GRAVITY_VALUE, tstep);
    }
    if(vkfOn)
        VertKFBaro(mCurrentFrame.mTimeStamp);

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    mCurrentFrame.setIntegrated();

    FT_CALC_LOG("imu.preintegrate.end samples=%d dT=%f", n,
                mCurrentFrame.mpImuPreintegratedFrame->dT);

    //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}


bool Tracking::PredictStateIMU()
{
    FT_CALC_SCOPE();
    if(!mCurrentFrame.mpPrevFrame)
    {
        FT_CALC_LOG("imu.predict.skip reason=no_previous_frame");
        return false;
    }

    if(mbMapUpdated && mpLastKeyFrame && mpImuPreintegratedFromLastKF)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());

        // ZUPT output hold: if stationary, prevent integration from accumulating
        if(mCurrentFrame.IsStationary() && (Vwb2.norm() < mZuptVelExitTh || mCurrentFrame.IsStrongRest()))
        {
            Vwb2.setZero();
            twb2 = twb1;
            if(mbZuptHoldRotation)
                Rwb2 = Rwb1;
        }
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        FT_CALC_LOG("imu.predict.source=last_keyframe dt=%f map_updated=1", t12);
        return true;
    }
    else if(!mbMapUpdated && mCurrentFrame.mpImuPreintegratedFrame)
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        // ZUPT output hold: if stationary, prevent integration from accumulating
        if(mCurrentFrame.IsStationary() && (Vwb2.norm() < mZuptVelExitTh || mCurrentFrame.IsStrongRest()))
        {
            Vwb2.setZero();
            twb2 = twb1;
            if(mbZuptHoldRotation)
                Rwb2 = Rwb1;
        }

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        FT_CALC_LOG("imu.predict.source=last_frame dt=%f map_updated=0", t12);
        return true;
    }
    else
        FT_CALC_LOG("imu.predict.skip reason=missing_preintegration map_updated=%d", mbMapUpdated ? 1 : 0);

    return false;
}

void Tracking::UpdateZuptState()
{
    mCurrentFrame.SetStationary(false);
    mCurrentFrame.SetStrongRest(false);

    if(!mbZuptEnabled)
        return;

    bool gyroMeanGate = false, accelNormGate = false, windowFull = false;
    double varG, varA;
    {
        unique_lock<mutex> lock(mMutexImuQueue);

        const int n = (int)mlZuptImuWindow.size();
        if(n < 2)
            return;
        windowFull = (n >= mZuptWindow);

        // Variance of gyro and accel magnitudes over the sliding window
        double sumG = 0.0, sumA = 0.0;
        for(int i = 0; i < n; i++)
        {
            const IMU::Point &p = mlZuptImuWindow[i];
            sumG += p.w.norm();
            sumA += p.a.norm();
        }
        const double meanG = sumG / n;
        const double meanA = sumA / n;

        // Motion gates: sustained rotation or accel-magnitude far from gravity
        // are strong evidence of motion even when the window variance stays low.
        gyroMeanGate = (meanG > mZuptGyroMeanGate);
        accelNormGate = (fabs(meanA - IMU::GRAVITY_VALUE) > mZuptAccelNormGate);

        varG = 0.0;
        varA = 0.0;
        for(int i = 0; i < n; i++)
        {
            const IMU::Point &p = mlZuptImuWindow[i];
            const double dG = p.w.norm() - meanG;
            const double dA = p.a.norm() - meanA;
            varG += dG * dG;
            varA += dA * dA;
        }
        varG /= n;
        varA /= n;
    }

    // Strong rest: the IMU window is far stiller than any flight (Sep-30 flight, 1 s windows in the air:
    // gyro-magnitude variance >= 5.2e-5, accel >= 7.7e-4; on the pad 2026-10-22: ~1e-6 and 1-3e-5).
    // Then the vehicle is at rest whatever the estimator's velocity says; with weak vision (camera cm above
    // the pad) that velocity drifted to 0.5-1 m/s and the velocity gates below kept ZUPT off for 45 s.
    bool strongRest = false;
    if(mbZuptStrongRestEnabled && windowFull && !gyroMeanGate && !accelNormGate &&
       varG < mZuptStrongGyroVarTh && varA < mZuptStrongAccelVarTh)
    {
        if(mZuptStrongSince < 0.0 || mCurrentFrame.mTimeStamp < mZuptStrongSince)
            mZuptStrongSince = mCurrentFrame.mTimeStamp;
        strongRest = (mCurrentFrame.mTimeStamp - mZuptStrongSince) >= mZuptStrongMinS;
    }
    else
    {
        mZuptStrongSince = -1.0;
    }
    if(strongRest != mbZuptStrongRest)
    {
        cout << "ZUPT: strong rest " << (strongRest ? "begins" : "ends") << " (gyroVar=" << varG
             << ", accelVar=" << varA << ")" << endl;
        FT_CALC_LOG("zupt.strong_rest state=%d gyro_var=%e accel_var=%e", strongRest ? 1 : 0, varG, varA);
        mbZuptStrongRest = strongRest;
    }

    bool velocityGate = false;
    double lastVelNorm = 0.0;
    if(mLastFrame.HasVelocity())
    {
        lastVelNorm = mLastFrame.GetVelocity().norm();
        // EMA of the tracked velocity magnitude so a single noisy reading does
        // not flip the ZUPT hold. During hover the tracked velocity is exactly
        // the signal that drifts, so releasing on it requires a higher,
        // sustained threshold (mZuptVelExitTh) with hysteresis, instead of the
        // 0.05 m/s single-frame gate that dropped the hold during dead-still
        // hover in the Sep-4 flights.
        const double velAlpha = 0.3;
        mZuptVelEma = velAlpha * lastVelNorm + (1.0 - velAlpha) * mZuptVelEma;
        if(mbZuptActive)
        {
            if(mZuptVelEma > mZuptVelExitTh)
            {
                mZuptVelExitCount++;
                if(mZuptVelExitCount >= mZuptVelHyst)
                    velocityGate = true;
            }
            else if(mZuptVelEma <= mZuptVelGate)
            {
                mZuptVelExitCount = 0;
            }
        }
        else
        {
            // Block entering stationary while the tracked velocity is
            // non-trivial (e.g. smooth constant-velocity flight).
            velocityGate = (mZuptVelEma > mZuptVelGate);
        }
    }
    else
    {
        mZuptVelEma *= 0.9;
    }
    if(strongRest)
    {
        velocityGate = false;
        mZuptVelExitCount = 0;
    }

    const bool belowTh = (varG < mZuptGyroVarTh) && (varA < mZuptAccelVarTh);
    const bool forcedMotion = gyroMeanGate || accelNormGate || velocityGate;

    if(mbZuptActive)
    {
        if(forcedMotion)
        {
            mbZuptActive = false;
            mZuptExitCount = 0;
            cout << "ZUPT: exiting stationary (motion gate) gyroM=" << gyroMeanGate
                 << " accelN=" << accelNormGate
                 << " velGate=" << velocityGate
                 << " vel=" << lastVelNorm << endl;
        }
        else if(belowTh || (varG < mZuptExitRatio * mZuptGyroVarTh && varA < mZuptExitRatio * mZuptAccelVarTh))
        {
            mZuptExitCount = 0;
        }
        else
        {
            mZuptExitCount++;
            if(mZuptExitCount >= mZuptHystExit)
            {
                mbZuptActive = false;
                mZuptExitCount = 0;
            }
        }
    }
    else
    {
        if(belowTh && !forcedMotion)
        {
            mZuptEnterCount++;
            if(mZuptEnterCount >= mZuptHystEnter || strongRest)
            {
                mbZuptActive = true;
                mZuptEnterCount = 0;
                cout << "ZUPT: entering stationary (gyroVar=" << varG << ", accelVar=" << varA << ")" << endl;
            }
        }
        else
        {
            mZuptEnterCount = 0;
        }
    }

    mCurrentFrame.SetStationary(mbZuptActive);
    mCurrentFrame.SetStrongRest(mbZuptActive && strongRest);

    // WP B2: at strong rest the window means are measurements of the gyro bias and of gravity in the body frame.
    bool bRestObs = false;
    Eigen::Vector3f mg = Eigen::Vector3f::Zero(), ma = Eigen::Vector3f::Zero();
    double infoG = 0.0, infoA = 0.0;
    if(mbRestObsEnabled && mbZuptActive && strongRest)
    {
        unique_lock<mutex> lock(mMutexImuQueue);
        const int n = (int)mlZuptImuWindow.size();
        if(n >= mZuptWindow && n > 1)
        {
            for(const IMU::Point &p : mlZuptImuWindow) { mg += p.w; ma += p.a; }
            mg /= (float)n; ma /= (float)n;
            // Window-mean uncertainty from the white-noise densities, floored (bias is not perfectly constant).
            const double f = mImuFreq > 0 ? mImuFreq : 200.0;
            const double sg = std::max(mRestObsGyroSigma, (double)mpImuCalib->Cov.diagonal()(0) > 0 ? std::sqrt((double)mpImuCalib->Cov.diagonal()(0) / n) : 0.0);
            const double sa = std::max(mRestObsAccSigma, (double)mpImuCalib->Cov.diagonal()(3) > 0 ? std::sqrt((double)mpImuCalib->Cov.diagonal()(3) / n) : 0.0);
            infoG = 1.0 / (sg * sg); infoA = 1.0 / (sa * sa);
            bRestObs = mg.allFinite() && ma.allFinite() && std::fabs(ma.norm() - IMU::GRAVITY_VALUE) < 0.5;
            (void)f;
        }
    }
    mCurrentFrame.SetRestObs(bRestObs, mg, ma, infoG, infoA);
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}


void Tracking::Track()
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("tracking.step.begin frame=%llu state=%d sensor=%d only_tracking=%d",
                static_cast<unsigned long long>(mCurrentFrame.mnId), static_cast<int>(mState),
                mSensor, mbOnlyTracking ? 1 : 0);

    if (bStepByStep)
    {
        FT_CALC_LOG("tracking.step.waiting_for_manual_release frame=%llu",
                    static_cast<unsigned long long>(mCurrentFrame.mnId));
        while(!mbStep && bStepByStep)
            usleep(500);
        mbStep = false;
    }

    if(mpLocalMapper->mbBadImu)
    {
        FT_CALC_LOG("tracking.reset reason=local_mapper_bad_imu");
        mpSystem->ResetActiveMap();
        return;
    }

    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if(!pCurrentMap)
    {
        FT_CALC_LOG("tracking.abort reason=no_active_map");
        return;
    }

    const double allowedGapS = mAllowedGapOnceS;  // AllowTimeGapOnce: applies to this frame only
    mAllowedGapOnceS = 0.0;
    if(mState!=NO_IMAGES_YET && allowedGapS > 0.0 && mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
    {
        FT_CALC_LOG("tracking.timestamp_gap_allowed gap=%f allowed=%f", mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp, allowedGapS);
        std::cerr << "FastTrack: expected image gap of " << (mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp)
                  << " s (wrapper pause); predicting across it with the IMU" << std::endl;
    }
    if(mState!=NO_IMAGES_YET)
    {
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {
            FT_CALC_LOG("tracking.reset reason=timestamp_regression previous=%f current=%f",
                        mLastFrame.mTimeStamp, mCurrentFrame.mTimeStamp);
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0+allowedGapS)
        {
            // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
            // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            if(mpAtlas->isInertial())
            {

                if(mpAtlas->isImuInitialized())
                {
                    FT_CALC_LOG("tracking.reset reason=timestamp_jump imu_initialized=1 previous=%f current=%f",
                                mLastFrame.mTimeStamp, mCurrentFrame.mTimeStamp);
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        mpSystem->ResetActiveMap();
                    }
                    else
                    {
                        CreateMapInAtlas();
                    }
                }
                else
                {
                    // Before IMU initialization a jump used to skip the frame without updating mLastFrame, so every
                    // later frame was also a jump and initialization never resumed (2026-10-21: a >1 s pause in the
                    // image stream stalled initialization until restart). Upstream ORB-SLAM3 resets the active map.
                    FT_CALC_LOG("tracking.timestamp_jump imu_initialized=0 action=reset_active_map previous=%f current=%f",
                                mLastFrame.mTimeStamp, mCurrentFrame.mTimeStamp);
                    std::cerr << "FastTrack: image time jump of " << (mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp)
                              << " s before IMU initialization; resetting the active map" << std::endl;
                    mpSystem->ResetActiveMap();
                }
                return;
            }

        }
    }


    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU();
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu);
#endif

    }
    mbCreatedMap = false;

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        UpdateZuptState();

    // Prior-map start (PriorMap.cc): relocalize a stationary vehicle into the loaded inertial map instead of the
    // motion-based initialization. May switch pCurrentMap (session map for motion initialization / prior map).
    bool bPriorReloc = false;
    if(mpPriorMap && mState==NOT_INITIALIZED && mSensor==System::IMU_MONOCULAR)
    {
        const PriorStep step = PriorMapStep(pCurrentMap);
        if(step == kPriorWait)
        {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }
        bPriorReloc = (step == kPriorRelocalized);
    }
    else if(mpPriorMap && mSensor==System::IMU_MONOCULAR && mPriorOpt.reloc_while_moving && pCurrentMap != mpPriorMap &&
            (mState==OK || mState==RECENTLY_LOST) && GetPriorMapStatus().state != PriorMapStatus::kRelocalized)
    {
        // Session map tracking (motion initialization done): keep trying the prior map (PriorMapStep, moving
        // attempts). A failed attempt must not disturb this frame's normal tracking, so the frame is restored.
        Frame frameBackup(mCurrentFrame);
        const PriorStep step = PriorMapStep(pCurrentMap, true);
        if(step == kPriorRelocalized)
            bPriorReloc = true;
        else
            mCurrentFrame = frameBackup;
    }

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);
    // RESEARCH frame-epoch contract: ApplyScaledRotation needs this mutex, so the epoch cannot
    // change while this frame is tracked.
    mnPoseMapId = pCurrentMap->GetId();
    mnPoseFrameEpoch = pCurrentMap->GetFrameEpoch();
    mbPoseEpochRecorded = true;

    mbMapUpdated = false;

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }
    if(mbPriorForceMapUpdated)
    {
        // First frame after the prior-map keyframe: predict from it (last-keyframe inertial optimization).
        mbMapUpdated = true;
        mbPriorForceMapUpdated = false;
    }


    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
        {
            StereoInitialization();
        }
        else
        {
            MonocularInitialization();
        }

        //mpFrameDrawer->Update(this);

        if(mState!=OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        if(mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(bPriorReloc)
            {
                bOK = true;  // pose from the prior-map relocalization
            }
            else if(mState==OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    FT_CALC_LOG("tracking.pose_estimation.method=reference_keyframe velocity=%d imu_initialized=%d",
                                mbVelocity ? 1 : 0, pCurrentMap->isImuInitialized() ? 1 : 0);
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    FT_CALC_LOG("tracking.pose_estimation.method=motion_model");
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                    {
                        FT_CALC_LOG("tracking.pose_estimation.fallback=reference_keyframe");
                        bOK = TrackReferenceKeyFrame();
                    }
                }

                FT_CALC_LOG("tracking.pose_estimation.result ok=%d", bOK ? 1 : 0);


                if (!bOK)
                {
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST;
                        FT_CALC_LOG("tracking.state_transition from=ok to=lost reason=imu_recovery_window");
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10)
                    {
                        // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                        FT_CALC_LOG("tracking.state_transition from=ok to=recently_lost keyframes=%d",
                                    pCurrentMap->KeyFramesInMap());
                    }
                    else
                    {
                        mState = LOST;
                        FT_CALC_LOG("tracking.state_transition from=ok to=lost keyframes=%d",
                                    pCurrentMap->KeyFramesInMap());
                    }
                }
            }
            else
            {

                if (mState == RECENTLY_LOST)
                {
                    FT_CALC_LOG("tracking.recovery.begin state=recently_lost");

                    bOK = true;
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST;
                            FT_CALC_LOG("tracking.state_transition from=recently_lost to=lost elapsed=%f",
                                        mCurrentFrame.mTimeStamp - mTimeStampLost);
                            bOK=false;
                        }
                    }
                    else
                    {
                        // Relocalization
                        bOK = Relocalization();
                        //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        if(mCurrentFrame.mTimeStamp-mTimeStampLost>3.0f && !bOK)
                        {
                            mState = LOST;
                            FT_CALC_LOG("tracking.state_transition from=recently_lost to=lost elapsed=%f",
                                        mCurrentFrame.mTimeStamp - mTimeStampLost);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST)
                {
                    FT_CALC_LOG("tracking.map_recovery.begin keyframes=%d", pCurrentMap->KeyFramesInMap());

                    if (pCurrentMap->KeyFramesInMap()<10)
                    {
                        mpSystem->ResetActiveMap();
                        FT_CALC_LOG("tracking.map_recovery.action=reset_active_map");
                    }else
                    {
                        CreateMapInAtlas();
                        FT_CALC_LOG("tracking.map_recovery.action=create_map");
                    }

                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    return;
                }
            }

        }
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if(mState==LOST)
            {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(mbVelocity)
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    Sophus::SE3f TcwMM;
                    if(mbVelocity)
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.GetPose();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred);
#endif


#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                bOK = TrackLocalMap();

            }
            if(!bOK)
                FT_CALC_LOG("tracking.local_map.result=fail");
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bPriorReloc && !bOK)
        {
            PriorMapOnLocalMapFail();
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        if(bOK)
            mState = OK;
        else if (mState == OK)
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    FT_CALC_LOG("tracking.reset reason=imu_not_ready_or_recently_initialized");
                    mpSystem->ResetActiveMap();
                }

                mState=RECENTLY_LOST;
            }
            else
                mState=RECENTLY_LOST; // visual to lost

            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            //}
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
           (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
        {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if(pCurrentMap->isImuInitialized())
        {
            if(bOK)
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    FT_CALC_LOG("tracking.imu_reset frame=%llu",
                                static_cast<unsigned long long>(mCurrentFrame.mnId));
                    ResetFrameIMU();
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    mLastBias = mCurrentFrame.mImuBias;
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        // Update drawer
        mpFrameDrawer->Update(this);
        if(mCurrentFrame.isSet())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        if(bOK || mState==RECENTLY_LOST)
        {
            // Update motion model
            if(mLastFrame.isSet() && mCurrentFrame.isSet())
            {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                mVelocity = mCurrentFrame.GetPose() * LastTwc;
                mbVelocity = true;
            }
            else {
                mbVelocity = false;
            }

            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = NeedNewKeyFrame();

            // Check if we need to insert a new keyframe
            // if(bNeedKF && bOK)
            if(bPriorReloc && bOK)
                PriorMapOnRelocalizedKeyFrame();  // first live keyframe: new temporal chain, velocity 0, map bias
            else if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST &&
                                   (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                CreateNewKeyFrame();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(pCurrentMap->KeyFramesInMap()<=10)
            {
                FT_CALC_LOG("tracking.reset reason=lost_early_initialization keyframes=%d",
                            pCurrentMap->KeyFramesInMap());
                mpSystem->ResetActiveMap();
                return;
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    FT_CALC_LOG("tracking.reset reason=lost_before_imu_initialization");
                    mpSystem->ResetActiveMap();
                    return;
                }

            CreateMapInAtlas();

            return;
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }




    if(mState==OK || mState==RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(mCurrentFrame.isSet())
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState==LOST);
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);
        }

    }

    FT_CALC_LOG("tracking.step.end frame=%llu state=%d success=%d tracked_map_points=%d",
                static_cast<unsigned long long>(mCurrentFrame.mnId), static_cast<int>(mState),
                mState == OK ? 1 : 0, mTrackedFr);

#ifdef REGISTER_LOOP
    if (Stop()) {

        // Safe area to stop
        while(isStopped())
        {
            usleep(3000);
        }
    }
#endif
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                FT_CALC_LOG("tracking.initialization.abort reason=missing_imu_measurement");
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA).norm()<0.5)
            {
                FT_CALC_LOG("tracking.initialization.abort reason=insufficient_acceleration");
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        else
            mCurrentFrame.SetPose(Sophus::SE3f());

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if(z>0)
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereo(i, x3D);
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        //cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        //mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        mState=OK;
    }
}


void Tracking::MonocularInitialization()
{

    if(!mbReadyToInitializate)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {

            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            if (mSensor == System::IMU_MONOCULAR)
            {
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }

            mbReadyToInitializate = true;

            return;
        }
    }
    else
    {
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            mbReadyToInitializate = false;

            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            mbReadyToInitializate = false;
            return;
        }

        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvIniMatches,Tcw,mvIniP3D,vbTriangulated))
        {
            // B4b: direction of the current camera centre in the initial camera frame vs the optical axis (z).
            {
                const Eigen::Vector3f c = Tcw.inverse().translation();
                const float axisDeg = c.norm() > 0.f ? std::acos(std::min(1.f, std::fabs(c(2)) / c.norm())) * 180.f / (float)M_PI : 0.f;
                int nTri = 0; for(bool b : vbTriangulated) nTri += b ? 1 : 0;
                const bool reject = mInitMinAxisAngleDeg > 0.0 && axisDeg < mInitMinAxisAngleDeg;
                std::cerr << "FastTrack mono init: t=" << std::fixed << std::setprecision(3) << mCurrentFrame.mTimeStamp
                          << " dt=" << mCurrentFrame.mTimeStamp - mInitialFrame.mTimeStamp << std::defaultfloat << std::setprecision(6)
                          << " axis angle " << axisDeg << " deg, triangulated " << nTri << ", matches " << nmatches
                          << (reject ? " -> rejected (Init.MinAxisAngleDeg)" : "") << std::endl;
                if(reject)
                {
                    mbReadyToInitializate = false;
                    return;
                }
            }
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(Sophus::SE3f());
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}



void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }


    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    mpLocalMapper->SetMetricScaleFixed(false);
    if(mbMetricInitEnabled && mSensor == System::IMU_MONOCULAR && medianDepth > 0)
    {
        const float s = MetricInitScale(pKFini, pKFcur);
        if(s > 0.f)
        {
            invMedianDepth = s;
            mpLocalMapper->SetMetricScaleFixed(true);
        }
    }

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap();
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }


    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    //mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;

    initID = pKFcur->mnId;
}


void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame.mnId+1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
    //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mbReadyToInitializate = false;
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    FT_CALC_SCOPE();
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
    {
        FT_CALC_LOG("tracking.reference_keyframe.abort matches=%d minimum=15", nmatches);
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    //mCurrentFrame.PrintPointDistribution();


    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        //if(i >= mCurrentFrame.Nleft) break;
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    Sophus::SE3f Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    const int Nfeat = mLastFrame.Nleft == -1? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for(int i=0; i<Nfeat;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
            bCreateNew = true;

        if(bCreateNew)
        {
            Eigen::Vector3f x3D;

            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else{
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);
            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;

    }
}

bool Tracking::TrackWithMotionModel()
{

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point TWM_start = std::chrono::steady_clock::now();
#endif

    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
        if(mbKltEnabled)
            KltAssociate();
        return true;
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    }




    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
    if(mbKltEnabled)
        KltAssociate();

    // Project points seen in previous frame
    int th;

    if(mSensor==System::STEREO)
        th=7;
    else
        th=15;

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point PE_start = std::chrono::steady_clock::now();
#endif

    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point PE_end = std::chrono::steady_clock::now();
#endif

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point TWM_end = std::chrono::steady_clock::now();
    double trackWithMotionModel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(TWM_end - TWM_start).count();
    double poseEstimation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(PE_end - PE_start).count();
    TrackingStats::getInstance().trackWithMotionModel_time.emplace_back(mCurrentFrame.mnId, trackWithMotionModel);
    TrackingStats::getInstance().TWM_poseEstimation_time.emplace_back(mCurrentFrame.mnId, poseEstimation);
#endif
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return false;
    }

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point PO_start = std::chrono::steady_clock::now();
#endif
    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point PO_end = std::chrono::steady_clock::now();
#endif

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point TWM_end = std::chrono::steady_clock::now();
    double trackWithMotionModel = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(TWM_end - TWM_start).count();
    double poseEstimation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(PE_end - PE_start).count();
    double poseOptimization = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(PO_end - PO_start).count();
    TrackingStats::getInstance().trackWithMotionModel_time.emplace_back(mCurrentFrame.mnId, trackWithMotionModel);
    TrackingStats::getInstance().TWM_poseEstimation_time.emplace_back(mCurrentFrame.mnId, poseEstimation);
    TrackingStats::getInstance().TWM_poseOptimization_time.emplace_back(mCurrentFrame.mnId, poseOptimization);
#endif

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{

#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point TLM_start = std::chrono::steady_clock::now();
#endif
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;
#ifdef REGISTER_STATS    
    std::chrono::steady_clock::time_point ULM_start = std::chrono::steady_clock::now();
#endif 
    UpdateLocalMap();

#ifdef REGISTER_STATS    
    std::chrono::steady_clock::time_point ULM_end = std::chrono::steady_clock::now(); 
    std::chrono::steady_clock::time_point SLP_start = std::chrono::steady_clock::now(); 
#endif 
    SearchLocalPoints();
#ifdef REGISTER_STATS 
    std::chrono::steady_clock::time_point SLP_end = std::chrono::steady_clock::now();
#endif

    // TOO check outliers before PO
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++) {
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }
    }

#ifdef REGISTER_STATS 
    std::chrono::steady_clock::time_point PO_start = std::chrono::steady_clock::now(); 
#endif

    bool bPoseOpt = KernelController::poseOptimizationRunStatus == 1;
    if (mPoseOptMode == 0 || mPoseOptMode == 1)
        bPoseOpt = mPoseOptMode == 1;
    else if (mPoseOptMode == 2 && mpAtlas->GetCurrentMap()->GetIniertialBA2())
        bPoseOpt = true;
    else if (mPoseOptMode == 3 || mPoseOptMode == 4)
    {
        // P3 (2026-10-30): turns (and for mode 4 map updates) only, after inertial BA2.
        if (mCurrentFrame.mpImuPreintegratedFrame && mCurrentFrame.mpImuPreintegratedFrame->dT > 1e-3f)
        {
            const Eigen::Matrix3f dR = IMU::NormalizeRotation(mCurrentFrame.mpImuPreintegratedFrame->GetUpdatedDeltaRotation());
            const double rate = Eigen::AngleAxisf(dR).angle() / mCurrentFrame.mpImuPreintegratedFrame->dT * 180.0 / M_PI;
            if (rate >= mPoseOptTurnRateDps)
                mPoseOptLastTurnT = mCurrentFrame.mTimeStamp;
        }
        const bool turning = mPoseOptLastTurnT >= 0.0 && mCurrentFrame.mTimeStamp - mPoseOptLastTurnT <= mPoseOptTurnHoldS;
        bPoseOpt = mpAtlas->GetCurrentMap()->GetIniertialBA2() && (turning || (mPoseOptMode == 4 && mbMapUpdated));
    }
    if (bPoseOpt) {
        int inliers;
        if (!mpAtlas->isImuInitialized())
            Optimizer::PoseOptimization(&mCurrentFrame);
        else
        {
            if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
            {
                Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
                Optimizer::PoseOptimization(&mCurrentFrame);
            }
            else
            {
                // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
                // P1b (2026-10-30): with pose_optimization_mode 2 the optimization starts at inertial BA2, so the
                // previous frame may carry no IMU constraint (mpcpi NULL): PoseInertialOptimizationLastFrame would
                // dereference it (garuda03 replay f1609-t1p1-1: SIGSEGV in EdgePriorPoseImu). Start from the last
                // keyframe instead; that call creates the constraint for the next frame.
                if(!mbMapUpdated && mCurrentFrame.mpPrevFrame && mCurrentFrame.mpPrevFrame->mpcpi) //  && (mnMatchesInliers>30))
                {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                }
                else
                {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                }
            }
        }
    }
#ifdef REGISTER_STATS 
    std::chrono::steady_clock::time_point PO_end = std::chrono::steady_clock::now(); 
#endif

    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++) {
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }
    }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }
#ifdef REGISTER_STATS  
    std::chrono::steady_clock::time_point TLM_end = std::chrono::steady_clock::now();

    double trackLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(TLM_end - TLM_start).count();
    double updateLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(ULM_end - ULM_start).count();
    double searchLocalPoints = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(SLP_end - SLP_start).count();
    double poseOptimization = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(PO_end - PO_start).count();
    TrackingStats::getInstance().trackLocalMap_time.emplace_back(mCurrentFrame.mnId, trackLocalMap);
    TrackingStats::getInstance().updateLocalMap_time.emplace_back(mCurrentFrame.mnId, updateLocalMap);
    TrackingStats::getInstance().searchLocalPoints_time.emplace_back(mCurrentFrame.mnId, searchLocalPoints);
    TrackingStats::getInstance().TLM_poseOptimization_time.emplace_back(mCurrentFrame.mnId, poseOptimization);
#endif

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers;
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST))
        return true;


    if (mSensor == System::IMU_MONOCULAR)
    {
        if((mnMatchesInliers<15 && mpAtlas->isImuInitialized())||(mnMatchesInliers<50 && !mpAtlas->isImuInitialized()))
        {
            return false;
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else
    {
        if(mnMatchesInliers<30)
            return false;
        else
            return true;
    }
}

bool Tracking::NeedNewKeyFrame()
{
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        FT_CALC_LOG("tracking.new_keyframe.decision=false reason=local_mapper_stopped");
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
        //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && mSensor!=System::IMU_RGBD && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
            {
                //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                return false;
            }
        }
    }
    else
        return false;
}

void Tracking::SetExternalReference(const bool bHasYaw, const double yawDeg, const bool bHasAlt, const double altM)
{
    unique_lock<mutex> lock(mMutexExtRef);
    mbExtRefHasYaw = bHasYaw;
    mbExtRefHasAlt = bHasAlt;
    if(bHasYaw)
        mExtRefYawDeg = yawDeg;
    if(bHasAlt)
        mExtRefAltM = altM;
    if((bHasYaw || bHasAlt) && !mbExtRefPrinted)
    {
        cout << "External reference (magnetometer yaw + barometric alt) feed ENABLED:"
             << " magYawSigmaDeg=" << mExtMagYawSigmaDeg
             << " baroAltSigmaM=" << mExtBaroAltSigmaM << endl;
        mbExtRefPrinted = true;
    }
}

void Tracking::LatchExternalRef()
{
    // Must hold mMutexExtRef. Requires the map to be IMU-initialized so that
    // scale/gravity are settled and the world Z axis is gravity-aligned.
    if(mExtRefLatched)
        return;
    if(!mpAtlas->isImuInitialized())
        return;

    const Sophus::SE3f Twc = mCurrentFrame.GetPose().inverse();
    const Eigen::Matrix3f R = Twc.rotationMatrix();
    const double slamYaw = atan2((double)R(1,0), (double)R(0,0));
    const double slamZ = (double)mCurrentFrame.GetCameraCenter()(2);

    bool latched = false;
    if(mbExtRefHasYaw)
    {
        const double magYawRad = mExtRefYawDeg * M_PI / 180.0;
        // target = offset - heading (clockwise heading) or offset + heading (legacy)
        mExtMagYawOffsetRad = mbExtMagYawClockwise ? slamYaw + magYawRad : slamYaw - magYawRad;
        latched = true;
    }
    if(mbExtRefHasAlt)
    {
        mExtBaroAltOffsetM = slamZ - mExtRefAltM;
        latched = true;
    }
    if(latched)
    {
        mExtRefLatched = true;
        mExtRefLatchInitKF = mpAtlas->GetCurrentMap()->GetInitKFid();
        cout << "External ref latched: magYawOffsetRad=" << mExtMagYawOffsetRad
             << ", baroAltOffsetM=" << mExtBaroAltOffsetM << endl;
    }
}

void Tracking::SetKeyFrameExternalRefs(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutexExtRef);

    // Re-latch if the active map changed (reset/relocalization re-zeroes the frame)
    if(mExtRefLatched && mExtRefLatchInitKF != mpAtlas->GetCurrentMap()->GetInitKFid())
        mExtRefLatched = false;
    LatchExternalRef();

    if(!mExtRefLatched)
    {
        pKF->ClearExternalYaw();
        pKF->ClearExternalAlt();
        return;
    }

    if(mbExtRefHasYaw)
    {
        const double headingRad = mExtRefYawDeg * M_PI / 180.0;
        const double targetYaw = mbExtMagYawClockwise ? mExtMagYawOffsetRad - headingRad
                                                      : headingRad + mExtMagYawOffsetRad;
        pKF->SetExternalYaw(targetYaw, mExtMagYawSigmaDeg * M_PI / 180.0);
    }
    else
    {
        pKF->ClearExternalYaw();
    }

    if(mbExtRefHasAlt)
    {
        const double targetAlt = mExtRefAltM + mExtBaroAltOffsetM;
        pKF->SetExternalAlt(targetAlt, mExtBaroAltSigmaM);
    }
    else
    {
        pKF->ClearExternalAlt();
    }
}

void Tracking::CreateNewKeyFrame()
{
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    SetKeyFrameExternalRefs(pKF);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    Eigen::Vector3f x3D;

                    if(mCurrentFrame.Nleft == -1){
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }
            //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
    }


    mpLocalMapper->InsertKeyFrame(pKF);
    {
        // WP A1 deterministic replay (FASTTRACK_DETERMINISTIC=1): wait for local mapping to finish this keyframe.
        static const bool bDet = [](){ const char* e = std::getenv("FASTTRACK_DETERMINISTIC");
                                       return e && std::string(e) == "1"; }();
        if(bDet)
        {
            const auto t0 = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            while((mpLocalMapper->KeyframesInQueue() > 0 || !mpLocalMapper->AcceptKeyFrames()) &&
                  std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5))
                std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
#ifdef REGISTER_STATS  
    std::chrono::steady_clock::time_point first_itr_start = std::chrono::steady_clock::now();
#endif
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

#ifdef REGISTER_STATS  
    std::chrono::steady_clock::time_point first_itr_end = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point second_itr_start = std::chrono::steady_clock::now();
#endif
    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
        if(pMP->mbTrackInView)
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }
#ifdef REGISTER_STATS  
    std::chrono::steady_clock::time_point second_itr_end = std::chrono::steady_clock::now();
#endif
    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            th=3;
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=6;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15;

        // printf("===== F: %lu, num mappoints: %d\n", mCurrentFrame.mnId, mvpLocalMapPoints.size());
#ifdef REGISTER_STATS  
    std::chrono::steady_clock::time_point SBP_start = std::chrono::steady_clock::now();
#endif
        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
#ifdef REGISTER_STATS  
    std::chrono::steady_clock::time_point SBP_end = std::chrono::steady_clock::now();
    double searchByProjection = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(SBP_end - SBP_start).count();
    TrackingStats::getInstance().SLP_searchByProjection_time.emplace_back(mCurrentFrame.mnId, searchByProjection);
    TrackingStats::getInstance().num_local_mappoints.emplace_back(mCurrentFrame.mnId, mvpLocalMapPoints.size());
#endif
    }
#ifdef REGISTER_STATS  
    double firstItr = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(first_itr_end - first_itr_start).count();
    double secondItr = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(second_itr_end - second_itr_start).count();
    TrackingStats::getInstance().SLP_frameMapPointsItr_time.emplace_back(mCurrentFrame.mnId, firstItr);
    TrackingStats::getInstance().SLP_localMapPointsItr_time.emplace_back(mCurrentFrame.mnId, secondItr);
#endif
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
#ifdef REGISTER_STATS  
    std::chrono::steady_clock::time_point ULK_start = std::chrono::steady_clock::now();
#endif
    UpdateLocalKeyFrames();
#ifdef REGISTER_STATS  
    std::chrono::steady_clock::time_point ULK_end = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point ULP_start = std::chrono::steady_clock::now();
#endif
    UpdateLocalPoints();
#ifdef REGISTER_STATS  
    std::chrono::steady_clock::time_point ULP_end = std::chrono::steady_clock::now();
    double updateLocalKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(ULK_end - ULK_start).count();
    double updateLocalPoints = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(ULP_end - ULP_start).count();
    TrackingStats::getInstance().updateLocalKF_time.emplace_back(mCurrentFrame.mnId, updateLocalKF);
    TrackingStats::getInstance().updateLocalPoints_time.emplace_back(mCurrentFrame.mnId, updateLocalPoints);
#endif
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    int continue_counts = 0;

    int localKeyFrames_size = mvpLocalKeyFrames.size();

    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF;
        count_pts = 0;
        continue_counts = 0;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {

            MapPoint* pMP = *itMP;
            if(!pMP) {
                continue;
            }
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId){
                continue_counts += 1;
                continue;
            }
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    auto ULK_findKFs_start = std::chrono::system_clock::now();
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    auto ULK_findKFs_end = std::chrono::system_clock::now();
    std::chrono::duration<double>  ULK_findKFs_elapsed_seconds = ULK_findKFs_end - ULK_findKFs_start;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    auto ULK_addKFs_start = std::chrono::system_clock::now();
    int k[5] = {0, 0, 0, 0, 0};
    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        k[0]+=1;
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 80
            break;

        KeyFrame* pKF = *itKF;
        
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&mvpLocalKeyFrames.size()<80)
    {
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
    auto ULK_addKFs_end = std::chrono::system_clock::now();
    std::chrono::duration<double>  ULK_addKFs_elapsed_seconds = ULK_addKFs_end - ULK_addKFs_start;
    std::ofstream myfile;

    // myfile.open("mvpLocalKeyFrames_size.csv", std::fstream::app);
    // myfile << mvpLocalKeyFrames.size() << "\n";
    // myfile.close();
}

bool Tracking::Relocalization(Map* pTargetMap)
{
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#endif    

    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    if(!pTargetMap)
        pTargetMap = mpAtlas->GetCurrentMap();
    mnRelocBestBowMatches = 0;
    mnRelocInliers = 0;
    // Inliers needed: 50 (upstream); into the loaded prior map PriorMapOptions::reloc_min_inliers, compensated by a longer
    // confirmation (PriorMapStep). 2026-10-22 pad-board replays: best attempts 36 inliers, most 0.
    const int nNeed = (pTargetMap && pTargetMap == mpPriorMap) ? std::max(10, mPriorOpt.reloc_min_inliers) : 50;
    mnRelocCandidates = 0; mnRelocBestPnpInliers = 0; mnRelocBestGood = 0;
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, pTargetMap);

    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double relocalization = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    TrackingStats::getInstance().relocalization_time.emplace_back(mCurrentFrame.mnId, relocalization);
#endif
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);
    matcher.mbMutualCheck = (pTargetMap && pTargetMap == mpPriorMap) && mPriorOpt.reloc_mutual_check;

    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            mnRelocBestBowMatches = std::max(mnRelocBestBowMatches, nmatches);
            mnRelocCandidates++;
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw);
            if(bTcw) mnRelocBestPnpInliers = std::max(mnRelocBestPnpInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(bTcw)
            {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw);
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                mnRelocBestGood = std::max(mnRelocBestGood, nGood);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<nNeed)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=nNeed)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>(nNeed*3)/5 && nGood<nNeed)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=nNeed)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=nNeed)
                {
                    mnRelocInliers = nGood;
                    bMatch = true;
                    break;
                }
            }
        }
    }
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double relocalization = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(end - start).count();
    TrackingStats::getInstance().relocalization_time.emplace_back(mCurrentFrame.mnId, relocalization);
#endif

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        FT_CALC_LOG("tracking.relocalization.success frame=%llu",
                    static_cast<unsigned long long>(mCurrentFrame.mnId));
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("tracking.reset.begin local_mapping_reset=%d", bLocMap ? 0 : 1);
    PriorMapNoteMapReset(nullptr, "full reset (all maps cleared)");
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    // Prior map active: a session reset keeps it (PriorMap.cc). Only if that fails is the prior map cleared.
    if(mpPriorMap && mpAtlas->GetCurrentMap() == mpPriorMap && HasUsablePriorMap())
    {
        std::string why;
        if(ResetToPriorMap(why))
            return;
        PriorMapNoteMapReset(mpPriorMap, ("active map reset; keeping the prior map failed: " + why).c_str());
    }
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    FT_CALC_LOG("tracking.reset.trajectory first_frame_id=%u", mnFirstFrameId);
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    FT_CALC_LOG("tracking.reset.trajectory initial_frame_id=%u", mnInitialFrameId);

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    FT_CALC_LOG("tracking.reset.trajectory frames_set_lost=%d", num_lost);
    TrackingStats::getInstance().num_frames_lost = num_lost;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    FT_CALC_LOG("tracking.reset.end");
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        // RESEARCH culling-v4b-r2 (release rev2): a culled reference may legitimately have no live
        // parent; the old parent walk dereferenced NULL here (rev1 crash, VIBA2 UpdateFrameIMU).
        // Resolve exactly as System::SaveTrajectory* does and leave unresolvable entries unscaled.
        Sophus::SE3f TReferenceToRoot;
        if(!pKF || !pKF->ResolveTrajectoryReference(pKF, TReferenceToRoot))
            continue;

        if(pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    // UFI: after a re-initialisation the last frame may have no preintegration / last keyframe (crash 2026-10-27).
    if(!mLastFrame.mpLastKeyFrame || (mLastFrame.mnId != mLastFrame.mpLastKeyFrame->mnFrameId && !mLastFrame.mpImuPreintegrated))
    {
        // nothing to refresh for the last frame
    }
    else if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated && mCurrentFrame.mpLastKeyFrame)  // UFI
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        FT_CALC_LOG("tracking.worker.stop");
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

void Tracking::AddAltitudeReference(double t, double h)
{
    if(!std::isfinite(t) || !std::isfinite(h))
        return;
    unique_lock<mutex> lock(mMutexAltRef);
    if(!mdAltRef.empty() && t <= mdAltRef.back().first)
        return;
    mdAltRef.emplace_back(t, h);
    while(!mdAltRef.empty() && t - mdAltRef.front().first > 120.0)
        mdAltRef.pop_front();
    if(!mbAltPad)
    {
        if(mAltFirstT < 0) mAltFirstT = t;
        mvAltFirst.push_back(h);
        if(t - mAltFirstT >= 3.0 && mvAltFirst.size() >= 5)
        {
            std::vector<double> v = mvAltFirst;
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            mAltPad = v[v.size() / 2]; mbAltPad = true;
            cerr << "FastTrack metric init: pad altitude reference " << mAltPad << " m from " << v.size() << " samples" << endl;
        }
    }
}

bool Tracking::AltitudeAt(double t, double &h, double &hPad)
{
    unique_lock<mutex> lock(mMutexAltRef);
    if(!mbAltPad || mdAltRef.size() < 2)
        return false;
    auto it = std::lower_bound(mdAltRef.begin(), mdAltRef.end(), t,
        [](const std::pair<double,double> &a, double v) { return a.first < v; });
    if(it == mdAltRef.begin() || it == mdAltRef.end())
        return false;
    auto ip = std::prev(it);
    if(it->first - ip->first > 0.5)
        return false;
    const double w = (t - ip->first) / (it->first - ip->first);
    h = ip->second + w * (it->second - ip->second);
    hPad = mAltPad;
    return true;
}

bool Tracking::ClimbRateAbove(double t, double thr, double &rate)
{
    rate = 0.0;
    double h1, h0, hPad;
    if(!AltitudeAt(t, h1, hPad) || !AltitudeAt(t - 1.0, h0, hPad))
        return false;
    rate = h1 - h0;
    return std::fabs(rate) > thr;
}

bool Tracking::ClimbingAt(double t, double &rate)
{
    rate = 0.0;
    if(!(mDeferClimbRate > 0.0))
        return false;
    double h1, h0, hPad;
    if(!AltitudeAt(t, h1, hPad) || !AltitudeAt(t - 1.0, h0, hPad))
        return false;
    rate = h1 - h0;
    return std::fabs(rate) > mDeferClimbRate;
}

float Tracking::MetricInitScale(KeyFrame* pKFini, KeyFrame* pKFcur)
{
    double h = 0.0, hPad = 0.0;
    if(!AltitudeAt(pKFcur->mTimeStamp, h, hPad))
    {
        cerr << "FastTrack metric init: no altitude at t=" << std::fixed << pKFcur->mTimeStamp << "; scale left to the IMU" << endl;
        return -1.f;
    }
    const double hAbove = h - hPad;
    const double hCam = hAbove + mMetricPadCamHeight;
    if(hAbove < mMetricMinHeight)
    {
        cerr << "FastTrack metric init: height " << hAbove << " m < " << mMetricMinHeight << " m; scale left to the IMU" << endl;
        return -1.f;
    }
    std::vector<Eigen::Vector3f> P;
    for(MapPoint* pMP : pKFcur->GetMapPointMatches())
        if(pMP && !pMP->isBad())
            P.push_back(pMP->GetWorldPos());
    const int n = (int)P.size();
    if(n < 30)
    {
        cerr << "FastTrack metric init: " << n << " map points, too few for the ground plane" << endl;
        return -1.f;
    }
    // Scale of the unscaled map: median distance of the points to the current camera.
    const Eigen::Vector3f Oc = pKFcur->GetCameraCenter();
    std::vector<float> dists; for(auto &p : P) dists.push_back((p - Oc).norm());
    std::nth_element(dists.begin(), dists.begin() + n / 2, dists.end());
    const float med = dists[n / 2];
    const float tau = 0.03f * med;
    // RANSAC plane (deterministic seed for repeatable replays)
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> U(0, n - 1);
    int bestInl = 0; Eigen::Vector3f bestN(0, 0, 1); float bestC = 0.f;
    for(int it = 0; it < 300; it++)
    {
        const Eigen::Vector3f &a = P[U(rng)], &b = P[U(rng)], &c = P[U(rng)];
        Eigen::Vector3f nrm = (b - a).cross(c - a);
        if(nrm.norm() < 1e-9f) continue;
        nrm.normalize(); const float cc = -nrm.dot(a);
        int inl = 0; for(auto &p : P) if(std::fabs(nrm.dot(p) + cc) < tau) inl++;
        if(inl > bestInl) { bestInl = inl; bestN = nrm; bestC = cc; }
    }
    if(bestInl < 0.6 * n)
    {
        cerr << "FastTrack metric init: ground plane inliers " << bestInl << "/" << n << " < 60%; scale left to the IMU" << endl;
        return -1.f;
    }
    // Least-squares refinement on the inliers
    Eigen::Vector3f mean = Eigen::Vector3f::Zero(); int m = 0;
    for(auto &p : P) if(std::fabs(bestN.dot(p) + bestC) < tau) { mean += p; m++; }
    mean /= (float)m;
    Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
    for(auto &p : P) if(std::fabs(bestN.dot(p) + bestC) < tau) { Eigen::Vector3f d = p - mean; C += d * d.transpose(); }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
    Eigen::Vector3f nrm = es.eigenvectors().col(0).normalized();
    const float cc = -nrm.dot(mean);
    const float dVis = std::fabs(nrm.dot(Oc) + cc);
    // The camera should look roughly along the plane normal (downward camera over the ground).
    const Eigen::Vector3f zc = pKFcur->GetRotation().transpose() * Eigen::Vector3f(0, 0, 1);
    const float cosView = std::fabs(zc.dot(nrm));
    if(!(dVis > 1e-6f) || cosView < std::cos(35.f * (float)M_PI / 180.f))
    {
        cerr << "FastTrack metric init: plane distance " << dVis << ", view angle cos " << cosView << "; scale left to the IMU" << endl;
        return -1.f;
    }
    const float s = (float)(hCam / dVis);
    cerr << "FastTrack metric init: height above pad " << hAbove << " m + pad camera " << mMetricPadCamHeight
         << " m; plane inliers " << m << "/" << n << ", view cos " << cosView << ", map distance " << dVis
         << " -> scale " << s << " (median depth " << med * s << " m)" << endl;
    return s;
}

static bool FitGroundPlane(const std::vector<Eigen::Vector3f> &P, float tau, Eigen::Vector3f &nrm, float &cc, int &nInl)
{
    const int n = (int)P.size();
    if(n < 30) return false;
    std::mt19937 rng(54321);
    std::uniform_int_distribution<int> U(0, n - 1);
    int best = 0; Eigen::Vector3f bN(0, 0, 1); float bC = 0.f;
    for(int it = 0; it < 300; it++)
    {
        const Eigen::Vector3f &a = P[U(rng)], &b = P[U(rng)], &c = P[U(rng)];
        Eigen::Vector3f q = (b - a).cross(c - a);
        if(q.norm() < 1e-9f) continue;
        q.normalize(); const float d = -q.dot(a);
        int k = 0; for(auto &p : P) if(std::fabs(q.dot(p) + d) < tau) k++;
        if(k > best) { best = k; bN = q; bC = d; }
    }
    if(best < 0.5 * n) return false;
    Eigen::Vector3f mean = Eigen::Vector3f::Zero(); int m = 0;
    for(auto &p : P) if(std::fabs(bN.dot(p) + bC) < tau) { mean += p; m++; }
    mean /= (float)m;
    Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
    for(auto &p : P) if(std::fabs(bN.dot(p) + bC) < tau) { Eigen::Vector3f d = p - mean; C += d * d.transpose(); }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
    nrm = es.eigenvectors().col(0).normalized(); cc = -nrm.dot(mean); nInl = m;
    return true;
}

bool Tracking::BaroScaleFit(Map* pMap, const std::vector<KeyFrame*> &vpKF, double &s, double &c, int &n, double &range,
                            double &rms)
{
    n = 0; range = 0; rms = 0;
    std::vector<Eigen::Vector3f> P;
    for(MapPoint* pMP : pMap->GetAllMapPoints())
        if(pMP && !pMP->isBad()) P.push_back(pMP->GetWorldPos());
    if(P.size() < 30) return false;
    // tau relative to the map size: median keyframe-to-point distance
    std::vector<std::pair<double,double>> dh;  // (d_i, h_i)
    std::vector<float> depth;
    for(KeyFrame* pKF : vpKF) if(pKF && !pKF->isBad()) depth.push_back((P[P.size() / 2] - pKF->GetCameraCenter()).norm());
    if(depth.empty()) return false;
    std::nth_element(depth.begin(), depth.begin() + depth.size() / 2, depth.end());
    Eigen::Vector3f nrm; float cc; int nInl;
    if(!FitGroundPlane(P, 0.05f * depth[depth.size() / 2], nrm, cc, nInl)) return false;
    double hmin = 1e9, hmax = -1e9;
    for(KeyFrame* pKF : vpKF)
    {
        if(!pKF || pKF->isBad()) continue;
        double h, hPad;
        if(!AltitudeAt(pKF->mTimeStamp, h, hPad)) continue;
        const double d = std::fabs(nrm.dot(pKF->GetCameraCenter()) + cc);
        dh.emplace_back(d, h - hPad); hmin = std::min(hmin, h - hPad); hmax = std::max(hmax, h - hPad);
    }
    n = (int)dh.size(); range = hmax - hmin;
    if(n < 6 || range < 3.0) return false;
    // s*d - c = h, two passes (second without residuals > 3x rms)
    std::vector<bool> use(n, true);
    for(int pass = 0; pass < 2; pass++)
    {
        double Sdd = 0, Sd = 0, S1 = 0, Sdh = 0, Sh = 0;
        for(int i = 0; i < n; i++) if(use[i]) { const double d = dh[i].first, h = dh[i].second; Sdd += d * d; Sd += d; S1 += 1; Sdh += d * h; Sh += h; }
        // normal equations for [s, -c]
        const double det = Sdd * S1 - Sd * Sd;
        if(S1 < 6 || std::fabs(det) < 1e-12) return false;
        s = (Sdh * S1 - Sd * Sh) / det;
        const double negc = (Sdd * Sh - Sd * Sdh) / det;
        c = -negc;
        double e2 = 0; int k = 0;
        for(int i = 0; i < n; i++) if(use[i]) { const double r = s * dh[i].first - c - dh[i].second; e2 += r * r; k++; }
        rms = std::sqrt(e2 / std::max(1, k));
        if(pass == 0)
            for(int i = 0; i < n; i++) { const double r = s * dh[i].first - c - dh[i].second; if(std::fabs(r) > 3 * rms + 0.2) use[i] = false; }
    }
    return std::isfinite(s) && s > 0;
}

bool Tracking::LocalGroundDistance(KeyFrame* pKF, double &d, double &tiltDeg, int &nInl, int &nPts)
{
    nInl = 0; nPts = 0;
    if(!pKF || pKF->isBad()) return false;
    const Eigen::Vector3f O = pKF->GetCameraCenter();
    std::vector<Eigen::Vector3f> P; std::vector<float> depth;
    for(MapPoint* pMP : pKF->GetMapPointMatches())
        if(pMP && !pMP->isBad()) { P.push_back(pMP->GetWorldPos()); depth.push_back((P.back() - O).norm()); }
    nPts = (int)P.size();
    if(nPts < 30) return false;
    std::nth_element(depth.begin(), depth.begin() + depth.size() / 2, depth.end());
    Eigen::Vector3f nrm; float cc;
    if(!FitGroundPlane(P, 0.03f * depth[depth.size() / 2], nrm, cc, nInl)) return false;
    d = std::fabs(nrm.dot(O) + cc);
    tiltDeg = std::acos(std::min(1.f, std::fabs(nrm(2)))) * 180.0 / M_PI;
    return true;
}

void Tracking::KltAssociate()
{
    if(mKltPrevImg.empty() || mImGray.empty() || mKltPrevImg.size() != mImGray.size() || mImGray.type() != CV_8UC1)
        return;
    if(mLastFrame.N <= 0 || mLastFrame.mvKeys.size() != (size_t)mLastFrame.N || mCurrentFrame.N <= 0)
        return;
    const Sophus::SE3f Tcw = mCurrentFrame.GetPose();
    const Sophus::SE3f TcwLast = mLastFrame.GetPose();
    std::vector<int> idx;
    std::vector<cv::Point2f> p0, p1;
    for(int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP || mLastFrame.mvbOutlier[i] || pMP->isBad())
            continue;
        const cv::Point2f last = mLastFrame.mvKeys[i].pt;
        cv::Point2f guess = last;
        // Initial guess: last position shifted by the predicted (undistorted) image motion of the point.
        const Eigen::Vector3f Pw = pMP->GetWorldPos();
        const Eigen::Vector3f Pc = Tcw * Pw, PcL = TcwLast * Pw;
        if(Pc(2) > 0.f && PcL(2) > 0.f)
        {
            const Eigen::Vector2f u = mpCamera->project(Pc), uL = mpCamera->project(PcL);
            const Eigen::Vector2f d = u - uL;
            if(d.allFinite() && d.norm() < 200.f)
                guess = last + cv::Point2f(d(0), d(1));
        }
        idx.push_back(i); p0.push_back(last); p1.push_back(guess);
    }
    mKltStatFrames++;
    mKltStatTried += idx.size();
    if(idx.size() < 8)
        return;
    std::vector<uchar> st, stb;
    std::vector<float> err, errb;
    const cv::Size win(mKltWin, mKltWin);
    const cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(mKltPrevImg, mImGray, p0, p1, st, err, win, mKltLevels, tc, cv::OPTFLOW_USE_INITIAL_FLOW);
    std::vector<cv::Point2f> pb = p0;
    cv::calcOpticalFlowPyrLK(mImGray, mKltPrevImg, p1, pb, stb, errb, win, mKltLevels, tc, cv::OPTFLOW_USE_INITIAL_FLOW);
    std::vector<int> good;
    std::vector<cv::Point2f> g0, g1;
    for(size_t k = 0; k < idx.size(); k++)
    {
        if(!st[k] || !stb[k]) continue;
        const cv::Point2f e = pb[k] - p0[k];
        if(e.x * e.x + e.y * e.y > mKltMaxFb * mKltMaxFb) continue;
        if(p1[k].x < 0 || p1[k].y < 0 || p1[k].x >= mImGray.cols || p1[k].y >= mImGray.rows) continue;
        good.push_back((int)k); g0.push_back(p0[k]); g1.push_back(p1[k]);
    }
    if(g0.size() >= 15)
    {
        std::vector<uchar> inl;
        cv::findFundamentalMat(g0, g1, cv::FM_RANSAC, 1.5, 0.99, inl);
        if(inl.size() == g0.size())
        {
            std::vector<int> g2; std::vector<cv::Point2f> h1;
            for(size_t k = 0; k < good.size(); k++)
                if(inl[k]) { g2.push_back(good[k]); h1.push_back(g1[k]); }
            good.swap(g2); g1.swap(h1);
        }
    }
    mKltStatTracked += good.size();
    if(good.empty()) return;
    // Tracked positions are in the distorted image; the frame grid uses undistorted keypoints.
    std::vector<cv::Point2f> un;
    cv::undistortPoints(g1, un, mCurrentFrame.mK, mCurrentFrame.mDistCoef, cv::noArray(), mCurrentFrame.mK);
    int nAssoc = 0, nAppend = 0;
    for(size_t k = 0; k < good.size(); k++)
    {
        const int i = idx[good[k]];
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        const int lvl = mLastFrame.mvKeysUn[i].octave;
        const std::vector<size_t> cand = mCurrentFrame.GetFeaturesInArea(un[k].x, un[k].y, mKltRadius,
                                                                           std::max(0, lvl - 1), lvl + 1);
        const cv::Mat dMP = pMP->GetDescriptor();
        int best = -1, bestDist = mKltMaxHamming + 1;
        for(size_t c : cand)
        {
            if(mCurrentFrame.mvpMapPoints[c]) continue;
            const int dist = ORBmatcher::DescriptorDistance(dMP, mCurrentFrame.mDescriptors.row(c));
            if(dist < bestDist) { bestDist = dist; best = (int)c; }
        }
        if(best >= 0)
        {
            mCurrentFrame.mvpMapPoints[best] = pMP;
            nAssoc++;
        }
        else if(mbKltPersistent && nAppend < mKltMaxAppend)
        {
            // C1b (2026-10-31): no ORB feature here -> the tracked position itself becomes the observation.
            const cv::KeyPoint &kl = mLastFrame.mvKeys[i];
            cv::KeyPoint kp(g1[k], kl.size, kl.angle, kl.response, kl.octave, kl.class_id);
            cv::KeyPoint kpu(un[k], kl.size, kl.angle, kl.response, kl.octave, kl.class_id);
            if(mCurrentFrame.AppendObservation(kp, kpu, dMP, pMP))
                nAppend++;
        }
    }
    mKltStatAssoc += nAssoc;
    mKltStatAppended += nAppend;
    if(mKltStatFrames % 300 == 0)
        cout << "KLT: frames " << mKltStatFrames << " tried " << mKltStatTried << " tracked " << mKltStatTracked
             << " associated " << mKltStatAssoc << " appended " << mKltStatAppended
             << " (last frame " << nAssoc << "+" << nAppend << "/" << idx.size() << ")" << endl;
}

// C7c (2026-10-31): baro-inertial vertical KF, x = [h, v_z, b], a_true = a_meas - b.
void Tracking::VertKFPredict(double az, double dt)
{
    if(!(dt > 0.0) || dt > 0.1 || !std::isfinite(az))
        return;
    unique_lock<mutex> lk(mMutexVkf);
    double (&x)[3] = mVkfX; double (&P)[3][3] = mVkfP;
    const double a = az - x[2];
    x[0] += x[1] * dt + 0.5 * a * dt * dt;
    x[1] += a * dt;
    const double F[3][3] = {{1, dt, -0.5 * dt * dt}, {0, 1, -dt}, {0, 0, 1}};
    double FP[3][3], Pn[3][3];
    for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) { FP[i][j] = 0; for(int k = 0; k < 3; k++) FP[i][j] += F[i][k] * P[k][j]; }
    for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) { Pn[i][j] = 0; for(int k = 0; k < 3; k++) Pn[i][j] += FP[i][k] * F[j][k]; }
    const double qa = mVkfAccNoise * mVkfAccNoise * dt, qb = mVkfBiasRW * mVkfBiasRW * dt;
    Pn[0][0] += 0.25 * dt * dt * qa; Pn[0][1] += 0.5 * dt * qa; Pn[1][0] += 0.5 * dt * qa; Pn[1][1] += qa; Pn[2][2] += qb;
    for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) P[i][j] = Pn[i][j];
}

void Tracking::VertKFBaro(double t)
{
    double h = 0, hPad = 0, tb = 0;
    if(!LatestAltitude(t, tb, h, hPad))
        return;
    unique_lock<mutex> lk(mMutexVkf);
    if(tb <= mVkfLastBaroT)
        return;  // no new baro sample since the last update
    mVkfLastBaroT = tb;
    double (&x)[3] = mVkfX; double (&P)[3][3] = mVkfP;
    // the state is at t, the sample at tb <= t: move the sample to t with the KF vertical velocity
    const double y = (h - hPad + x[1] * (t - tb)) - x[0], S = P[0][0] + mVkfBaroNoise * mVkfBaroNoise;
    if(!(S > 0.0) || !std::isfinite(y))
        return;
    const double K[3] = {P[0][0] / S, P[1][0] / S, P[2][0] / S};
    for(int i = 0; i < 3; i++) x[i] += K[i] * y;
    double Pn[3][3];
    for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) Pn[i][j] = P[i][j] - K[i] * P[0][j];
    for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) P[i][j] = Pn[i][j];
}

bool Tracking::GetVertKF(double &b, double &sd, double &h, double &vz)
{
    unique_lock<mutex> lk(mMutexVkf);
    if(!mVkfActive)
        return false;
    b = mVkfX[2]; sd = std::sqrt(std::max(0.0, mVkfP[2][2])); h = mVkfX[0]; vz = mVkfX[1];
    return true;
}

bool Tracking::LatestAltitude(double tNow, double &tSample, double &h, double &hPad)
{
    unique_lock<mutex> lock(mMutexAltRef);
    if(!mbAltPad || mdAltRef.empty())
        return false;
    auto it = std::upper_bound(mdAltRef.begin(), mdAltRef.end(), tNow,
        [](double v, const std::pair<double,double> &a) { return v < a.first; });
    if(it == mdAltRef.begin())
        return false;
    --it;  // newest sample at or before tNow
    if(tNow - it->first > 0.5)
        return false;
    tSample = it->first; h = it->second; hPad = mAltPad;
    return true;
}

// C8 (2026-10-31): unit normal (pointing to +z) of the keyframe's local ground plane in the map frame.
bool Tracking::LocalGroundNormal(KeyFrame* pKF, Eigen::Vector3f &nrmOut, int &nInl, int &nPts)
{
    nInl = 0; nPts = 0;
    if(!pKF || pKF->isBad()) return false;
    const Eigen::Vector3f O = pKF->GetCameraCenter();
    std::vector<Eigen::Vector3f> P; std::vector<float> depth;
    for(MapPoint* pMP : pKF->GetMapPointMatches())
        if(pMP && !pMP->isBad()) { P.push_back(pMP->GetWorldPos()); depth.push_back((P.back() - O).norm()); }
    nPts = (int)P.size();
    if(nPts < 30) return false;
    std::nth_element(depth.begin(), depth.begin() + depth.size() / 2, depth.end());
    Eigen::Vector3f nrm; float cc;
    if(!FitGroundPlane(P, 0.03f * depth[depth.size() / 2], nrm, cc, nInl)) return false;
    if(nrm(2) < 0.f) nrm = -nrm;
    nrmOut = nrm.normalized();
    return true;
}

} //namespace ORB_SLAM
