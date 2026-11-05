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


#ifndef SYSTEM_H
#define SYSTEM_H


#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<thread>
#include<opencv2/core/core.hpp>
#include <Eigen/Core>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"
#include "ImuTypes.h"
#include "Settings.h"
#include "Stats/TrackingStats.h"
#include "CalculationLogApi.h"
#include "PriorMap.h"


namespace ORB_SLAM3
{

class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_QUIET=0,
        VERBOSITY_NORMAL=1,
        VERBOSITY_VERBOSE=2,
        VERBOSITY_VERY_VERBOSE=3,
        VERBOSITY_DEBUG=4
    };

    static eLevel th;

public:
    static void PrintMess(std::string str, eLevel lev)
    {
        if(lev <= th)
            CalculationLog::text(str.c_str());
    }

    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

class Viewer;
class FrameDrawer;
class MapDrawer;
class Atlas;
class Tracking;
class LocalMapping;
class LoopClosing;
class Settings;

struct PreparedMonocularFrame;   // defined in Tracking.h

class System
{
public:
    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2,
        IMU_MONOCULAR=3,
        IMU_STEREO=4,
        IMU_RGBD=5,
    };

    // File type
    enum FileType{
        TEXT_FILE=0,
        BINARY_FILE=1,
    };

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true, const int initFr = 0, const string &strSequence = std::string());
    // Same, with prior-map start / map save options (PriorMap.h). PriorMapOptions() reproduces the constructor above.
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const PriorMapOptions &prior,
           const bool bUseViewer = true, const int initFr = 0, const string &strSequence = std::string());

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // Proccess the given monocular frame and optionally imu data
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // Pipelined monocular input. TrackMonocular() == PrepareMonocular() + TrackPreparedMonocular().
    // PrepareMonocular() (ORB extraction only) may run on another thread ahead of tracking, so the
    // extraction of frame N+1 overlaps Track() of frame N. TrackPreparedMonocular() must be called
    // from the tracking caller, in timestamp order. Returns the same poses as TrackMonocular().
    std::shared_ptr<PreparedMonocularFrame> PrepareMonocular(const cv::Mat &im, const double &timestamp, string filename="");
    Sophus::SE3f TrackPreparedMonocular(const std::shared_ptr<PreparedMonocularFrame> &prepared, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>());
    unsigned long PipelineReextractions() const;

    // Inject an external absolute reference into the running SLAM system
    // (magnetometer/compass heading in deg and barometric altitude in m).
    // When both flags are false the external-reference path stays inert.
    void SetExternalReference(const bool bHasYaw, const double yawDeg, const bool bHasAlt, const double altM);

    // Gyro bias measured while the vehicle was verifiably at rest (rad/s, IMU body frame, w_true = w_meas - bg),
    // used as the gyro-bias prior (1/sigma^2) by IMU initialization, BA1 and BA2 (LocalMapping::SetRestGyroBiasPrior).
    // Ignored unless sigma > 0 and bg is finite. ClearRestGyroBiasPrior restores the default priors.
    void SetRestGyroBiasPrior(const Eigen::Vector3f &bg, const float sigma);
    // Flight-controller attitude (world z up, IMU body frame) for IMU initialization gravity (LocalMapping).
    void AddGravityReference(double t, const Eigen::Matrix3f &Rwb);
    // B3 metric initialization: barometric height samples in the IMU clock.
    void AddAltitudeReference(double t, double h);
    void SetGravityFromReference(bool b);
    void ClearRestGyroBiasPrior();
    // Init profile (2026-10-26): accelerometer-bias prior from a saved initialization (LocalMapping::SetAccBiasPrior).
    void SetAccBiasPrior(const Eigen::Vector3f &ba, const float sigma);
    void ClearAccBiasPrior();
    // Converged IMU biases of the last keyframe of the active map, for saving an init profile. False unless the map
    // is IMU-initialized and inertial BA2 is done. nKF = keyframes in the map.
    bool GetInitProfileState(Eigen::Vector3f &ba, Eigen::Vector3f &bg, int &nKF);
    // F1 (2026-10-27): last keyframe barometric scale observation (h = baro height above the pad, d = camera to the
    // local ground plane, s = (h + c) / d, z = camera z, tilt = plane tilt deg, nI/nP = plane inliers/points).
    // seq increases with every observation. False before the first one.
    // P1: per-frame pose optimization mode (Tracking::SetPoseOptimizationMode). Call from the tracking thread.
    void SetPoseOptimizationMode(int mode);
    void SetPoseOptimizationTurn(double rate_dps, double hold_s);  // P3: modes 3/4
    bool GetBaroScaleObservation(double &t, double &h, double &d, double &s, double &z, double &tilt, int &nI,
                                 int &nP, unsigned long &mapId, unsigned long &seq);


    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged();

    // Reset the system (clear Atlas or the active map)
    void Reset();
    void ResetActiveMap();

    // Prior-map start. ResetToPriorMap: session reset that keeps the prior map (executed in the next Track call;
    // without a usable prior map it is a full Reset()). SaveMap: call from the thread that calls Track*, between
    // frames; it holds LocalMapping and waits for LoopClosing (and a running global BA, up to gba_wait_s).
    void ResetToPriorMap();
    bool HasUsablePriorMap();
    // Call from the thread that calls Track*, before the frame after a known pause (e.g. SaveMap).
    void AllowTimeGapOnce(double gapS);
    PriorMapStatus GetPriorMapStatus();
    SaveMapResult SaveMap(const std::string &path = std::string());

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown(); 
    bool isShutDown();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    void SaveTrajectoryEuRoC(const string &filename);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename);

    void SaveTrajectoryEuRoC(const string &filename, Map* pMap);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename, Map* pMap);

    // Save data used for initialization debug
    void SaveDebugData(const int &iniIdx);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();

    // Read-only snapshot of the active map's frame identity for consumers that
    // must know when the world frame is gravity-aligned/metric and when it can
    // have changed (new map, IMU (re)initialisation, loop closure, merge).
    struct ActiveMapFrameInfo
    {
        bool valid = false;
        unsigned long map_id = 0;
        bool imu_initialized = false;
        bool inertial_ba1 = false;
        bool inertial_ba2 = false;
        int change_index = 0;          // Map::GetMapChangeIndex()
        int change_index_notified = 0; // Map::GetLastMapChange()
        // RESEARCH frame-epoch contract
        unsigned long frame_epoch = 0;       // active map's current epoch
        bool pose_epoch_recorded = false;
        unsigned long pose_map_id = 0;       // map of the last tracked pose
        unsigned long pose_frame_epoch = 0;  // epoch of the last tracked pose
        // G1 (2026-10-31): accelerometer bias of the last tracked frame (valid once the IMU is initialized)
        bool acc_bias_valid = false;
        float acc_bias[3] = {0.f, 0.f, 0.f};
    };
    ActiveMapFrameInfo GetActiveMapFrameInfo();
    // History (bounded) of the active map's whole-map similarity transforms.
    std::vector<Map::FrameTransform> GetActiveMapFrameTransforms();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

    // For debugging
    double GetTimeFromIMUInit();
    bool isLost();
    bool isFinished();

    void ChangeDataset();

    float GetImageScale();

#ifdef REGISTER_TIMES
    void InsertRectTime(double& time);
    void InsertResizeTime(double& time);
    void InsertTrackTime(double& time);
#endif

private:

    void SaveAtlas(int type);
    bool LoadAtlas(int type);
    bool LoadAtlasFile(const std::string &path, PriorMapStatus &st);
    SaveMapResult WriteAtlasFile(const std::string &path);
    PriorMapOptions mPriorOpt;
    bool mbResetToPriorMap;

    string CalculateCheckSum(string filename, int type);

    // Input sensor
    eSensor mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;
    Atlas* mpAtlas;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer* mpViewer;

    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;
    bool mbResetActiveMap;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // Shutdown flag
    bool mbShutDown;

    // Tracking state
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;

    //
    string mStrLoadAtlasFromFile;
    string mStrSaveAtlasToFile;

    string mStrVocabularyFilePath;

    Settings* settings_;
};

}// namespace ORB_SLAM

#endif // SYSTEM_H
