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


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "MapDrawer.h"
#include "System.h"
#include "ImuTypes.h"
#include "Settings.h"
#include "PriorMap.h"
#include "Kernels/CudaUtils.h"

#include "GeometricCamera.h"

#include <mutex>
#include <deque>
#include <unordered_set>
#include <deque>

namespace ORB_SLAM3
{

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;

class Tracking
{  

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Atlas* pAtlas,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq=std::string());

    ~Tracking();

    // Parse the config file
    bool ParseCamParamFile(cv::FileStorage &fSettings);
    bool ParseORBParamFile(cv::FileStorage &fSettings);
    bool ParseIMUParamFile(cv::FileStorage &fSettings);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    Sophus::SE3f GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename);
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);

    void GrabImuData(const IMU::Point &imuMeasurement);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
    void SetStepByStep(bool bSet);

    // P1 (2026-10-30): per-frame pose optimization in TrackLocalMap. -1: the GPU run-mode flag decides (production
    // gpu_pose_optimization=false skips it, so the frame pose is the IMU prediction and every match counts as an
    // inlier); 0: never; 1: always; 2: once the active map has finished inertial BA2 (before BA2 as -1).
    void SetPoseOptimizationMode(int mode) { mPoseOptMode = mode; }
    // P3 (2026-10-30): modes 3/4 (turns): rotation-rate threshold (deg/s) and hold time (s) after the last turn frame.
    void SetPoseOptimizationTurn(double rate_dps, double hold_s) { mPoseOptTurnRateDps = rate_dps; mPoseOptTurnHoldS = hold_s; }
    int GetPoseOptimizationMode() const { return mPoseOptMode; }
    bool GetStepByStep();

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    // External absolute reference injection (magnetometer heading + barometric altitude).
    // Called by the wrapper thread before each TrackMonocular. When the flags are false
    // (feature disabled at launch) the external-reference path stays inert and the
    // system behaves exactly as vanilla.
    void SetExternalReference(const bool bHasYaw, const double yawDeg, const bool bHasAlt, const double altM);

    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);
    KeyFrame* GetLastKeyFrame()
    {
        return mpLastKeyFrame;
    }

    void CreateMapInAtlas();
    //std::mutex mMutexTracks;

    //--
    void NewDataset();
    int GetNumberDataset();
    int GetMatchesInliers();

    //DEBUG
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder="");
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap);

    float GetImageScale();

#ifdef REGISTER_LOOP
    void RequestStop();
    bool isStopped();
    void Release();
    bool stopRequested();
#endif

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        RECENTLY_LOST=3,
        LOST=4,
        OK_KLT=5
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;

    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Sophus::SE3f> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // frames with estimated pose
    int mTrackedFr;
    bool mbStep;
    int mPoseOptMode{-1};
    double mPoseOptTurnRateDps{10.0};
    double mPoseOptTurnHoldS{3.0};
    double mPoseOptLastTurnT{-1.0};

    // RESEARCH frame-epoch contract: map id and frame epoch under which the most recent Track()
    // computed its pose (recorded while holding that map's mMutexMapUpdate).
    unsigned long mnPoseMapId = 0;
    unsigned long mnPoseFrameEpoch = 0;
    bool mbPoseEpochRecorded = false;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset(bool bLocMap = false);
    void ResetActiveMap(bool bLocMap = false);

    // Prior-map start (PriorMap.h / PriorMap.cc). Tracking-thread functions unless noted.
    static bool sbInstallSigintHandler;  // set by System before constructing Tracking
    void ConfigurePriorMap(const PriorMapOptions& opt, const PriorMapStatus& loadStatus, bool bLoaded);
    PriorMapStatus GetPriorMapStatus();  // any thread
    bool HasUsablePriorMap();
    // One-shot tolerance for a known image gap (the wrapper paused for a map save): the next frame may be up to
    // gapS later than the last one without the timestamp-jump reset/new map; the IMU predicts across it.
    void AllowTimeGapOnce(double gapS) { mAllowedGapOnceS = gapS > 0.0 ? gapS : 0.0; }
    bool ResetToPriorMap(std::string& reason);
    bool StopMappingThreads(bool bResetLocalMapper, double gbaWaitS, std::string& reason);
    void ReleaseMappingThreads();
    void AdoptActiveMapAsPrior(const SaveMapResult& r);
    void NotePriorSave(const SaveMapResult& r);

    float mMeanTrack;
    bool mbInitWith3KFs;
    double t0; // time-stamp of first read frame
    double t0vis; // time-stamp of first inserted keyframe
    double t0IMU; // time-stamp of IMU initialization
    bool mFastInit = false;


    vector<MapPoint*> GetLocalMapMPS();

    bool mbWriteStats;

#ifdef REGISTER_TIMES
    void LocalMapStats2File();
    void TrackStats2File();
    void PrintTimeStats();

    vector<double> vdRectStereo_ms;
    vector<double> vdResizeImage_ms;
    vector<double> vdORBExtract_ms;
    vector<double> vdStereoMatch_ms;
    vector<double> vdIMUInteg_ms;
    vector<double> vdPosePred_ms;
    vector<double> vdLMTrack_ms;
    vector<double> vdNewKF_ms;
    vector<double> vdTrackTotal_ms;
#endif

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    static void signalHandler(int signum);

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    //void CreateNewMapPoints();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();
    bool PredictStateIMU();

    bool Relocalization(Map* pTargetMap = nullptr);
    int mnRelocBestBowMatches = 0;  // best SearchByBoW matches of the last Relocalization()
    int mnRelocInliers = 0;         // inliers of the last successful Relocalization()
    int mnRelocCandidates = 0, mnRelocBestPnpInliers = 0, mnRelocBestGood = 0;  // diagnostics of the last Relocalization()

    // Prior-map start (PriorMap.cc)
    enum PriorStep { kPriorNone = 0, kPriorWait, kPriorRelocalized };
    PriorStep PriorMapStep(Map*& pCurrentMap, bool bSessionTracking = false);
    void PriorMapOnRelocalizedKeyFrame();
    void PriorMapOnLocalMapFail();
    void PriorMapNoteMapReset(Map* pMap, const char* why);
    void PriorMapWaitFrame();
    void PriorMapClearFrameMatches();
    bool PriorMapUsable();
    void BeginPriorSession();
    void SetPriorStatus(PriorMapStatus::State s, const std::string& reason);
    PriorMapOptions mPriorOpt;
    Map* mpPriorMap = nullptr;
    PriorMapStatus mPriorStatus;
    std::mutex mMutexPriorStatus;
    unsigned long mnSessionFirstKFId = 0;
    unsigned long mnSessionFirstMPId = 0;
    unsigned long mnSessionFirstMapId = 0;
    unsigned long mnPriorSessionFrames = 0;
    int mnPriorConfirm = 0;
    Sophus::SE3f mPriorConfirmTwb;
    double mPriorConfirmT = 0.0;     // frame time of the last confirmation (moving attempts)
    bool mbPriorConfirmMoving = false;
    bool mbPriorForceMapUpdated = false;

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();
    void LatchExternalRef();                      // compute yaw/z offsets once (IMU-initialized)
    void SetKeyFrameExternalRefs(KeyFrame* pKF);  // copy latched refs onto a new keyframe

    // Perform preintegration from last frame
    void PreintegrateIMU();

    // Reset IMU biases and compute frame velocity
    void ResetFrameIMU();

    // ZUPT (Zero Velocity Update) support
    void UpdateZuptState();

    bool mbMapUpdated;

    // Imu preintegration from last frame
    IMU::Preintegrated *mpImuPreintegratedFromLastKF;

    // Queue of IMU measurements between frames
    std::list<IMU::Point> mlQueueImuData;

    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> mvImuFromLastFrame;
    std::mutex mMutexImuQueue;

    // Imu calibration parameters
    IMU::Calib *mpImuCalib;

    // Last Bias Estimation (at keyframe creation)
    IMU::Bias mLastBias;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    static ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    bool mbReadyToInitializate;
    bool mbSetInit;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    bool bStepByStep;

    //Atlas
    Atlas* mpAtlas;

    //Calibration matrix
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    cv::Mat mDistCoef;
    float mbf;
    float mImageScale;

    float mImuFreq;
    double mImuPer;
    bool mInsertKFsLost;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    int mnFirstImuFrameId;
    int mnFramesToResetIMU;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    double mTimeStampLost;
    double time_recently_lost;

    // ZUPT (Zero Velocity Update) state
    std::string mSettingsPath;
    bool mbZuptEnabled;
    double mZuptGyroVarTh;   // variance threshold on gyro magnitude (rad/s)^2
    double mZuptAccelVarTh;  // variance threshold on accel magnitude (m/s^2)^2
    int mZuptWindow;         // number of recent IMU samples kept for variance
    int mZuptHystEnter;      // consecutive frames below th before declaring stationary
    int mZuptHystExit;       // consecutive frames above (*ratio) before declaring motion
    double mZuptExitRatio;   // multiplier on thresholds for exiting stationary
    double mZuptGyroMeanGate;  // abs mean gyro magnitude above which motion is forced (rad/s)
    double mZuptAccelNormGate; // |mean accel magnitude - g| above which motion is forced (m/s^2)
    double mZuptVelGate;       // low gate used to block entering stationary (m/s)
    double mZuptVelExitTh;     // sustained EMA velocity above which the hold is released (m/s)
    int    mZuptVelHyst;       // consecutive EMAs above the exit threshold before release
    bool   mbZuptHoldRotation; // IMU.ZUPT.HoldRotation: while stationary also hold attitude (default off)
    double mAllowedGapOnceS = 0.0;  // AllowTimeGapOnce
    // IMU.ZUPT.StrongRest.*: IMU stillness that flight never produces overrides the velocity gates, because the
    // estimated velocity is the signal that drifts when vision is weak (default off).
    bool   mbZuptStrongRestEnabled;
    double mZuptStrongGyroVarTh;   // (rad/s)^2 on the gyro magnitude over the window
    double mZuptStrongAccelVarTh;  // (m/s^2)^2 on the accel magnitude over the window
    double mZuptStrongMinS;        // seconds the strong-rest test must hold before it applies
    double mZuptStrongSince;       // frame time the current strong-rest run began (-1: none)
    bool   mbZuptStrongRest;       // strong rest currently applies
public:
    // B3 metric initialization (Init.Metric.*, default off): barometric height sets the initial map scale.
    void AddAltitudeReference(double t, double h);
    bool AltitudeAt(double t, double &h, double &hPad);
    // C1 (2026-10-31): baro climb rate over the last 1 s above Metric.DeferClimbRate (false when off or no baro).
    bool ClimbingAt(double t, double &rate);
    double mDeferClimbRate = 0.0;
    // C3 (2026-10-31): minimum triangulation parallax while climbing faster than mClimbParallaxRate (0 deg = off).
    bool ClimbRateAbove(double t, double thr, double &rate);
    double mClimbMinParallaxDeg = 0.0, mClimbParallaxRate = 0.5;
    // C4 (2026-10-31): baro vertical-velocity factor in the local inertial BA.
    bool mbBaroVz = false;
    double mBaroVzSigma = 0.3, mBaroVzWindow = 1.0;
    bool mbClimbFullTail = false;  // C6 (2026-10-31)
    float MetricInitScale(KeyFrame* pKFini, KeyFrame* pKFcur);
    bool   mbMetricInitEnabled = false;
    double mMetricPadCamHeight = 0.0;   // m, hint only: camera above the ground on the pad (estimated later)
    bool BaroScaleFit(Map* pMap, const std::vector<KeyFrame*> &vpKF, double &s, double &c, int &n, double &range,
                      double &rms);
    double mMetricMinHeight = 3.0;
    // B7 research: cruise scale correction from the B4 observation (Metric.Cruise.*).
    bool mbCruiseScale = false;
    int mCruiseWindow = 15, mCruisePeriod = 5;
    double mCruiseGain = 0.5, mCruiseDeadband = 0.01, mCruiseMaxStep = 0.05, mCruiseMinHeight = 20.0,
           mCruiseMaxTilt = 25.0, mCruiseMinInlier = 0.6;
    double mInitMinAxisAngleDeg = 0.0;  // B4b: reject mono inits whose translation is within this angle of the optical axis
    // B4 diagnostic: distance from the keyframe camera to the plane of its own map points, plane tilt from vertical.
    bool LocalGroundDistance(KeyFrame* pKF, double &d, double &tiltDeg, int &nInl, int &nPts);      // m, minimum height above the pad for the metric scale
    std::deque<std::pair<double,double>> mdAltRef;
    std::mutex mMutexAltRef;
    double mAltPad = 0.0; bool mbAltPad = false; double mAltFirstT = -1.0; std::vector<double> mvAltFirst;
    // WP C1 (Klt.*, default off): KLT-guided association of the last frame's map points.
    void KltAssociate();
    bool   mbKltEnabled = false;
    int    mKltWin = 21;
    int    mKltLevels = 3;
    double mKltMaxFb = 1.0;       // px, forward-backward error
    double mKltRadius = 3.0;      // px, keypoint search radius around the tracked position
    int    mKltMaxHamming = 100;  // descriptor sanity check (ORBmatcher::TH_HIGH)
    bool   mbKltPersistent = false;  // C1b (2026-10-31): append KLT positions as observations
    int    mKltMaxAppend = 200;
    unsigned long mKltStatAppended = 0;
    cv::Mat mKltPrevImg;          // previous grayscale image (clone)
    unsigned long mKltStatFrames = 0, mKltStatTried = 0, mKltStatTracked = 0, mKltStatAssoc = 0;
    // WP B2 (IMU.RestObs.*, default off): rest observations for the inertial optimizations.
    bool   mbRestObsEnabled = false;
    double mRestObsGyroSigma = 5e-4;  // rad/s, floor on the window-mean gyro uncertainty
    double mRestObsAccSigma = 0.02;   // m/s^2, floor on the window-mean accel uncertainty
    int    mZuptVelExitCount;  // velocity-exit hysteresis counter
    double mZuptVelEma;        // EMA of the tracked velocity magnitude (m/s)
    bool mbZuptActive;       // current stationarity state
    int mZuptEnterCount;
    int mZuptExitCount;
    std::deque<IMU::Point> mlZuptImuWindow;

    // External absolute reference (compass heading + barometric altitude) anchoring.
    bool mbExtRefHasYaw;            // a valid mag heading is available
    bool mbExtRefHasAlt;            // a valid baro altitude is available
    double mExtRefYawDeg;           // latest compass heading (deg, 0=N)
    double mExtRefAltM;             // latest baro altitude (m, rel. to home/takeoff)
    bool mExtRefLatched;            // offsets computed (once, after IMU init)
    unsigned long mExtRefLatchInitKF; // map init KF id at latch time (re-latch on reset)
    double mExtMagYawOffsetRad;     // slam yaw - mag yaw (constant; absorbs Tcb + scalar offsets)
    double mExtBaroAltOffsetM;      // slam z - baro alt (constant; absorbs origin offset)
    double mExtMagYawSigmaDeg;      // trust sigma for the mag yaw prior
    bool mbExtMagYawClockwise;      // nf1: heading is clockwise (target = offset - heading)
    double mExtBaroAltSigmaM;       // trust sigma for the baro alt prior
    bool mbExtRefPrinted;           // one-shot info print
    std::mutex mMutexExtRef;

    unsigned int mnFirstFrameId;
    unsigned int mnInitialFrameId;
    unsigned int mnLastInitFrameId;

    bool mbCreatedMap;

    //Motion Model
    bool mbVelocity{false};
    Sophus::SE3f mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;

    //int nMapChangeIndex;

    int mnNumDataset;

    ofstream f_track_stats;

    ofstream f_track_times;
    double mTime_PreIntIMU;
    double mTime_PosePred;
    double mTime_LocalMapTrack;
    double mTime_NewKF_Dec;

    GeometricCamera* mpCamera, *mpCamera2;

    int initID, lastID;

    Sophus::SE3f mTlr;

    void newParameterLoader(Settings* settings);

#ifdef REGISTER_LOOP
    bool Stop();

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;
#endif

public:
    cv::Mat mImRight;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
