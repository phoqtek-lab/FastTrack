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


#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Atlas.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "Settings.h"

#include <mutex>
#include <deque>


namespace ORB_SLAM3
{

class System;
class Tracking;
class LoopClosing;
class Atlas;

class LocalMapping
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LocalMapping(System* pSys, Atlas* pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName=std::string());

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);
    void EmptyQueue();

    // Thread Synch
    void RequestStop();
    void RequestReset();
    void RequestResetActiveMap(Map* pMap);
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    bool IsInitializing();
    double GetCurrKFTime();

    // Gyro bias measured while the vehicle was verifiably at rest (rad/s; ORB-SLAM3 convention
    // w_true = w_meas - bg). While set, IMU initialization (InertialOptimization) and the
    // initialization bundle adjustments (BA1 and BA2) use it as the gyro-bias prior with
    // information 1/sigma^2, instead of the default weak prior towards zero (none in BA2).
    // sigma must be > 0. Thread-safe.
    void SetRestGyroBiasPrior(const Eigen::Vector3f &bg, float sigma);
    // Gravity reference (2026-10-23): flight-controller attitude R_world_body (world z up, same body frame as the IMU).
    // When enabled, IMU initialization takes the gravity direction from it instead of estimating it, and keeps it fixed
    // in the BA1/BA2 refinements.
    void AddGravityReference(double t, const Eigen::Matrix3f &Rwb);
    void SetGravityFromReference(bool b) { mbGravityFromRef = b; }
    bool GravityDirFromReference(const std::vector<KeyFrame*> &vpKF, Eigen::Vector3f &dirG, int &nUsed);
    std::mutex mMutexGravRef;
    std::deque<std::pair<double, Eigen::Matrix3f>> mdGravRef;
    bool mbGravityFromRef = false;
public:
    // B3 metric initialization: the map scale came from the barometer; keep it in the inertial initialization.
    void SetMetricScaleFixed(bool b) { mbMetricScaleFixed = b; }
    bool mbMetricScaleFixed = false;
    double mMetricC = 0.0;
    // B7 research: recent accepted B4 scale observations of the current map, count since the last step, total applied.
    std::deque<double> mCruiseObs; int mCruiseSince = 0; double mCruiseTotal = 1.0; Map* mpCruiseMap = nullptr;
    void CruiseScaleStep(double s, double h, double tilt);  // m, pad offset c of the last accepted baro scale fit (B4 diagnostic)
    bool mbMapGravityFromRef = false;
    void ClearRestGyroBiasPrior();
    bool GetRestGyroBiasPrior(Eigen::Vector3f &bg, double &info);
    // Init profile (2026-10-26): accelerometer bias of a saved initialization (m/s^2, IMU body frame). While set, IMU
    // initialization, BA1/BA2 and the local inertial BA use it as the accelerometer-bias prior (1/sigma^2) instead of
    // the prior towards zero. Thread-safe.
    void SetAccBiasPrior(const Eigen::Vector3f &ba, float sigma);
    void ClearAccBiasPrior();
    bool GetAccBiasPrior(Eigen::Vector3f &ba, double &info);
    // F1 (2026-10-27): last barometric scale observation of a keyframe (B4 diagnostic values), for the node's
    // divergence monitor. seq counts observations; false before the first one. Thread-safe.
    struct BaroScaleObs { double t = 0, h = 0, d = 0, s = 0, z = 0, tilt = 0; int nI = 0, nP = 0;
                          unsigned long mapId = 0, seq = 0; };
    bool GetBaroScaleObs(BaroScaleObs &o);
    std::mutex mMutexBaroObs;
    BaroScaleObs mBaroObs;
    KeyFrame* GetCurrKF();

    std::mutex mMutexImuInit;

    std::mutex mMutexRestGyroBias;
    bool mbRestGyroBias = false;
    Eigen::Vector3f mRestGyroBias = Eigen::Vector3f::Zero();
    double mRestGyroBiasInfo = 0.0;
    std::mutex mMutexAccBiasPrior;
    bool mbAccBiasPrior = false;
    Eigen::Vector3f mAccBiasPrior = Eigen::Vector3f::Zero();
    double mAccBiasPriorInfo = 0.0;

    Eigen::MatrixXd mcovInertial;
    Eigen::Matrix3d mRwg;
    Eigen::Vector3d mbg;
    Eigen::Vector3d mba;
    double mScale;
    double mInitTime;
    double mCostTime;

    unsigned int mInitSect;
    unsigned int mIdxInit;
    unsigned int mnKFs;
    double mFirstTs;
    int mnMatchesInliers;

    // For debugging (erase in normal mode)
    int mInitFr;
    int mIdxIteration;
    string strSequence;

    bool mbNotBA1;
    bool mbNotBA2;
    bool mbBadImu;

    bool mbWriteStats;

    // not consider far points (clouds)
    bool mbFarPoints;
    float mThFarPoints;

#ifdef REGISTER_TIMES
    vector<double> vdKFInsert_ms;
    vector<double> vdMPCulling_ms;
    vector<double> vdMPCreation_ms;
    vector<double> vdLBA_ms;
    vector<double> vdKFCulling_ms;
    vector<double> vdLMTotal_ms;


    vector<double> vdLBASync_ms;
    vector<double> vdKFCullingSync_ms;
    vector<int> vnLBA_edges;
    vector<int> vnLBA_KFopt;
    vector<int> vnLBA_KFfixed;
    vector<int> vnLBA_MPs;
    int nLBA_exec;
    int nLBA_abort;
#endif
protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();
    void KeyFrameCulling();

    System *mpSystem;

    bool mbMonocular;
    bool mbInertial;

    void ResetIfRequested();
    bool mbResetRequested;
    bool mbResetRequestedActiveMap;
    Map* mpMapToReset;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Atlas* mpAtlas;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;

    void InitializeIMU(float priorG = 1e2, float priorA = 1e6, bool bFirst = false);
    void ScaleRefinement(bool baroOnly = false);
    unsigned long mDeferClimbLog = 0;  // C1 (2026-10-31)
    long mnNewMpC2 = 0;  // C2 (2026-10-31) log: map points added by the last CreateNewMapPoints
    float mParMedC3 = -1.f, mParP10C3 = -1.f; bool mbClimbC3 = false;  // C3 (2026-10-31) log
    double mVzbC4 = 0.0;  // C4 (2026-10-31) log: baro vertical-velocity target of the last keyframe (NaN = none)
    int mClimbFitTick = 0, mClimbFitLog = 0;  // B5 climb baro scale

    bool bInitializing;

    Eigen::MatrixXd infoInertial;
    int mNumLM;
    int mNumKFCulling;

    float mTinit;

    int countRefinement;

    //DEBUG
    ofstream f_lm;

    };

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
