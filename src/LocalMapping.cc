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


#include "LocalMapping.h"
#include <iomanip>
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "GeometricTools.h"
#include "CalculationLogger.h"

#include<mutex>
#include <algorithm>
#include <cmath>
#include <limits>  // C4 (2026-10-31)
#include<chrono>

namespace ORB_SLAM3
{

LocalMapping::LocalMapping(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
    mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false), mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), bInitializing(false),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    mIdxInit(0), mScale(1.0), mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0), infoInertial(Eigen::MatrixXd::Zero(9,9))
{
    mnMatchesInliers = 0;

    mbBadImu = false;

    mTinit = 0.f;

    mNumLM = 0;
    mNumKFCulling=0;

#ifdef REGISTER_TIMES
    nLBA_exec = 0;
    nLBA_abort = 0;
#endif

}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{
    CalculationLog::preallocate();
    FT_CALC_SCOPE();
    FT_CALC_LOG("local_mapping.worker.start inertial=%d monocular=%d", mbInertial ? 1 : 0, mbMonocular ? 1 : 0);
    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames() && !mbBadImu)
        {
            FT_CALC_LOG("local_mapping.cycle.begin queue=%d", KeyframesInQueue());
#ifdef REGISTER_TIMES
            double timeLBA_ms = 0;
            double timeKFCulling_ms = 0;

            std::chrono::steady_clock::time_point time_StartProcessKF = std::chrono::steady_clock::now();
#endif
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndProcessKF = std::chrono::steady_clock::now();

            double timeProcessKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndProcessKF - time_StartProcessKF).count();
            vdKFInsert_ms.push_back(timeProcessKF);
#endif

            // Check recent MapPoints
            MapPointCulling();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCulling = std::chrono::steady_clock::now();

            double timeMPCulling = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCulling - time_EndProcessKF).count();
            vdMPCulling_ms.push_back(timeMPCulling);
#endif

            // Triangulate new MapPoints
            const size_t nMpBeforeC2 = mpAtlas->GetCurrentMap()->MapPointsInMap();  // C2 (2026-10-31) log
            CreateNewMapPoints();
            mnNewMpC2 = (long)mpAtlas->GetCurrentMap()->MapPointsInMap() - (long)nMpBeforeC2;

            mbAbortBA = false;

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCreation = std::chrono::steady_clock::now();

            double timeMPCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCreation - time_EndMPCulling).count();
            vdMPCreation_ms.push_back(timeMPCreation);
#endif

            bool b_doneLBA = false;
            int num_FixedKF_BA = 0;
            int num_OptKF_BA = 0;
            int num_MPs_BA = 0;
            int num_edges_BA = 0;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                if(mpAtlas->KeyFramesInMap()>2)
                {

                    if(mbInertial && mpCurrentKeyFrame->GetMap()->isImuInitialized())
                    {
                        // A keyframe chain can be shorter than 3 (prior-map start begins a new chain).
                        float dist = 0.f;
                        if(mpCurrentKeyFrame->mPrevKF && mpCurrentKeyFrame->mPrevKF->mPrevKF)
                            dist = (mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()).norm() +
                                (mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter()).norm();

                        if(dist>0.05)
                            mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            if((mTinit<10.f) && (dist<0.02))
                            {
                                FT_CALC_LOG("local_mapping.imu_reset reason=insufficient_motion distance=%f init_time=%f",
                                            dist, mTinit);
                                unique_lock<mutex> lock(mMutexReset);
                                mbResetRequestedActiveMap = true;
                                mpMapToReset = mpCurrentKeyFrame->GetMap();
                                mbBadImu = true;
                            }
                        }

                        // Init profile: accelerometer-bias prior for this keyframe in the local inertial BA.
                        {
                            Eigen::Vector3f pba; double pinfo = 0.0;
                            if(GetAccBiasPrior(pba, pinfo))
                            { mpCurrentKeyFrame->mbAccPrior = true; mpCurrentKeyFrame->mAccPrior = pba; mpCurrentKeyFrame->mAccPriorInfo = pinfo; }
                        }
                        // C6 (2026-10-31): full-weight tail link while climbing.
                        {
                            double rC6 = 0.0;
                            mpCurrentKeyFrame->mbC6FullTail = mpTracker->mbClimbFullTail &&
                                mpTracker->ClimbRateAbove(mpCurrentKeyFrame->mTimeStamp, mpTracker->mClimbParallaxRate, rC6);
                        }
                        // C4 (2026-10-31): baro vertical-velocity target for this keyframe (metric map after BA2).
                        mVzbC4 = std::numeric_limits<double>::quiet_NaN();
                        if(mpTracker->mbBaroVz && mbMonocular && mbMetricScaleFixed && mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            const double W = mpTracker->mBaroVzWindow; double h1 = 0, h0 = 0, hp = 0;
                            if(mpTracker->AltitudeAt(mpCurrentKeyFrame->mTimeStamp, h1, hp) && mpTracker->AltitudeAt(mpCurrentKeyFrame->mTimeStamp - W, h0, hp))
                            {
                                mpCurrentKeyFrame->mBaroVz = (h1 - h0) / W;
                                mpCurrentKeyFrame->mBaroVzInfo = 1.0 / (mpTracker->mBaroVzSigma * mpTracker->mBaroVzSigma);
                                mpCurrentKeyFrame->mbBaroVzObs = true;
                                mVzbC4 = mpCurrentKeyFrame->mBaroVz;
                            }
                        }
                        bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>100)&&!mbMonocular);
                        const Eigen::Vector3f zPreC2 = mpCurrentKeyFrame->GetCameraCenter(), vPreC2 = mpCurrentKeyFrame->GetVelocity();
                        Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
                        b_doneLBA = true;
                        // C2 (2026-10-31, log only): climb diagnostic per keyframe while airborne (baro > 1 m).
                        {
                            double hC2 = 0, hPadC2 = 0;
                            if(mpTracker->AltitudeAt(mpCurrentKeyFrame->mTimeStamp, hC2, hPadC2) && hC2 - hPadC2 > 1.0)
                            {
                                const Eigen::Vector3f z1 = mpCurrentKeyFrame->GetCameraCenter(), v1 = mpCurrentKeyFrame->GetVelocity();
                                std::cerr << std::fixed << std::setprecision(3) << "FastTrack C2 kf t=" << mpCurrentKeyFrame->mTimeStamp
                                          << " h=" << hC2 - hPadC2 << " z_pre=" << zPreC2(2) << " vz_pre=" << vPreC2(2)
                                          << " z=" << z1(2) << " vz=" << v1(2) << " vzb=" << mVzbC4 << " ba_z=" << mpCurrentKeyFrame->GetAccBias()(2)
                                          << " new_mp=" << mnNewMpC2 << " par_med=" << mParMedC3 << " par_p10=" << mParP10C3
                                          << " climb=" << (mbClimbC3 ? 1 : 0) << " inl=" << mpTracker->GetMatchesInliers()
                                          << " fixedKF=" << num_FixedKF_BA << " optKF=" << num_OptKF_BA << " mps=" << num_MPs_BA
                                          << std::defaultfloat << std::setprecision(6) << std::endl;
                            }
                        }
                        // C7c (2026-10-31): set the keyframe accelerometer bias from the baro-inertial vertical KF.
                        if(mpTracker->mbVertKF && mbMonocular && mbMetricScaleFixed && mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                            VertKFInject();
                        // C7 (2026-10-31): in-flight accelerometer-bias observer (metric map after BA2).
                        if(mpTracker->mbAccBiasObs && mbMonocular && mbMetricScaleFixed && mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                            AccBiasObserverStep(vPreC2(2));
                        else
                            mC7.clear();
                        // B4 diagnostic (log only): barometric scale observation for this keyframe after BA2.
                        if(mbMetricScaleFixed && mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            double d = 0, tilt = 0, h = 0, hPad = 0; int nI = 0, nP = 0;
                            const bool g = mpTracker->LocalGroundDistance(mpCurrentKeyFrame, d, tilt, nI, nP);
                            if(g && d > 1e-3 && mpTracker->AltitudeAt(mpCurrentKeyFrame->mTimeStamp, h, hPad))
                            {
                                std::cerr << std::fixed << std::setprecision(3) << "FastTrack B4 kf t=" << mpCurrentKeyFrame->mTimeStamp
                                          << " h=" << h - hPad << " d=" << d << " s=" << (h - hPad + mMetricC) / d
                                          << " z=" << mpCurrentKeyFrame->GetCameraCenter()(2) << " tilt=" << tilt
                                          << " inl=" << nI << "/" << nP << std::defaultfloat << std::setprecision(6) << std::endl;
                                {
                                    unique_lock<mutex> lockObs(mMutexBaroObs);
                                    mBaroObs.t = mpCurrentKeyFrame->mTimeStamp; mBaroObs.h = h - hPad; mBaroObs.d = d;
                                    mBaroObs.s = (h - hPad + mMetricC) / d; mBaroObs.z = mpCurrentKeyFrame->GetCameraCenter()(2);
                                    mBaroObs.tilt = tilt; mBaroObs.nI = nI; mBaroObs.nP = nP;
                                    mBaroObs.mapId = mpCurrentKeyFrame->GetMap()->GetId(); ++mBaroObs.seq;
                                }
                                if(mpTracker->mbCruiseScale && mbMonocular && nP >= 60 && nI >= mpTracker->mCruiseMinInlier * nP)
                                    CruiseScaleStep((h - hPad + mMetricC) / d, h - hPad, tilt);
                                if(mpTracker->mbTiltCorr && mbMonocular)  // C8
                                    TiltCorrStep(h - hPad);
                            }
                        }
                    }
                    else
                    {
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                        b_doneLBA = true;
                    }

                }
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndLBA = std::chrono::steady_clock::now();

                if(b_doneLBA)
                {
                    timeLBA_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLBA - time_EndMPCreation).count();
                    vdLBA_ms.push_back(timeLBA_ms);

                    nLBA_exec += 1;
                    if(mbAbortBA)
                    {
                        nLBA_abort += 1;
                    }
                    vnLBA_edges.push_back(num_edges_BA);
                    vnLBA_KFopt.push_back(num_OptKF_BA);
                    vnLBA_KFfixed.push_back(num_FixedKF_BA);
                    vnLBA_MPs.push_back(num_MPs_BA);
                }

#endif

                // Initialize IMU here
                if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
                {
                    if (mbMonocular)
                        InitializeIMU(1e2, 1e10, true);
                    else
                        InitializeIMU(1e2, 1e5, true);
                }


                // Check redundant local Keyframes
                KeyFrameCulling();

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndKFCulling = std::chrono::steady_clock::now();

                timeKFCulling_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndKFCulling - time_EndLBA).count();
                vdKFCulling_ms.push_back(timeKFCulling_ms);
#endif

                if ((mTinit<50.0f) && mbInertial)
                {
                    if(mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState==Tracking::OK) // Enter here everytime local-mapping is called
                    {
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA1()){
                            if (mTinit>5.0f)
                            {
                                FT_CALC_LOG("local_mapping.viba.start stage=1 init_time=%f", mTinit);
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA1();
                                if (mbMonocular)
                                    InitializeIMU(1.f, 1e5, true);
                                else
                                    InitializeIMU(1.f, 1e5, true);

                                FT_CALC_LOG("local_mapping.viba.end stage=1");
                            }
                        }
                        else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2()){
                            if (mTinit>15.0f){
                                FT_CALC_LOG("local_mapping.viba.start stage=2 init_time=%f", mTinit);
                                // P2 (2026-10-30): BA2 is marked done only after its optimization has updated the map
                                // (was before it: the nav output latched on the pre-BA2 map and jumped when BA2 landed).
                                const unsigned int nIdxBefore = mIdxInit;
                                if (mbMonocular)
                                    InitializeIMU(0.f, 0.f, true);
                                else
                                    InitializeIMU(0.f, 0.f, true);
                                if (mIdxInit != nIdxBefore)
                                    mpCurrentKeyFrame->GetMap()->SetIniertialBA2();

                                FT_CALC_LOG("local_mapping.viba.end stage=2");
                            }
                        }

                        // scale refinement
                        // C1 (2026-10-31): no whole-map scale operation while climbing (Metric.DeferClimbRate > 0).
                        double climbRate = 0.0;
                        const bool bDeferClimb = mpTracker->ClimbingAt(mpCurrentKeyFrame->mTimeStamp, climbRate);
                        if (bDeferClimb && (++mDeferClimbLog % 10 == 1))
                            std::cerr << "FastTrack C1 scale operation deferred: climb " << climbRate << " m/s, tinit " << mTinit
                                      << " (" << mDeferClimbLog << " keyframes)" << std::endl;
                        // B5: pad start, map not metric yet -> try the baro scale fit every 5th keyframe after BA2.
                        const bool bClimbFit = !bDeferClimb && mbMonocular && mpTracker->mbMetricInitEnabled && !mbMetricScaleFixed &&
                                               mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && (++mClimbFitTick % 5 == 0);
                        if (bDeferClimb)
                        {
                        }
                        else if (bClimbFit)
                            ScaleRefinement(true);
                        else if (((mpAtlas->KeyFramesInMap())<=200) &&
                                ((mTinit>25.0f && mTinit<25.5f)||
                                (mTinit>35.0f && mTinit<35.5f)||
                                (mTinit>45.0f && mTinit<45.5f)||
                                (mTinit>55.0f && mTinit<55.5f)||
                                (mTinit>65.0f && mTinit<65.5f)||
                                (mTinit>75.0f && mTinit<75.5f))){
                            if (mbMonocular && !mbMetricScaleFixed)
                            {
                                std::cerr << "FastTrack scheduled scale refinement: tinit " << mTinit << std::endl;  // C1: was silent
                                ScaleRefinement();
                            }
                        }
                    }
                }
            }

#ifdef REGISTER_TIMES
            vdLBASync_ms.push_back(timeKFCulling_ms);
            vdKFCullingSync_ms.push_back(timeKFCulling_ms);
#endif

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

            FT_CALC_LOG("local_mapping.cycle.end keyframe=%llu lba_done=%d abort=%d optimized_kfs=%d optimized_mps=%d edges=%d",
                        static_cast<unsigned long long>(mpCurrentKeyFrame->mnId), b_doneLBA ? 1 : 0,
                        mbAbortBA ? 1 : 0, num_OptKF_BA, num_MPs_BA, num_edges_BA);

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndLocalMap = std::chrono::steady_clock::now();

            double timeLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLocalMap - time_StartProcessKF).count();
            vdLMTotal_ms.push_back(timeLocalMap);
#endif
        }
        else if(Stop() && !mbBadImu)
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
    FT_CALC_LOG("local_mapping.worker.end");
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    FT_CALC_LOG("local_mapping.keyframe.enqueue id=%llu", static_cast<unsigned long long>(pKF->mnId));
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    FT_CALC_SCOPE();
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
    FT_CALC_LOG("local_mapping.keyframe.processed id=%llu map_points=%llu",
                static_cast<unsigned long long>(mpCurrentKeyFrame->mnId),
                static_cast<unsigned long long>(vpMapPointMatches.size()));
}

void LocalMapping::EmptyQueue()
{
    while(CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

void LocalMapping::MapPointCulling()
{
    FT_CALC_SCOPE();
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    int borrar = mlpRecentAddedMapPoints.size();

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;

        if(pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        else if(pMP->GetFoundRatio()<0.25f)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
        {
            lit++;
            borrar--;
        }
    }
    FT_CALC_LOG("local_mapping.mappoint_culling keyframe=%llu remaining=%d",
                static_cast<unsigned long long>(nCurrentKFid), borrar);
}


void LocalMapping::CreateNewMapPoints()
{
    FT_CALC_SCOPE();
    // C3 (2026-10-31): parallax gate while climbing + parallax of the created points (log).
    double climbRateC3 = 0.0;
    const bool bClimbC3 = mpTracker->ClimbRateAbove(mpCurrentKeyFrame->mTimeStamp, mpTracker->mClimbParallaxRate, climbRateC3);
    mbClimbC3 = bClimbC3;
    const float cosMinInertial = (bClimbC3 && mpTracker->mClimbMinParallaxDeg > 0.0) ?
        std::min(0.9996f, (float)std::cos(mpTracker->mClimbMinParallaxDeg * M_PI / 180.0)) : 0.9996f;
    std::vector<float> vParC3;
    mParMedC3 = -1.f; mParP10C3 = -1.f;  // stays -1 when the loop returns early (new keyframe waiting)
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    // For stereo inertial case
    if(mbMonocular)
        nn=30;
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    if (mbInertial)
    {
        KeyFrame* pKF = mpCurrentKeyFrame;
        int count=0;
        while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
        {
            vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
            if(it==vpNeighKFs.end())
                vpNeighKFs.push_back(pKF->mPrevKF);
            pKF = pKF->mPrevKF;
        }
    }

    float th = 0.6f;

    ORBmatcher matcher(th,false);

    Sophus::SE3<float> sophTcw1 = mpCurrentKeyFrame->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose();
    Eigen::Vector3f tcw1 = sophTcw1.translation();
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
    int countStereo = 0;
    int countStereoGoodProj = 0;
    int countStereoAttempt = 0;
    int totalStereoPts = 0;
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        Eigen::Vector3f vBaseline = Ow2-Ow1;
        const float baseline = vBaseline.norm();

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        bool bCoarse = mbInertial && mpTracker->mState==Tracking::RECENTLY_LOST && mpCurrentKeyFrame->GetMap()->GetIniertialBA2();

        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse);

        Sophus::SE3<float> sophTcw2 = pKF2->GetPose();
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4();
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0);
        Eigen::Matrix<float,3,3> Rwc2 = Rcw2.transpose();
        Eigen::Vector3f tcw2 = sophTcw2.translation();

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = (mpCurrentKeyFrame -> NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
                                                                         : (idx1 < mpCurrentKeyFrame -> NLeft) ? mpCurrentKeyFrame -> mvKeys[idx1]
                                                                                                               : mpCurrentKeyFrame -> mvKeysRight[idx1 - mpCurrentKeyFrame -> NLeft];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur>=0);
            const bool bRight1 = (mpCurrentKeyFrame -> NLeft == -1 || idx1 < mpCurrentKeyFrame -> NLeft) ? false
                                                                                                         : true;

            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];

            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur>=0);
            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                               : true;

            if(mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2){
                if(bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if(bRight1 && !bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if(!bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else{
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
                eigTcw1 = sophTcw1.matrix3x4();
                Rcw1 = eigTcw1.block<3,3>(0,0);
                Rwc1 = Rcw1.transpose();
                tcw1 = sophTcw1.translation();

                eigTcw2 = sophTcw2.matrix3x4();
                Rcw2 = eigTcw2.block<3,3>(0,0);
                Rwc2 = Rcw2.transpose();
                tcw2 = sophTcw2.translation();
            }

            // Check parallax between rays
            Eigen::Vector3f xn1 = pCamera1->unprojectEig(kp1.pt);
            Eigen::Vector3f xn2 = pCamera2->unprojectEig(kp2.pt);

            Eigen::Vector3f ray1 = Rwc1 * xn1;
            Eigen::Vector3f ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm() * ray2.norm());

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            if (bStereo1 || bStereo2) totalStereoPts++;
            
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            Eigen::Vector3f x3D;

            bool goodProj = false;
            bool bPointStereo = false;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 ||
                                                                          (cosParallaxRays<cosMinInertial && mbInertial) || (cosParallaxRays<0.9998 && !mbInertial)))
            {
                goodProj = GeometricTools::Triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D);
                if(!goodProj)
                    continue;
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj = mpCurrentKeyFrame->UnprojectStereo(idx1, x3D);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                countStereoAttempt++;
                bPointStereo = true;
                goodProj = pKF2->UnprojectStereo(idx2, x3D);
            }
            else
            {
                continue; //No stereo and very low parallax
            }

            if(goodProj && bPointStereo)
                countStereoGoodProj++;

            if(!goodProj)
                continue;

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1(0);
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1));
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;

                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;

            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2(0);
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2));
                float errX2 = uv2.x - kp2.pt.x;
                float errY2 = uv2.y - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            Eigen::Vector3f normal1 = x3D - Ow1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = x3D - Ow2;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());
            vParC3.push_back((float)(std::acos(std::min(1.f, std::max(-1.f, cosParallaxRays))) * 180.0 / M_PI));  // C3 log
            if (bPointStereo)
                countStereo++;
            
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }    
    // C3 (2026-10-31) log: parallax of the points created at this keyframe.
    if(vParC3.empty()) { mParMedC3 = -1.f; mParP10C3 = -1.f; }
    else
    {
        std::sort(vParC3.begin(), vParC3.end());
        mParMedC3 = vParC3[vParC3.size() / 2]; mParP10C3 = vParC3[vParC3.size() / 10];
    }
}

void LocalMapping::SearchInNeighbors()
{
    FT_CALC_SCOPE();
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=30;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    for(int i=0, imax=vpTargetKFs.size(); i<imax; i++)
    {
        const vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
        }
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    if(mbInertial)
    {
        KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF;
        while(vpTargetKFs.size()<20 && pKFi)
        {
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            {
                pKFi = pKFi->mPrevKF;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
            pKFi = pKFi->mPrevKF;
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
        if(pKFi->NLeft != -1) matcher.Fuse(pKFi,vpMapPointMatches,true);
    }


    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);
    if(mpCurrentKeyFrame->NLeft != -1) matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates,true);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
    FT_CALC_LOG("local_mapping.neighbor_search keyframe=%llu targets=%llu fuse_candidates=%llu aborted=%d",
                static_cast<unsigned long long>(mpCurrentKeyFrame->mnId),
                static_cast<unsigned long long>(vpTargetKFs.size()),
                static_cast<unsigned long long>(vpFuseCandidates.size()), mbAbortBA ? 1 : 0);
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        FT_CALC_LOG("local_mapping.worker.stop");
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    FT_CALC_LOG("local_mapping.worker.release");
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21;
    mpCurrentKeyFrame->UpdateBestCovisibles();
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th;
    if(!mbInertial)
        redundant_th = 0.9;
    else if (mbMonocular)
        redundant_th = 0.9;
    else
        redundant_th = 0.5;

    const bool bInitImu = mpAtlas->isImuInitialized();
    int count=0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID;
    if (mbInertial)
    {
        int count = 0;
        KeyFrame* aux_KF = mpCurrentKeyFrame;
        while(count<Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }

    int itr = 0;
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        count++;
        KeyFrame* pKF = *vit;

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        itr += vpMapPoints.size();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = (pKF -> NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                                     : (i < pKF -> NLeft) ? pKF -> mvKeys[i].octave
                                                                                          : pKF -> mvKeysRight[i].octave;
                        const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            tuple<int,int> indexes = mit->second;
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int scaleLeveli = -1;
                            if(pKFi -> NLeft == -1)
                                scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
                            else {
                                if (leftIndex != -1) {
                                    scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                                }
                                if (rightIndex != -1) {
                                    int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
                                    scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                  : scaleLeveli;
                                }
                            }

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>thObs)
                                    break;
                            }
                        }
                        if(nObs>thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>redundant_th*nMPs)
        {
            if (mbInertial)
            {
                if (mpAtlas->KeyFramesInMap()<=Nd)
                    continue;

                if(pKF->mnId>(mpCurrentKeyFrame->mnId-2))
                    continue;

                if(pKF->mPrevKF && pKF->mNextKF)
                {
                    const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp;

                    if((bInitImu && (pKF->mnId<last_ID) && t<3.) || (t<0.5))
                    {
                        KeyFrame* pPrevKF = pKF->mPrevKF;
                        if(!pKF->SetBadFlag(pPrevKF, true))
                            FT_CALC_LOG("research.cull.retained kf=%llu reason=tree_repair_unavailable",
                                        static_cast<unsigned long long>(pKF->mnId));
                    }
                    else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && ((pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition()).norm()<0.02) && (t<3))
                    {
                        KeyFrame* pPrevKF = pKF->mPrevKF;
                        if(!pKF->SetBadFlag(pPrevKF, true))
                            FT_CALC_LOG("research.cull.retained kf=%llu reason=tree_repair_unavailable",
                                        static_cast<unsigned long long>(pKF->mnId));
                    }
                }
            }
            else
            {
                if(!pKF->SetBadFlag(pKF->mPrevKF))
                    FT_CALC_LOG("research.cull.retained kf=%llu reason=tree_repair_unavailable",
                                static_cast<unsigned long long>(pKF->mnId));
            }
        }
        if((count > 20 && mbAbortBA) || count>100)
        {
            break;
        }
    }
}

void LocalMapping::RequestReset()
{
    FT_CALC_LOG("local_mapping.reset.request type=atlas");
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
    FT_CALC_LOG("local_mapping.reset.complete type=atlas");
}

void LocalMapping::RequestResetActiveMap(Map* pMap)
{
    FT_CALC_LOG("local_mapping.reset.request type=active_map id=%llu",
                static_cast<unsigned long long>(pMap ? pMap->GetId() : 0));
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequestedActiveMap)
                break;
        }
        usleep(3000);
    }
    FT_CALC_LOG("local_mapping.reset.complete type=active_map");
}

void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            executed_reset = true;

            FT_CALC_LOG("local_mapping.reset.execute type=atlas");
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mbResetRequested = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;
            mbMetricScaleFixed = false;

            mIdxInit=0;

            FT_CALC_LOG("local_mapping.reset.execute_complete type=atlas");
        }

        if(mbResetRequestedActiveMap) {
            executed_reset = true;
            FT_CALC_LOG("local_mapping.reset.execute type=active_map");
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mbResetRequested = false;
            mbResetRequestedActiveMap = false;
            FT_CALC_LOG("local_mapping.reset.execute_complete type=active_map");
        }
    }
    if(executed_reset)
        FT_CALC_LOG("local_mapping.reset.unlock");

}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::AddGravityReference(double t, const Eigen::Matrix3f &Rwb)
{
    if(!std::isfinite(t) || !Rwb.allFinite())
        return;
    unique_lock<mutex> lock(mMutexGravRef);
    if(!mdGravRef.empty() && t <= mdGravRef.back().first)
        return;
    mdGravRef.emplace_back(t, Rwb);
    while(!mdGravRef.empty() && t - mdGravRef.front().first > 120.0)
        mdGravRef.pop_front();
}

bool LocalMapping::GravityDirFromReference(const std::vector<KeyFrame*> &vpKF, Eigen::Vector3f &dirG, int &nUsed)
{
    // Gravity (down) in the map frame: R_map_body(KF) * R_world_body(FC, nearest in time)^T * (0,0,-1).
    Eigen::Vector3f acc = Eigen::Vector3f::Zero();
    nUsed = 0;
    unique_lock<mutex> lock(mMutexGravRef);
    if(mdGravRef.empty())
        return false;
    for(KeyFrame* pKF : vpKF)
    {
        if(!pKF || pKF->isBad())
            continue;
        const double t = pKF->mTimeStamp;
        auto it = std::lower_bound(mdGravRef.begin(), mdGravRef.end(), t,
            [](const std::pair<double, Eigen::Matrix3f> &a, double v) { return a.first < v; });
        const std::pair<double, Eigen::Matrix3f> *best = nullptr;
        if(it != mdGravRef.end()) best = &(*it);
        if(it != mdGravRef.begin()) { auto ip = std::prev(it); if(!best || std::fabs(ip->first - t) < std::fabs(best->first - t)) best = &(*ip); }
        if(!best || std::fabs(best->first - t) > 0.05)
            continue;
        const Eigen::Vector3f gBody = best->second.transpose() * Eigen::Vector3f(0.f, 0.f, -1.f);
        acc += pKF->GetImuRotation() * gBody;
        nUsed++;
    }
    if(nUsed < 3 || acc.norm() < 1e-3f)
        return false;
    dirG = acc.normalized();
    return true;
}

void LocalMapping::SetRestGyroBiasPrior(const Eigen::Vector3f &bg, float sigma)
{
    if (!(sigma > 0.f) || !bg.allFinite())
        return;
    unique_lock<mutex> lock(mMutexRestGyroBias);
    mRestGyroBias = bg;
    mRestGyroBiasInfo = 1.0 / (static_cast<double>(sigma) * static_cast<double>(sigma));
    mbRestGyroBias = true;
}

void LocalMapping::ClearRestGyroBiasPrior()
{
    unique_lock<mutex> lock(mMutexRestGyroBias);
    mbRestGyroBias = false;
}

bool LocalMapping::GetRestGyroBiasPrior(Eigen::Vector3f &bg, double &info)
{
    unique_lock<mutex> lock(mMutexRestGyroBias);
    if (!mbRestGyroBias)
        return false;
    bg = mRestGyroBias;
    info = mRestGyroBiasInfo;
    return true;
}

void LocalMapping::SetAccBiasPrior(const Eigen::Vector3f &ba, float sigma)
{
    if (!(sigma > 0.f) || !ba.allFinite())
        return;
    unique_lock<mutex> lock(mMutexAccBiasPrior);
    mAccBiasPrior = ba;
    mAccBiasPriorInfo = 1.0 / (static_cast<double>(sigma) * static_cast<double>(sigma));
    mbAccBiasPrior = true;
}

void LocalMapping::ClearAccBiasPrior()
{
    unique_lock<mutex> lock(mMutexAccBiasPrior);
    mbAccBiasPrior = false;
}

bool LocalMapping::GetBaroScaleObs(BaroScaleObs &o)
{
    unique_lock<mutex> lock(mMutexBaroObs);
    o = mBaroObs;
    return mBaroObs.seq > 0;
}

bool LocalMapping::GetAccBiasPrior(Eigen::Vector3f &ba, double &info)
{
    unique_lock<mutex> lock(mMutexAccBiasPrior);
    if (!mbAccBiasPrior)
        return false;
    ba = mAccBiasPrior;
    info = mAccBiasPriorInfo;
    return true;
}

void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("local_mapping.imu_initialize.begin prior_g=%f prior_a=%f full_ba=%d",
                priorG, priorA, bFIBA ? 1 : 0);
    if (mbResetRequested)
        return;

    float minTime;
    int nMinKF;
    if (mbMonocular)
    {
        minTime = 2.0;
        nMinKF = 10;
    }
    else
    {
        minTime = 1.0;
        nMinKF = 10;
    }


    if(mpAtlas->KeyFramesInMap()<nMinKF)
        return;

    // Retrieve all keyframe in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    if(vpKF.size()<nMinKF)
        return;

    mFirstTs=vpKF.front()->mTimeStamp;
    if(mpCurrentKeyFrame->mTimeStamp-mFirstTs<minTime)
        return;

    bInitializing = true;

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);

    // Compute and KF velocities mRwg estimation
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized())
    {
        Eigen::Matrix3f Rwg;
        Eigen::Vector3f dirG;
        dirG.setZero();
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            if (!(*itKF)->mPrevKF)
                continue;

            dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            Eigen::Vector3f _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }

        dirG = dirG/dirG.norm();
        mbMapGravityFromRef = false;
        if(mbGravityFromRef)
        {
            Eigen::Vector3f dirRef; int nRef = 0;
            if(GravityDirFromReference(vpKF, dirRef, nRef))
            {
                const float angDeg = std::acos(std::max(-1.f, std::min(1.f, dirRef.dot(dirG)))) * 180.f / static_cast<float>(M_PI);
                std::cerr << "FastTrack IMU init: gravity from the flight controller (" << nRef << " keyframes); the "
                          << "velocity-based estimate was " << angDeg << " deg away" << std::endl;
                FT_CALC_LOG("local_mapping.imu_initialize.gravity_from_reference kfs=%d estimate_error_deg=%f", nRef, angDeg);
                dirG = dirRef;
                mbMapGravityFromRef = true;
            }
            else
                std::cerr << "FastTrack IMU init: no flight-controller attitude for the keyframes; gravity estimated" << std::endl;
        }
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f v = gI.cross(dirG);
        const float nv = v.norm();
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        Eigen::Vector3f vzg = v*ang/nv;
        Rwg = Sophus::SO3f::exp(vzg).matrix();
        mRwg = Rwg.cast<double>();
        mTinit = mpCurrentKeyFrame->mTimeStamp-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = mpCurrentKeyFrame->GetGyroBias().cast<double>();
        mba = mpCurrentKeyFrame->GetAccBias().cast<double>();
    }

    mScale=1.0;
    // B3: baro scale fit whenever metric initialization is enabled; a successful fit makes the map scale metric.
    if(mbMetricScaleFixed || mpTracker->mbMetricInitEnabled)
    {
        double s = 0, c = 0, range = 0, rms = 0; int n = 0;
        const bool fit = mpTracker->BaroScaleFit(mpCurrentKeyFrame->GetMap(), vpKF, s, c, n, range, rms);
        // First metric fix: any positive s (the map scale is arbitrary until now); afterwards corrections within 0.5-2.
        // Consistency: rms within 0.5 m + 5% of the height range; c (pad camera height + baro pad offset) within -2..5 m.
        const bool sOk = mbMetricScaleFixed ? (s > 0.5 && s < 2.0) : (s > 1e-3 && s < 1e3);
        const bool consistent = fit && sOk && rms < 0.5 + 0.05 * range && c > -2.0 && c < 5.0;
        if(consistent)
        {
            mScale = s;
            mbMetricScaleFixed = true;
            mMetricC = c;
            std::cerr << "FastTrack baro scale fit: s=" << s << " pad camera height c=" << c << " m, " << n
                      << " keyframes, height range " << range << " m, rms " << rms << " m" << std::endl;
        }
        else
            std::cerr << "FastTrack baro scale fit: not used (fit " << fit << ", n=" << n << ", range " << range << " m, s=" << s
                      << ", c=" << c << " m, rms " << rms << " m, metric " << mbMetricScaleFixed << ")" << std::endl;
    }

    mInitTime = mpTracker->mLastFrame.mTimeStamp-vpKF.front()->mTimeStamp;

    // Rest-measured gyro bias prior (SetRestGyroBiasPrior), sampled once for this initialization stage.
    Eigen::Vector3f restBg;
    double restBgInfo = 0.0;
    const bool bRestBg = GetRestGyroBiasPrior(restBg, restBgInfo);
    const Eigen::Vector3f *pRestBg = bRestBg ? &restBg : NULL;
    // Init profile: accelerometer bias of a saved initialization replaces the prior towards zero.
    Eigen::Vector3f profBa;
    double profBaInfo = 0.0;
    const bool bProfBa = GetAccBiasPrior(profBa, profBaInfo);
    const Eigen::Vector3f *pProfBa = bProfBa ? &profBa : NULL;
    if (bProfBa)
        std::cerr << "FastTrack IMU init: accelerometer bias from the init profile [" << profBa.transpose()
                  << "] m/s^2, info " << profBaInfo << " (stage prior_a " << priorA << ")" << std::endl;
    FT_CALC_LOG("local_mapping.imu_initialize.rest_gyro_prior used=%d bg=%f,%f,%f info=%f prior_g=%f",
                bRestBg ? 1 : 0, bRestBg ? restBg.x() : 0.f, bRestBg ? restBg.y() : 0.f, bRestBg ? restBg.z() : 0.f,
                restBgInfo, priorG);

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    if(mbMetricScaleFixed)
        std::cerr << "FastTrack IMU init: scale fixed (metric initialization)" << std::endl;
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba, mbMonocular && !mbMetricScaleFixed, infoInertial, false, false, priorG, priorA,
                                    pRestBg, restBgInfo, mbGravityFromRef && mbMapGravityFromRef, pProfBa, profBaInfo);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1)
    {
        FT_CALC_LOG("local_mapping.imu_initialize.abort reason=scale_too_small scale=%f", mScale);
        bInitializing=false;
        return;
    }

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular || mbMetricScaleFixed) {
            Sophus::SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
            mpAtlas->GetCurrentMap()->ApplyScaledRotation(Twg, mScale, true);
            mpTracker->UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
        }

        // Check if initialization OK
        if (!mpAtlas->isImuInitialized())
            for (int i = 0; i < N; i++) {
                KeyFrame *pKF2 = vpKF[i];
                pKF2->bImu = true;
            }
    }

    mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
    if (!mpAtlas->isImuInitialized())
    {
        mpAtlas->SetImuInitialized();
        mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
        mpCurrentKeyFrame->bImu = true;
    }

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (bFIBA)
    {
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, true, priorG, priorA,
                                      NULL, NULL, pRestBg, restBgInfo, pProfBa, profBaInfo);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, false, 1e2, 1e6,
                                      NULL, NULL, pRestBg, restBgInfo, pProfBa, profBaInfo);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    Verbose::PrintMess("Global Bundle Adjustment finished\nUpdating map ...", Verbose::VERBOSITY_NORMAL);

    // Get Map Mutex
    Map* pUpdateLockedMap = mpAtlas->GetCurrentMap();
    unique_lock<mutex> lock(pUpdateLockedMap->mMutexMapUpdate);

    unsigned long GBAid = mpCurrentKeyFrame->mnId;

    // Process keyframes in the queue
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // Correct keyframes starting at map first keyframe
    list<KeyFrame*> lpKFtoCheck(mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.begin(),mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.end());

    while(!lpKFtoCheck.empty())
    {
        KeyFrame* pKF = lpKFtoCheck.front();
        const set<KeyFrame*> sChilds = pKF->GetChilds();
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
        {
            KeyFrame* pChild = *sit;
            if(!pChild || pChild->isBad())
                continue;

            if(pChild->mnBAGlobalForKF!=GBAid)
            {
                Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;

                Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                if(pChild->isVelocitySet()){
                    pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                }
                else {
                    Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);
                }

                pChild->mBiasGBA = pChild->GetImuBias();
                pChild->mnBAGlobalForKF = GBAid;

            }
            lpKFtoCheck.push_back(pChild);
        }

        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);

        if(pKF->bImu)
        {
            pKF->mVwbBefGBA = pKF->GetVelocity();
            pKF->SetVelocity(pKF->mVwbGBA);
            pKF->SetNewBias(pKF->mBiasGBA);
        } else {
            FT_CALC_LOG("local_mapping.imu_initialize.warning keyframe=%llu inertial=0",
                        static_cast<unsigned long long>(pKF->mnId));
        }

        lpKFtoCheck.pop_front();
    }

    // Correct MapPoints
    const vector<MapPoint*> vpMPs = mpAtlas->GetCurrentMap()->GetAllMapPoints();

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        if(pMP->mnBAGlobalForKF==GBAid)
        {
            // If optimized by Global BA, just update
            pMP->SetWorldPos(pMP->mPosGBA);
        }
        else
        {
            // Update according to the correction of its reference keyframe
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

            if(pRefKF->mnBAGlobalForKF!=GBAid)
                continue;

            // Map to non-corrected camera
            Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

            // Backproject using corrected camera
            pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
        }
    }

    Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);

    mnKFs=vpKF.size();
    mIdxInit++;

    // RESEARCH culling-v4b-r1: this thread holds pUpdateLockedMap->mMutexMapUpdate, which
    // SetBadFlag would otherwise lock again. The queue is drained above under that lock, so it is
    // normally empty here; count it so a non-empty case is visible.
    list<KeyFrame*> lPendingKFs;
    {
        unique_lock<mutex> lockNewKFs(mMutexNewKFs);
        lPendingKFs.swap(mlNewKeyFrames);
    }
    if(!lPendingKFs.empty())
        FT_CALC_LOG("research.kf.pending_after_drain n=%llu", static_cast<unsigned long long>(lPendingKFs.size()));
    for(list<KeyFrame*>::iterator lit = lPendingKFs.begin(), lend=lPendingKFs.end(); lit!=lend; lit++)
    {
        KeyFrame* pPendingKF = *lit;
        if(pPendingKF->SetBadFlagMapUpdateHeld(pUpdateLockedMap))
            delete pPendingKF;
        else
            FT_CALC_LOG("research.kf.pending_delete_refused id=%llu action=retain_in_map",
                        static_cast<unsigned long long>(pPendingKF->mnId));
    }

    mpTracker->mState=Tracking::OK;
    bInitializing = false;

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    // I1 (2026-10-30): stage summary on stderr. priorA identifies the stage (1e10 first init, 1e5 BA1, 0 BA2). In a
    // gravity-aligned map a horizontal accelerometer bias b is indistinguishable from a tilt atan(b / g) without
    // rotation excitation; the tilt-equivalent is printed for the 28 Oct tilt analysis (body axes, ~level at init).
    {
        const IMU::Bias b = mpCurrentKeyFrame->GetImuBias();
        const double bxy = std::sqrt(double(b.bax) * b.bax + double(b.bay) * b.bay);
        std::cerr << std::fixed << std::setprecision(4) << "FastTrack IMU init stage: t=" << mpCurrentKeyFrame->mTimeStamp
                  << " priorA=" << std::scientific << std::setprecision(1) << priorA << std::fixed << std::setprecision(4)
                  << " map=" << mpCurrentKeyFrame->GetMap()->GetId() << " kfs=" << mnKFs << " scale=" << mScale
                  << " ba=[" << b.bax << " " << b.bay << " " << b.baz << "] bg=[" << b.bwx << " " << b.bwy << " " << b.bwz
                  << "] |ba_xy|=" << bxy << " tilt_equiv_deg=" << std::setprecision(2) << std::atan2(bxy, 9.81) * 180.0 / M_PI
                  << " metric=" << (mbMetricScaleFixed ? 1 : 0) << std::defaultfloat << std::setprecision(6) << std::endl;
    }
    FT_CALC_LOG("local_mapping.imu_initialize.end scale=%f keyframes=%d", mScale, mnKFs);

    return;
}

void LocalMapping::ScaleRefinement(bool baroOnly)
{
    FT_CALC_SCOPE();
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    // unique_lock<mutex> lock0(mMutexImuInit);
    if (mbResetRequested)
        return;

    // Retrieve all keyframes in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    // B5: barometric scale from the climb (s*d = h + c over all keyframes since the map start, pad included). Tried
    // before the pending keyframes are drained, so a failed baro-only attempt leaves local mapping untouched.
    bool bBaroScale = false; double sBaro = 1.0;
    if(mpTracker->mbMetricInitEnabled && !mbMetricScaleFixed)
    {
        double s = 0, c = 0, range = 0, rms = 0; int n = 0;
        const bool fit = mpTracker->BaroScaleFit(mpCurrentKeyFrame->GetMap(), vpKF, s, c, n, range, rms);
        const bool ok = fit && s > 0.2 && s < 5.0 && range >= 10.0 && rms < 0.5 + 0.05 * range && c > -2.0 && c < 5.0;
        if(ok)
        {
            sBaro = s; bBaroScale = true; mbMetricScaleFixed = true; mMetricC = c;
            std::cerr << "FastTrack climb baro scale: s=" << s << " c=" << c << " m, " << n << " keyframes, height range "
                      << range << " m, rms " << rms << " m" << std::endl;
        }
        else if(baroOnly)
        {
            if(++mClimbFitLog % 4 == 1)
                std::cerr << "FastTrack climb baro scale: not yet (fit " << fit << ", n=" << n << ", range " << range
                          << " m, s=" << s << ", c=" << c << " m, rms " << rms << " m)" << std::endl;
            return;
        }
    }
    else if(baroOnly)
        return;

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();

    mRwg = Eigen::Matrix3d::Identity();
    mScale = bBaroScale ? sBaro : 1.0;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    if(!bBaroScale)
        Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1) // 1e-1
    {
        FT_CALC_LOG("local_mapping.scale_refinement.abort reason=scale_too_small scale=%f", mScale);
        bInitializing=false;
        return;
    }
    
    Sophus::SO3d so3wg(mRwg);
    // Before this line we are not changing the map
    Map* pUpdateLockedMap = mpAtlas->GetCurrentMap();
    unique_lock<mutex> lock(pUpdateLockedMap->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if ((fabs(mScale-1.f)>0.002)||!mbMonocular)
    {
        Sophus::SE3f Tgw(mRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Tgw,mScale,true);
        mpTracker->UpdateFrameIMU(mScale,mpCurrentKeyFrame->GetImuBias(),mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    // RESEARCH culling-v4b-r1: this thread holds pUpdateLockedMap->mMutexMapUpdate, which
    // SetBadFlag would otherwise lock again. The queue is drained above under that lock, so it is
    // normally empty here; count it so a non-empty case is visible.
    list<KeyFrame*> lPendingKFs;
    {
        unique_lock<mutex> lockNewKFs(mMutexNewKFs);
        lPendingKFs.swap(mlNewKeyFrames);
    }
    if(!lPendingKFs.empty())
        FT_CALC_LOG("research.kf.pending_after_drain n=%llu", static_cast<unsigned long long>(lPendingKFs.size()));
    for(list<KeyFrame*>::iterator lit = lPendingKFs.begin(), lend=lPendingKFs.end(); lit!=lend; lit++)
    {
        KeyFrame* pPendingKF = *lit;
        if(pPendingKF->SetBadFlagMapUpdateHeld(pUpdateLockedMap))
            delete pPendingKF;
        else
            FT_CALC_LOG("research.kf.pending_delete_refused id=%llu action=retain_in_map",
                        static_cast<unsigned long long>(pPendingKF->mnId));
    }

    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}



void LocalMapping::CruiseScaleStep(double s, double h, double tilt)
{
    Map* pMap = mpCurrentKeyFrame->GetMap();
    if(pMap != mpCruiseMap) { mCruiseObs.clear(); mCruiseSince = 0; mCruiseTotal = 1.0; mpCruiseMap = pMap; }
    if(h < mpTracker->mCruiseMinHeight || tilt > mpTracker->mCruiseMaxTilt || !(s > 0.2 && s < 5.0))
        return;
    mCruiseObs.push_back(s);
    while((int)mCruiseObs.size() > mpTracker->mCruiseWindow) mCruiseObs.pop_front();
    if(++mCruiseSince < mpTracker->mCruisePeriod || (int)mCruiseObs.size() < mpTracker->mCruiseWindow)
        return;
    std::vector<double> v(mCruiseObs.begin(), mCruiseObs.end());
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    const double sMed = v[v.size() / 2];
    if(std::fabs(sMed - 1.0) <= mpTracker->mCruiseDeadband)
        return;
    if(CheckNewKeyFrames())
        return;  // pending keyframes are not in the map yet and would miss the scaling; retry at the next observation
    const double step = std::max(1.0 - mpTracker->mCruiseMaxStep, std::min(1.0 + mpTracker->mCruiseMaxStep, 1.0 + mpTracker->mCruiseGain * (sMed - 1.0)));
    {
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const Eigen::Vector3f O = mpCurrentKeyFrame->GetCameraCenter();
        // Map::ApplyScaledRotation maps p -> s*R*p + t; R = I, t = (1 - s) O keeps the current camera centre fixed.
        Sophus::SE3f Tyw(Eigen::Matrix3f::Identity(), (1.f - (float)step) * O);
        pMap->ApplyScaledRotation(Tyw, (float)step, true);
        mpTracker->UpdateFrameIMU((float)step, mpCurrentKeyFrame->GetImuBias(), mpCurrentKeyFrame);
    }
    pMap->IncreaseChangeIndex();
    mCruiseTotal *= step;
    for(double &x : mCruiseObs) x /= step;  // earlier observations expressed in the rescaled map
    mCruiseSince = 0;
    std::cerr << std::fixed << std::setprecision(4) << "FastTrack B7 cruise scale: t=" << mpCurrentKeyFrame->mTimeStamp
              << " s_med=" << sMed << " step=" << step << " total=" << mCruiseTotal << " h=" << h << " n=" << v.size()
              << std::defaultfloat << std::setprecision(6) << std::endl;
}

bool LocalMapping::IsInitializing()
{
    return bInitializing;
}


double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

KeyFrame* LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

// C7 (2026-10-31): in-flight accelerometer-bias observer. e = vertical acceleration error of the IMU propagation from
// the previous keyframe (pre-BA velocity of this keyframe minus the optimized velocity of the previous one) against the
// baro vertical velocity change. A short accelerometer (a_m = a - d) gives e = -d along world z; the bias that removes
// it is ba + Rwb^T (0, 0, e). The median over Window s becomes a prior on this keyframe's accelerometer bias, with
// information only along the body direction of world z; the window then restarts so the next estimate measures the
// error left after the correction.
void LocalMapping::AccBiasObserverStep(float vzPre)
{
    KeyFrame* pKF = mpCurrentKeyFrame;
    const double t = pKF->mTimeStamp, W = mpTracker->mBaroVzWindow;
    double h1 = 0, h0 = 0, hPad = 0, hPad0 = 0;
    if(!mpTracker->AltitudeAt(t, h1, hPad) || !mpTracker->AltitudeAt(t - W, h0, hPad0) || h1 - hPad < mpTracker->mAccBiasObsMinHeight)
    {
        mC7.clear();
        return;
    }
    const double vzb = (h1 - h0) / W;
    const unsigned long mapId = pKF->GetMap()->GetId();
    if(mC7.have && mC7.mapId == mapId)
    {
        const double dt = t - mC7.t;
        if(dt > 0.05 && dt < 1.0)
            mC7.e.emplace_back(t, ((double(vzPre) - mC7.vz) - (vzb - mC7.vzb)) / dt);
    }
    else
        mC7.e.clear();
    mC7.have = true; mC7.mapId = mapId; mC7.t = t; mC7.vz = pKF->GetVelocity()(2); mC7.vzb = vzb;
    while(!mC7.e.empty() && t - mC7.e.front().first > mpTracker->mAccBiasObsWindow)
        mC7.e.pop_front();
    if((int)mC7.e.size() < mpTracker->mAccBiasObsMinKFs || t - mC7.e.front().first < 0.8 * mpTracker->mAccBiasObsWindow)
        return;
    std::vector<double> v;
    v.reserve(mC7.e.size());
    for(const auto &x : mC7.e)
        v.push_back(x.second);
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    const double eMed = v[v.size() / 2];
    const double step = std::max(-mpTracker->mAccBiasObsMaxStep, std::min(mpTracker->mAccBiasObsMaxStep, eMed));
    const Eigen::Vector3f u = pKF->GetImuRotation().transpose() * Eigen::Vector3f::UnitZ();  // world z in the body frame
    const Eigen::Vector3f ba = pKF->GetAccBias();
    Eigen::Vector3f target = ba + u * float(step);
    const float along = u.dot(target), maxB = float(mpTracker->mAccBiasObsMaxBias);
    if(std::fabs(along) > maxB)
        target -= u * (along - std::copysign(maxB, along));
    pKF->mAccBiasObs = target;
    pKF->mAccBiasObsDir = u;
    pKF->mAccBiasObsInfo = 1.0 / (mpTracker->mAccBiasObsSigma * mpTracker->mAccBiasObsSigma);
    pKF->mbAccBiasObs = true;
    std::cerr << std::fixed << std::setprecision(3) << "FastTrack C7 t=" << t << " e_med=" << eMed << " step=" << step
              << " n=" << v.size() << " ba=(" << ba(0) << "," << ba(1) << "," << ba(2) << ") target_along=" << u.dot(target)
              << std::defaultfloat << std::setprecision(6) << std::endl;
    mC7.e.clear();
}

// C7c (2026-10-31): the vertical KF's bias b (world z, same gravity constant as the VIO) is what the VIO bias must
// equal along u = R_wb^T z. Shift the bias of the newest keyframes (temporal chain) by the same step so the random-walk
// edges between them stay consistent; tracking takes the bias from the last keyframe.
void LocalMapping::VertKFInject()
{
    double b = 0, sd = 1, h = 0, vz = 0, hb = 0, hPad = 0;
    static int c7cInjDiag = 0;
    const bool kfOk = mpTracker->GetVertKF(b, sd, h, vz);
    const bool altOk = mpTracker->AltitudeAt(mpCurrentKeyFrame->mTimeStamp, hb, hPad);
    if(++c7cInjDiag % 20 == 1)  // C7c diag (log only)
        std::cerr << std::fixed << std::setprecision(3) << "FastTrack C7c inj-diag t=" << mpCurrentKeyFrame->mTimeStamp
                  << " kf=" << (kfOk ? 1 : 0) << " sd=" << sd << " maxsd=" << mpTracker->mVkfMaxSd
                  << " alt=" << (altOk ? 1 : 0) << " h=" << hb - hPad << " minh=" << mpTracker->mVkfMinHeight
                  << " b=" << b << std::defaultfloat << std::setprecision(6) << std::endl;
    if(!kfOk || sd > mpTracker->mVkfMaxSd)
        return;
    if(!altOk || hb - hPad < mpTracker->mVkfMinHeight)
        return;
    b = std::max(-mpTracker->mVkfMaxBias, std::min(mpTracker->mVkfMaxBias, b));
    const Eigen::Vector3f u0 = mpCurrentKeyFrame->GetImuRotation().transpose() * Eigen::Vector3f::UnitZ();
    const IMU::Bias b0 = mpCurrentKeyFrame->GetImuBias();
    const float along = u0.dot(Eigen::Vector3f(b0.bax, b0.bay, b0.baz));
    const float delta = float(b) - along;
    if(std::fabs(delta) < 0.005f)
        return;
    const float step = std::max(-float(mpTracker->mVkfMaxStep), std::min(float(mpTracker->mVkfMaxStep), delta));
    int n = 0;
    for(KeyFrame* pKF = mpCurrentKeyFrame; pKF && n < mpTracker->mVkfChainKFs; pKF = pKF->mPrevKF, n++)
    {
        if(pKF->isBad() || pKF->GetMap() != mpCurrentKeyFrame->GetMap())
            break;
        const Eigen::Vector3f u = pKF->GetImuRotation().transpose() * Eigen::Vector3f::UnitZ();
        const IMU::Bias bk = pKF->GetImuBias();
        const Eigen::Vector3f ba = Eigen::Vector3f(bk.bax, bk.bay, bk.baz) + u * step;
        pKF->SetNewBias(IMU::Bias(ba(0), ba(1), ba(2), bk.bwx, bk.bwy, bk.bwz));
    }
    static int c7cLog = 0;
    if(++c7cLog % 10 == 1)
        std::cerr << std::fixed << std::setprecision(3) << "FastTrack C7c t=" << mpCurrentKeyFrame->mTimeStamp
                  << " kf_b=" << b << " sd=" << sd << " kf_h=" << h << " kf_vz=" << vz << " baro_h=" << hb - hPad
                  << " vio_along=" << along << " step=" << step << " n=" << n
                  << std::defaultfloat << std::setprecision(6) << std::endl;
}

bool LocalMapping::FcRotationAt(double t, Eigen::Matrix3f &Rwb)
{
    unique_lock<mutex> lock(mMutexGravRef);
    if(mdGravRef.empty())
        return false;
    auto it = std::lower_bound(mdGravRef.begin(), mdGravRef.end(), t,
        [](const std::pair<double, Eigen::Matrix3f> &a, double v) { return a.first < v; });
    const std::pair<double, Eigen::Matrix3f> *best = nullptr;
    if(it != mdGravRef.end()) best = &(*it);
    if(it != mdGravRef.begin()) { auto ip = std::prev(it); if(!best || std::fabs(ip->first - t) < std::fabs(best->first - t)) best = &(*ip); }
    if(!best || std::fabs(best->first - t) > 0.1)
        return false;
    Rwb = best->second;
    return true;
}

// C8 (2026-10-31): rotate the map toward level when the averaged ground-plane normal is tilted (flat terrain), and shift
// the accelerometer biases so that f = R^T (a - g) + b is unchanged: with R' = dR R, b' = b + R'^T (g - dR g).
void LocalMapping::TiltCorrStep(double h)
{
    Map* pMap = mpCurrentKeyFrame->GetMap();
    if(pMap != mpTiltMap) { mTiltObs.clear(); mTiltSince = 0; mTiltTotal = 0.0; mpTiltMap = pMap; mTiltSnapped = false; }
    if(h < mpTracker->mTiltCorrMinHeight)
        return;
    Eigen::Vector3f n; int nI = 0, nP = 0;
    if(mpTracker->mTiltCorrSource == 1)  // C8f: true up from the FC attitude, expressed in the VIO world
    {
        Eigen::Matrix3f Rfc;
        if(!mpCurrentKeyFrame->bImu || !FcRotationAt(mpCurrentKeyFrame->mTimeStamp, Rfc))
            return;
        n = (mpCurrentKeyFrame->GetImuRotation() * Rfc.transpose() * Eigen::Vector3f::UnitZ()).normalized();
        if(mpTracker->mTiltCorrTurnHold > 0.0)  // C8f v2: no samples or steps during and shortly after turns
        {
            const double t = mpCurrentKeyFrame->mTimeStamp;
            const double t0 = (mTiltPrevT > 0.0 && t - mTiltPrevT < 5.0) ? mTiltPrevT : t - 0.4;
            mTiltPrevT = t;
            double rateMax = 0.0;
            Eigen::Matrix3f Ra, Rb;
            for(double ta = t0; ta + 0.2 <= t + 1e-6; ta += 0.2)
                if(FcRotationAt(ta, Ra) && FcRotationAt(ta + 0.2, Rb))
                    rateMax = std::max(rateMax, double(Eigen::AngleAxisf(Ra.transpose() * Rb).angle()) * 180.0 / M_PI / 0.2);
            if(rateMax > mpTracker->mTiltCorrTurnRate)
            {
                if(t - mTiltTurnT > mpTracker->mTiltCorrTurnHold)
                    std::cerr << std::fixed << std::setprecision(3) << "FastTrack C8 turn hold: t=" << t << " rate=" << rateMax
                              << std::defaultfloat << std::setprecision(6) << std::endl;
                mTiltTurnT = t;
                mTiltObs.clear();
                mTiltSince = 0;
            }
            if(t - mTiltTurnT < mpTracker->mTiltCorrTurnHold)
                return;
        }
    }
    else if(!mpTracker->LocalGroundNormal(mpCurrentKeyFrame, n, nI, nP) || nP < 60 || nI < mpTracker->mTiltCorrMinInlier * nP)
        return;
    mTiltObs.push_back(n);
    while((int)mTiltObs.size() > mpTracker->mTiltCorrWindow)
        mTiltObs.pop_front();
    const bool snap = mpTracker->mTiltCorrSnapMax > 0.0 && !mTiltSnapped;  // C8s
    if(snap)
    {
        if((int)mTiltObs.size() < mpTracker->mTiltCorrSnapMinObs)
            return;
    }
    else if(++mTiltSince < mpTracker->mTiltCorrPeriod || (int)mTiltObs.size() < mpTracker->mTiltCorrWindow)
        return;
    Eigen::Vector3f m = Eigen::Vector3f::Zero();
    for(const auto &x : mTiltObs) m += x;
    m.normalize();
    const double tilt = std::acos(std::min(1.f, m(2))) * 180.0 / M_PI;
    if(snap && tilt <= mpTracker->mTiltCorrDeadband)
        mTiltSnapped = true;  // C8s: already level, continue with normal steps
    if(snap && mpTracker->mTiltCorrSnapResetDeg > 0.0 && tilt > mpTracker->mTiltCorrSnapResetDeg)
    {
        // C8r: the map is broken, not tilted; start the initialisation again (same path as the bad-IMU reset).
        mTiltSnapped = true;
        std::cerr << std::fixed << std::setprecision(3) << "FastTrack C8 reset: t=" << mpCurrentKeyFrame->mTimeStamp
                  << " tilt=" << tilt << " limit=" << mpTracker->mTiltCorrSnapResetDeg << " h=" << h << " map=" << pMap->GetId()
                  << std::defaultfloat << std::setprecision(6) << std::endl;
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
        mbBadImu = true;
        return;
    }
    if(tilt <= mpTracker->mTiltCorrDeadband || mTiltTotal >= mpTracker->mTiltCorrMaxTotal)
        return;
    if(CheckNewKeyFrames())
        return;  // pending keyframes would miss the rotation; retry at the next observation
    Eigen::Vector3f axis = m.cross(Eigen::Vector3f::UnitZ());
    if(axis.norm() < 1e-6f)
        return;
    axis.normalize();
    const double stepDeg = snap ? std::min(mpTracker->mTiltCorrSnapMax, tilt)  // C8s
                                : std::min(mpTracker->mTiltCorrMaxStep, mpTracker->mTiltCorrGain * tilt);
    const Eigen::Matrix3f dR = Eigen::AngleAxisf(float(stepDeg * M_PI / 180.0), axis).toRotationMatrix();
    const Eigen::Vector3f g(0.f, 0.f, -IMU::GRAVITY_VALUE);
    const Eigen::Vector3f dg = g - dR * g;
    int nKF = 0;
    {
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const Eigen::Vector3f O = mpCurrentKeyFrame->GetCameraCenter();
        Sophus::SE3f Tyw(dR, O - dR * O);  // p -> dR (p - O) + O: the current camera centre stays fixed
        pMap->ApplyScaledRotation(Tyw, 1.0f, false);
        for(KeyFrame* pKF : pMap->GetAllKeyFrames())
        {
            if(!pKF || pKF->isBad() || !pKF->bImu)
                continue;
            const IMU::Bias b = pKF->GetImuBias();
            const Eigen::Vector3f db = pKF->GetImuRotation().transpose() * dg;  // rotation already updated (R')
            pKF->SetNewBias(IMU::Bias(b.bax + db(0), b.bay + db(1), b.baz + db(2), b.bwx, b.bwy, b.bwz));
            nKF++;
        }
        mpTracker->UpdateFrameIMU(1.0f, mpCurrentKeyFrame->GetImuBias(), mpCurrentKeyFrame);
    }
    pMap->IncreaseChangeIndex();
    mTiltTotal += stepDeg;
    if(snap) { mTiltSnapped = true; mTiltObs.clear(); }  // C8s: fresh window after the snap
    for(auto &x : mTiltObs) x = dR * x;
    mTiltSince = 0;
    std::cerr << std::fixed << std::setprecision(3) << "FastTrack C8 tilt: t=" << mpCurrentKeyFrame->mTimeStamp
              << " tilt=" << tilt << " step=" << stepDeg << " total=" << mTiltTotal << " h=" << h << " n=" << mTiltObs.size()
              << " kfs=" << nKF << " dbias=" << dg.norm() << " src=" << mpTracker->mTiltCorrSource << " snap=" << (snap ? 1 : 0) << std::defaultfloat << std::setprecision(6) << std::endl;
}

} //namespace ORB_SLAM
