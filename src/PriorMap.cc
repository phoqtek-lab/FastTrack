/**
 * Prior-map start, session reset that keeps the prior map, mapping-thread hold for map saving (2026-10-16).
 * See include/PriorMap.h. Everything here runs in the tracking thread (the caller of System::Track*), except
 * GetPriorMapStatus(), which only copies the status under its mutex.
 */

#include "Tracking.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <thread>

#include "Atlas.h"
#include "CalculationLogApi.h"
#include "CalculationLogger.h"
#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Map.h"
#include "MapPoint.h"
#include "System.h"

namespace ORB_SLAM3
{

const char* PriorMapStateName(PriorMapStatus::State s)
{
    switch(s)
    {
        case PriorMapStatus::kDisabled: return "DISABLED";
        case PriorMapStatus::kLoadFailed: return "LOAD_FAILED";
        case PriorMapStatus::kRejected: return "REJECTED";
        case PriorMapStatus::kWaiting: return "WAITING";
        case PriorMapStatus::kConfirming: return "CONFIRMING";
        case PriorMapStatus::kHandInit: return "HAND_INIT";
        case PriorMapStatus::kRelocalized: return "RELOCALIZED";
        case PriorMapStatus::kDiscarded: return "DISCARDED";
    }
    return "UNKNOWN";
}

bool Tracking::sbInstallSigintHandler = true;

namespace
{
double RotationAngleDeg(const Eigen::Matrix3f& R)
{
    const double c = std::max(-1.0, std::min(1.0, (static_cast<double>(R.trace()) - 1.0) / 2.0));
    return std::acos(c) * 180.0 / M_PI;
}

bool RelativeClose(const std::vector<float>& a, const std::vector<float>& b, double tol)
{
    if(a.size() != b.size())
        return false;
    for(size_t i = 0; i < a.size(); i++)
    {
        const double scale = std::max(1.0, std::fabs(static_cast<double>(a[i])));
        if(std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])) > tol * scale)
            return false;
    }
    return true;
}
}  // namespace

void Tracking::SetPriorStatus(PriorMapStatus::State s, const std::string& reason)
{
    std::lock_guard<std::mutex> lock(mMutexPriorStatus);
    const bool changed = mPriorStatus.state != s || mPriorStatus.reason != reason;
    mPriorStatus.state = s;
    mPriorStatus.reason = reason;
    if(changed)
        FT_CALC_LOG("prior_map.status state=%s reason=%s", PriorMapStateName(s), reason.c_str());
}

PriorMapStatus Tracking::GetPriorMapStatus()
{
    std::lock_guard<std::mutex> lock(mMutexPriorStatus);
    return mPriorStatus;
}

bool Tracking::PriorMapUsable()
{
    return mpPriorMap && !mpPriorMap->IsBad() && mpPriorMap->KeyFramesInMap() > 0 &&
           mpPriorMap->isImuInitialized() && mpPriorMap->GetIniertialBA2();
}

bool Tracking::HasUsablePriorMap()
{
    return mPriorOpt.enable && PriorMapUsable();
}

void Tracking::BeginPriorSession()
{
    mnSessionFirstKFId = KeyFrame::nNextId;
    mnSessionFirstMPId = MapPoint::nNextId;
    mnSessionFirstMapId = Map::nNextId;
    mnPriorSessionFrames = 0;
    mnPriorConfirm = 0;
    std::lock_guard<std::mutex> lock(mMutexPriorStatus);
    mPriorStatus.attempts = 0;
    mPriorStatus.successes = 0;
    mPriorStatus.last_bow_matches = 0;
    mPriorStatus.best_bow_matches = 0;
    mPriorStatus.last_inliers = 0;
    mPriorStatus.last_tilt_error_deg = -1.0;
    if(mpPriorMap)
    {
        mPriorStatus.map_id = mpPriorMap->GetId();
        mPriorStatus.keyframes = mpPriorMap->KeyFramesInMap();
        mPriorStatus.map_points = mpPriorMap->MapPointsInMap();
    }
}

// Called once by System after the Tracking constructor (camera and IMU calibration parsed) and before the mapping
// threads receive any keyframe. bLoaded: the atlas in mpAtlas was read from loadStatus.path.
void Tracking::ConfigurePriorMap(const PriorMapOptions& opt, const PriorMapStatus& loadStatus, bool bLoaded)
{
    mPriorOpt = opt;
    {
        std::lock_guard<std::mutex> lock(mMutexPriorStatus);
        mPriorStatus = loadStatus;
    }
    if(!opt.enable)
    {
        SetPriorStatus(PriorMapStatus::kDisabled, "prior-map start disabled");
        return;
    }
    if(!bLoaded)
    {
        if(opt.load_path.empty())
            SetPriorStatus(PriorMapStatus::kDisabled, "no prior map configured (load_path empty)");
        // kLoadFailed: the reason set by System is kept
        if(!mpAtlas->GetCurrentMap())
            mpAtlas->CreateNewMap();
        return;
    }

    // Choose the largest inertially initialized map; the others are fragments of the prime session.
    Map* pBest = nullptr;
    for(Map* pMi : mpAtlas->GetAllMaps())
    {
        if(!pMi || pMi->IsBad())
            continue;
        if(!pMi->isImuInitialized() || !pMi->GetIniertialBA2())
            continue;
        if(!pBest || pMi->KeyFramesInMap() > pBest->KeyFramesInMap())
            pBest = pMi;
    }

    std::string reject;
    char buf[256];
    if(!pBest)
        reject = "no inertially initialized map (IMU init + VIBA2) in the file";
    else if(static_cast<int>(pBest->KeyFramesInMap()) < opt.min_keyframes)
    {
        std::snprintf(buf, sizeof(buf), "best map has %lu keyframes, minimum %d",
                      static_cast<unsigned long>(pBest->KeyFramesInMap()), opt.min_keyframes);
        reject = buf;
    }
    else
    {
        // Calibration of the session vs the map. Atlas::AddCamera reuses a loaded camera only when it is equal, so a
        // keyframe camera that differs from the tracker camera means other intrinsics; compare values as well.
        int nCamMismatch = 0, nTbcMismatch = 0, nChecked = 0;
        double worstT = 0.0, worstR = 0.0;
        for(KeyFrame* pKF : pBest->GetAllKeyFrames())
        {
            if(!pKF || pKF->isBad())
                continue;
            nChecked++;
            if(!pKF->mpCamera || !mpCamera || pKF->mpCamera->GetType() != mpCamera->GetType() ||
               !RelativeClose(pKF->mpCamera->getParameters(), mpCamera->getParameters(), opt.camera_tolerance))
                nCamMismatch++;
            const Sophus::SE3f dT = pKF->mImuCalib.mTbc.inverse() * mpImuCalib->mTbc;
            const double t = dT.translation().norm();
            const double r = RotationAngleDeg(dT.rotationMatrix());
            worstT = std::max(worstT, t);
            worstR = std::max(worstR, r);
            if(t > opt.tbc_tolerance_m || r > opt.tbc_tolerance_deg)
                nTbcMismatch++;
        }
        if(nCamMismatch > 0)
        {
            std::snprintf(buf, sizeof(buf), "camera calibration differs from the map (%d of %d keyframes)", nCamMismatch, nChecked);
            reject = buf;
        }
        else if(nTbcMismatch > 0)
        {
            std::snprintf(buf, sizeof(buf), "camera-IMU extrinsics differ from the map (%d of %d keyframes; worst %.4f m, %.3f deg)",
                          nTbcMismatch, nChecked, worstT, worstR);
            reject = buf;
        }
    }

    // Id counters: the atlas stores them, but never let new objects reuse a loaded id.
    unsigned long maxKF = 0, maxMP = 0, maxMap = 0;
    for(Map* pMi : mpAtlas->GetAllMaps())
    {
        if(!pMi)
            continue;
        maxMap = std::max<unsigned long>(maxMap, pMi->GetId());
        for(KeyFrame* pKFi : pMi->GetAllKeyFrames())
            if(pKFi)
                maxKF = std::max<unsigned long>(maxKF, pKFi->mnId);
        for(MapPoint* pMPi : pMi->GetAllMapPoints())
            if(pMPi)
                maxMP = std::max<unsigned long>(maxMP, pMPi->mnId);
    }
    KeyFrame::nNextId = std::max<unsigned long>(KeyFrame::nNextId, maxKF + 1);
    MapPoint::nNextId = std::max<unsigned long>(MapPoint::nNextId, maxMP + 1);
    Map::nNextId = std::max<unsigned long>(Map::nNextId, maxMap + 1);

    if(!reject.empty())
    {
        FT_CALC_LOG("prior_map.reject reason=%s", reject.c_str());
        // Unusable: drop the loaded maps from place recognition so they cannot be merged into the session.
        for(Map* pMi : mpAtlas->GetAllMaps())
        {
            mpKeyFrameDB->clearMap(pMi);
            mpAtlas->SetMapBad(pMi);
        }
        mpAtlas->RemoveBadMaps();
        mpAtlas->CreateNewMap();
        SetPriorStatus(PriorMapStatus::kRejected, reject);
        return;
    }

    unsigned long nDiscarded = 0;
    if(opt.keep_only_best_map)
    {
        for(Map* pMi : mpAtlas->GetAllMaps())
        {
            if(pMi == pBest)
                continue;
            mpKeyFrameDB->clearMap(pMi);
            mpAtlas->SetMapBad(pMi);
            nDiscarded++;
        }
        mpAtlas->RemoveBadMaps();
    }

    mpPriorMap = pBest;
    mpAtlas->ChangeMap(pBest);
    BeginPriorSession();
    std::snprintf(buf, sizeof(buf), "prior map %lu loaded (%lu keyframes, %lu points, %lu other maps discarded); "
                  "waiting for a stationary frame", static_cast<unsigned long>(pBest->GetId()),
                  static_cast<unsigned long>(pBest->KeyFramesInMap()), static_cast<unsigned long>(pBest->MapPointsInMap()),
                  nDiscarded);
    FT_CALC_LOG("prior_map.configured map=%lu keyframes=%lu points=%lu discarded=%lu next_kf=%lu next_mp=%lu next_map=%lu",
                static_cast<unsigned long>(pBest->GetId()), static_cast<unsigned long>(pBest->KeyFramesInMap()),
                static_cast<unsigned long>(pBest->MapPointsInMap()), nDiscarded,
                static_cast<unsigned long>(KeyFrame::nNextId), static_cast<unsigned long>(MapPoint::nNextId),
                static_cast<unsigned long>(Map::nNextId));
    SetPriorStatus(PriorMapStatus::kWaiting, buf);
}

void Tracking::PriorMapWaitFrame()
{
    // No keyframe in this session yet: keep the keyframe preintegrator empty (otherwise it accumulates every
    // measurement since start-up with the first frame's dt).
    if(mpImuPreintegratedFromLastKF)
        mpImuPreintegratedFromLastKF->Initialize(IMU::Bias());
}

void Tracking::PriorMapClearFrameMatches()
{
    std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(NULL));
    std::fill(mCurrentFrame.mvbOutlier.begin(), mCurrentFrame.mvbOutlier.end(), false);
}

// NOT_INITIALIZED, IMU monocular, prior map configured. pCurrentMap may be changed (session map / prior map).
Tracking::PriorStep Tracking::PriorMapStep(Map*& pCurrentMap, bool bSessionTracking)
{
    if(!mPriorOpt.enable || !mpPriorMap)
        return kPriorNone;
    if(!PriorMapUsable())
    {
        mpPriorMap = nullptr;
        SetPriorStatus(PriorMapStatus::kDiscarded, "prior map no longer usable (bad, empty or not inertial)");
        return kPriorNone;
    }

    char buf[400];
    const bool bPriorActive = (pCurrentMap == mpPriorMap);
    const bool bStill = mCurrentFrame.IsStationary();
    mnPriorSessionFrames++;

    // The IMU preintegration and the ZUPT detector need about a second of frames.
    if(mnPriorSessionFrames <= static_cast<unsigned long>(std::max(mMaxFrames, 1)) + 2)
    {
        if(bSessionTracking)
            return kPriorNone;
        PriorMapWaitFrame();
        return kPriorWait;
    }

    // Moving attempt (reloc_while_moving): from a session map, at a low rotation rate. With an inertial session map
    // its velocity is carried over; otherwise only quasi-still frames qualify (velocity 0 at the switch).
    IMU::Preintegrated* pPreF = mCurrentFrame.mpImuPreintegratedFrame;
    double rate = -1.0;
    // Raw integrated rotation (not GetDeltaRotation: its bias correction aborted on NaN during the lift).
    const bool bPreOK = pPreF && pPreF->dT > 1e-4f && std::isfinite(pPreF->dT) && pPreF->dR.allFinite();
    if(bPreOK)
        rate = RotationAngleDeg(pPreF->dR) * M_PI / 180.0 / pPreF->dT;
    const bool bSessionInertial = bSessionTracking && pCurrentMap->isImuInitialized() && mLastFrame.HasVelocity();
    const bool bMovingOK = mPriorOpt.reloc_while_moving && !bPriorActive && !bStill && bPreOK && rate >= 0.0 &&
        rate < mPriorOpt.moving_max_rate_rad_s && (bSessionInertial || rate < mPriorOpt.quasi_still_rate_rad_s);
    if(bSessionTracking && !bStill && !bMovingOK)
        return kPriorNone;   // keep tracking the session map
    if(bMovingOK && mnPriorConfirm == 0 && (mnPriorSessionFrames % std::max(1, mPriorOpt.moving_probe_every)) != 0)
        return kPriorNone;   // rate-limited until an attempt succeeds, then every frame until confirmed

    if(mPriorOpt.require_stationary && !bStill && !bMovingOK)
    {
        mnPriorConfirm = 0;
        if(mPriorOpt.allow_hand_init)
        {
            if(bPriorActive)
            {
                // Motion initialization must never add keyframes to the prior map: start a session map.
                mpAtlas->CreateNewMap();
                mpAtlas->SetInertialSensor();
                pCurrentMap = mpAtlas->GetCurrentMap();
                mbReadyToInitializate = false;
                mvIniMatches.clear();
                FT_CALC_LOG("prior_map.hand_init session_map=%lu frame=%llu", static_cast<unsigned long>(pCurrentMap->GetId()),
                            static_cast<unsigned long long>(mCurrentFrame.mnId));
            }
            SetPriorStatus(PriorMapStatus::kHandInit,
                           "vehicle moving: motion initialization in a session map; relocalization is retried when stationary");
            return kPriorNone;
        }
        SetPriorStatus(PriorMapStatus::kWaiting, "waiting for the vehicle to be stationary");
        PriorMapWaitFrame();
        return kPriorWait;
    }

    // Stationary (or stationarity not required): relocalize into the prior map.
    bool bOK = false;
    {
        unique_lock<mutex> lockPrior(mpPriorMap->mMutexMapUpdate);
        bOK = Relocalization(mpPriorMap);
    }
    {
        std::lock_guard<std::mutex> lock(mMutexPriorStatus);
        mPriorStatus.attempts++;
        mPriorStatus.last_bow_matches = mnRelocBestBowMatches;
        mPriorStatus.best_bow_matches = std::max(mPriorStatus.best_bow_matches, mnRelocBestBowMatches);
        if(bOK)
        {
            mPriorStatus.successes++;
            mPriorStatus.last_inliers = mnRelocInliers;
        }
    }

    const bool bFallThrough = !bPriorActive && mPriorOpt.allow_hand_init;
    auto notAccepted = [&](PriorMapStatus::State s, const std::string& why) -> PriorStep {
        SetPriorStatus(s, why);
        PriorMapClearFrameMatches();
        if(bFallThrough)
            return kPriorNone;  // keep the motion initialization of the session map going
        PriorMapWaitFrame();
        return kPriorWait;
    };

    if(!bOK)
    {
        mnPriorConfirm = 0;
        std::snprintf(buf, sizeof(buf), "%s, not relocalized: candidates %d, best BoW %d (15 needed), PnP inliers %d, optimized %d (50 needed); "
                      "best BoW this session %d, attempts %lu", bStill ? "stationary" : "moving", mnRelocCandidates,
                      mnRelocBestBowMatches, mnRelocBestPnpInliers, mnRelocBestGood,
                      GetPriorMapStatus().best_bow_matches, GetPriorMapStatus().attempts);
        FT_CALC_LOG("prior_map.reloc_fail moving=%d cand=%d bow=%d pnp=%d good=%d", bStill ? 0 : 1, mnRelocCandidates,
                    mnRelocBestBowMatches, mnRelocBestPnpInliers, mnRelocBestGood);
        if(mnRelocBestBowMatches >= 15)
            std::cerr << "FastTrack prior map: " << buf << std::endl;
        return notAccepted(PriorMapStatus::kWaiting, buf);
    }

    // Gravity check: the relocalized attitude must agree with the measured specific force (stationary).
    const Sophus::SE3f Twb = mCurrentFrame.GetImuPose();
    double tilt = -1.0;
    IMU::Preintegrated* pPre = mCurrentFrame.mpImuPreintegratedFrame;
    if(pPre && pPre->dT > 1e-4f && pPre->dT < 0.5f)
    {
        const Eigen::Vector3f fb = pPre->dV / pPre->dT + Eigen::Vector3f(pPre->b.bax, pPre->b.bay, pPre->b.baz);
        const Eigen::Vector3f fw = Twb.rotationMatrix() * fb;
        const float n = fw.norm();
        if(n > 8.5f && n < 11.0f)
            tilt = std::acos(std::max(-1.0f, std::min(1.0f, fw.z() / n))) * 180.0 / M_PI;
        else
        {
            mnPriorConfirm = 0;
            std::snprintf(buf, sizeof(buf), "relocalized but the specific force is %.2f m/s^2 (not stationary)", n);
            return notAccepted(PriorMapStatus::kWaiting, buf);
        }
    }
    else
    {
        mnPriorConfirm = 0;
        return notAccepted(PriorMapStatus::kWaiting, "relocalized but no IMU preintegration for this frame");
    }
    {
        std::lock_guard<std::mutex> lock(mMutexPriorStatus);
        mPriorStatus.last_tilt_error_deg = tilt;
    }
    if(tilt > mPriorOpt.max_tilt_error_deg)
    {
        mnPriorConfirm = 0;
        std::snprintf(buf, sizeof(buf), "relocalized pose rejected: gravity disagrees by %.2f deg (max %.2f)", tilt,
                      mPriorOpt.max_tilt_error_deg);
        FT_CALC_LOG("prior_map.reject_pose frame=%llu tilt_deg=%f", static_cast<unsigned long long>(mCurrentFrame.mnId), tilt);
        return notAccepted(PriorMapStatus::kWaiting, buf);
    }

    // Confirmation: consecutive stationary relocalizations must agree. Moving: consecutive frames, the relocalized
    // rotation must follow the gyro and the position step stay small (also rejects a wrong cell of a repetitive pad).
    if(mnPriorConfirm > 0 && mbPriorConfirmMoving != !bStill)
        mnPriorConfirm = 0;
    if(mnPriorConfirm > 0 && bStill)
    {
        const Sophus::SE3f d = mPriorConfirmTwb.inverse() * Twb;
        if(d.translation().norm() > mPriorOpt.confirm_max_translation_m ||
           RotationAngleDeg(d.rotationMatrix()) > mPriorOpt.confirm_max_rotation_deg)
            mnPriorConfirm = 0;
    }
    else if(mnPriorConfirm > 0)
    {
        const double dtc = mCurrentFrame.mTimeStamp - mPriorConfirmT;
        const Eigen::Matrix3f Rrel = mPriorConfirmTwb.rotationMatrix().transpose() * Twb.rotationMatrix();
        const Eigen::Matrix3f Rgyro = pPreF->dR;
        const double rotErr = RotationAngleDeg(Rrel.transpose() * Rgyro);
        const double step = (Twb.translation() - mPriorConfirmTwb.translation()).norm();
        if(dtc <= 0.0 || dtc > 0.1 || rotErr > mPriorOpt.moving_confirm_max_rotation_deg ||
           step > mPriorOpt.moving_confirm_max_step_m)
        {
            FT_CALC_LOG("prior_map.moving_confirm_reset dt=%f rot_err_deg=%f step_m=%f", dtc, rotErr, step);
            mnPriorConfirm = 0;
        }
    }
    mbPriorConfirmMoving = !bStill;
    mPriorConfirmT = mCurrentFrame.mTimeStamp;
    if(mnPriorConfirm == 0)
        mPriorConfirmTwb = Twb;
    mnPriorConfirm++;
    FT_CALC_LOG("prior_map.reloc frame=%llu t=%f inliers=%d bow=%d tilt_deg=%f confirm=%d/%d twb=%f,%f,%f",
                static_cast<unsigned long long>(mCurrentFrame.mnId), mCurrentFrame.mTimeStamp, mnRelocInliers,
                mnRelocBestBowMatches, tilt, mnPriorConfirm, mPriorOpt.confirm_frames,
                Twb.translation().x(), Twb.translation().y(), Twb.translation().z());
    const int nConfirmNeed = mnRelocInliers < 50 ? std::max(mPriorOpt.confirm_frames, mPriorOpt.relaxed_confirm_frames)
                                                 : mPriorOpt.confirm_frames;
    if(mnPriorConfirm < std::max(1, nConfirmNeed))
    {
        std::snprintf(buf, sizeof(buf), "relocalized (%d inliers, gravity %.2f deg); confirming %d/%d", mnRelocInliers, tilt,
                      mnPriorConfirm, nConfirmNeed);
        return notAccepted(PriorMapStatus::kConfirming, buf);
    }

    // Accept. Track in the prior map; an empty session map (motion initialization never completed) is left unused.
    if(!bPriorActive)
    {
        FT_CALC_LOG("prior_map.switch from_map=%lu to_map=%lu", static_cast<unsigned long>(pCurrentMap->GetId()),
                    static_cast<unsigned long>(mpPriorMap->GetId()));
        mpAtlas->ChangeMap(mpPriorMap);
        pCurrentMap = mpPriorMap;
        mbReadyToInitializate = false;
        mvIniMatches.clear();
    }

    // IMU state: at rest (velocity 0), bias of the keyframe that shares most of the matched points.
    std::map<KeyFrame*, int> votes;
    int nMatched = 0;
    for(int i = 0; i < mCurrentFrame.N; i++)
    {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if(!pMP || pMP->isBad() || mCurrentFrame.mvbOutlier[i])
            continue;
        nMatched++;
        for(const auto& ob : pMP->GetObservations())
            votes[ob.first]++;
    }
    KeyFrame* pRef = nullptr;
    int nBestVotes = 0;
    for(const auto& v : votes)
        if(v.second > nBestVotes && !v.first->isBad() && v.first->GetMap() == mpPriorMap)
        {
            nBestVotes = v.second;
            pRef = v.first;
        }
    if(!pRef)
    {
        mnPriorConfirm = 0;
        return notAccepted(PriorMapStatus::kWaiting, "relocalized but no reference keyframe in the prior map");
    }
    const IMU::Bias b = pRef->GetImuBias();
    mCurrentFrame.mpImuPreintegrated = new IMU::Preintegrated(b, mCurrentFrame.mImuCalib);
    mCurrentFrame.SetNewBias(b);
    mCurrentFrame.mPredBias = b;
    // Velocity: zero at rest / quasi-still; from an inertial session map rotated into the prior map's frame.
    Eigen::Vector3f vAccept = Eigen::Vector3f::Zero();
    if(!bStill && bSessionInertial && bPreOK && mLastFrame.GetVelocity().allFinite())
    {
        const Eigen::Matrix3f RwbSessNow = mLastFrame.GetImuPose().rotationMatrix() * pPreF->dR;
        const Eigen::Matrix3f Rps = Twb.rotationMatrix() * RwbSessNow.transpose();
        vAccept = Rps * mLastFrame.GetVelocity();
    }
    mCurrentFrame.SetVelocity(vAccept);
    mpReferenceKF = pRef;
    mCurrentFrame.mpReferenceKF = pRef;
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);  // the first keyframe starts a new temporal chain
    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(b, mCurrentFrame.mImuCalib);
    mLastBias = b;
    mbVelocity = false;
    mState = OK;
    mnPriorConfirm = 0;
    {
        std::lock_guard<std::mutex> lock(mMutexPriorStatus);
        mPriorStatus.reloc_time = mCurrentFrame.mTimeStamp;
    }
    {
        KeyFrame* pNear = nullptr; double dtNear = 1e18;
        for(KeyFrame* pK : mpPriorMap->GetAllKeyFrames())
            if(pK && !pK->isBad() && std::fabs(pK->mTimeStamp - mCurrentFrame.mTimeStamp) < dtNear)
            { dtNear = std::fabs(pK->mTimeStamp - mCurrentFrame.mTimeStamp); pNear = pK; }
        std::cerr << "FastTrack prior map: RELOCALIZED at t=" << std::fixed << mCurrentFrame.mTimeStamp << " (" << (bStill ? "stationary" : "moving")
                  << ", inliers " << mnRelocInliers << ", gravity " << tilt << " deg)";
        if(pNear)
            std::cerr << "; nearest-in-time prior keyframe dt=" << dtNear << " s, position difference "
                      << (pNear->GetImuPosition() - Twb.translation()).norm() << " m, rotation difference "
                      << RotationAngleDeg(pNear->GetImuRotation().transpose() * Twb.rotationMatrix()) << " deg";
        std::cerr << std::endl;
    }
    FT_CALC_LOG("prior_map.accept frame=%llu t=%f matched=%d ref_kf=%lu votes=%d tilt_deg=%f twb=%f,%f,%f bg=%f,%f,%f ba=%f,%f,%f",
                static_cast<unsigned long long>(mCurrentFrame.mnId), mCurrentFrame.mTimeStamp, nMatched,
                static_cast<unsigned long>(pRef->mnId), nBestVotes, tilt, Twb.translation().x(), Twb.translation().y(),
                Twb.translation().z(), b.bwx, b.bwy, b.bwz, b.bax, b.bay, b.baz);
    return kPriorRelocalized;
}

// Relocalized frame tracked the local map: its keyframe opens the session chain in the prior map.
void Tracking::PriorMapOnRelocalizedKeyFrame()
{
    CreateNewKeyFrame();
    // The next frame predicts from this keyframe and runs the last-keyframe inertial optimization.
    mbPriorForceMapUpdated = true;
    // The IMU state is seeded: close the visual-only post-relocalization window.
    mnLastRelocFrameId = 0;
    mnPriorConfirm = 0;
    char buf[200];
    std::snprintf(buf, sizeof(buf), "tracking in prior map %lu since t=%.3f (keyframe %lu, %d inliers)",
                  static_cast<unsigned long>(mpPriorMap ? mpPriorMap->GetId() : 0), mCurrentFrame.mTimeStamp,
                  static_cast<unsigned long>(mpLastKeyFrame ? mpLastKeyFrame->mnId : 0), mnMatchesInliers);
    {
        std::lock_guard<std::mutex> lock(mMutexPriorStatus);
        mPriorStatus.relocalizations++;
    }
    SetPriorStatus(PriorMapStatus::kRelocalized, buf);
    FT_CALC_LOG("prior_map.keyframe id=%lu inliers=%d", static_cast<unsigned long>(mpLastKeyFrame ? mpLastKeyFrame->mnId : 0),
                mnMatchesInliers);
}

void Tracking::PriorMapOnLocalMapFail()
{
    FT_CALC_LOG("prior_map.local_map_fail frame=%llu inliers=%d", static_cast<unsigned long long>(mCurrentFrame.mnId),
                mnMatchesInliers);
    mState = NOT_INITIALIZED;
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mnPriorConfirm = 0;
    PriorMapWaitFrame();
    char buf[160];
    std::snprintf(buf, sizeof(buf), "relocalized but the local map tracked only %d inliers; retrying", mnMatchesInliers);
    SetPriorStatus(PriorMapStatus::kWaiting, buf);
}

void Tracking::PriorMapNoteMapReset(Map* pMap, const char* why)
{
    if(!mpPriorMap || (pMap && pMap != mpPriorMap))
        return;
    mpPriorMap = nullptr;
    SetPriorStatus(PriorMapStatus::kDiscarded, std::string("prior map cleared: ") + why);
}

// Holds LocalMapping stopped and LoopClosing between cycles with an empty queue; waits for a running global BA.
bool Tracking::StopMappingThreads(bool bResetLocalMapper, double gbaWaitS, std::string& reason)
{
    if(bResetLocalMapper)
        mpLocalMapper->RequestReset();  // drops queued keyframes of this session; blocks until executed
    mpLocalMapper->RequestStop();
    const auto t0 = std::chrono::steady_clock::now();
    while(!mpLocalMapper->isStopped())
    {
        if(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 10.0)
        {
            mpLocalMapper->Release();
            reason = "local mapping did not stop within 10 s";
            return false;
        }
        usleep(1000);
    }
    mpLoopClosing->RequestReset();  // returns between loop-closing cycles; clears its keyframe queue
    while(mpLoopClosing->isRunningGBA())
    {
        if(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > gbaWaitS)
        {
            mpLocalMapper->Release();
            reason = "global bundle adjustment still running";
            return false;
        }
        usleep(5000);
    }
    return true;
}

void Tracking::ReleaseMappingThreads()
{
    mpLocalMapper->Release();
}

// Session reset that keeps the prior map: removes this session's keyframes, points and maps, then waits for a
// stationary relocalization again. Tracking thread only (System::TrackMonocular reset block).
bool Tracking::ResetToPriorMap(std::string& reason)
{
    if(!HasUsablePriorMap())
    {
        reason = "no usable prior map";
        return false;
    }
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }
    if(!StopMappingThreads(true, mPriorOpt.gba_wait_s, reason))
    {
        if(mpViewer)
            mpViewer->Release();
        return false;
    }

    unsigned long nMaps = 0, nKFs = 0, nMPs = 0, nResidual = 0;
    for(Map* pMi : mpAtlas->GetAllMaps())
    {
        if(pMi == mpPriorMap || pMi->GetId() < mnSessionFirstMapId)
            continue;
        mpKeyFrameDB->clearMap(pMi);
        pMi->clear();
        mpAtlas->SetMapBad(pMi);
        nMaps++;
    }
    mpAtlas->RemoveBadMaps();
    mpAtlas->ChangeMap(mpPriorMap);

    std::vector<KeyFrame*> vKFs = mpPriorMap->GetAllKeyFrames();
    std::sort(vKFs.begin(), vKFs.end(), [](KeyFrame* a, KeyFrame* b) { return a->mnId > b->mnId; });
    for(KeyFrame* pKF : vKFs)
    {
        if(!pKF || pKF->isBad() || pKF->mnId < mnSessionFirstKFId)
            continue;
        pKF->SetErase();
        if(pKF->SetBadFlag() || pKF->isBad())
            nKFs++;
        else
            nResidual++;
    }
    for(MapPoint* pMP : mpPriorMap->GetAllMapPoints())
    {
        if(!pMP || pMP->isBad() || pMP->mnId < mnSessionFirstMPId)
            continue;
        pMP->SetBadFlag();
        nMPs++;
    }

    mState = NO_IMAGES_YET;
    mbReadyToInitializate = false;
    mbSetInit = false;
    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mnLastRelocFrameId = 0;
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();
    mbVelocity = false;
    mbPriorForceMapUpdated = false;
    if(mpImuPreintegratedFromLastKF)
        delete mpImuPreintegratedFromLastKF;
    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);

    ReleaseMappingThreads();
    if(mpViewer)
        mpViewer->Release();

    BeginPriorSession();
    {
        std::lock_guard<std::mutex> lock(mMutexPriorStatus);
        mPriorStatus.resets++;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "reset to prior map: removed %lu session keyframes (%lu not removable), %lu points, %lu maps; "
                  "waiting for a stationary frame", nKFs, nResidual, nMPs, nMaps);
    reason = buf;
    FT_CALC_LOG("prior_map.reset keyframes=%lu residual=%lu points=%lu maps=%lu prior_keyframes=%lu",
                nKFs, nResidual, nMPs, nMaps, static_cast<unsigned long>(mpPriorMap->KeyFramesInMap()));
    SetPriorStatus(PriorMapStatus::kWaiting, buf);
    return true;
}

// After a successful save: the saved (active) map becomes the prior map of this process.
void Tracking::AdoptActiveMapAsPrior(const SaveMapResult& r)
{
    Map* pActive = mpAtlas->GetCurrentMap();
    if(!mPriorOpt.enable || !mPriorOpt.adopt_after_save || !pActive || pActive->IsBad() || !pActive->isImuInitialized() ||
       !pActive->GetIniertialBA2())
        return;
    mpPriorMap = pActive;
    BeginPriorSession();
    {
        std::lock_guard<std::mutex> lock(mMutexPriorStatus);
        mPriorStatus.path = r.path;
        mPriorStatus.file_bytes = r.bytes;
        mPriorStatus.file_md5 = r.md5;
    }
    char buf[200];
    std::snprintf(buf, sizeof(buf), "saved map %lu adopted as the prior map", static_cast<unsigned long>(pActive->GetId()));
    SetPriorStatus(mState == OK ? PriorMapStatus::kRelocalized : PriorMapStatus::kWaiting, buf);
}

void Tracking::NotePriorSave(const SaveMapResult& r)
{
    std::lock_guard<std::mutex> lock(mMutexPriorStatus);
    if(r.ok)
        mPriorStatus.saves++;
    char buf[400];
    if(r.ok)
        std::snprintf(buf, sizeof(buf), "ok %s bytes=%llu md5=%s maps=%lu keyframes=%lu points=%lu seconds=%.2f", r.path.c_str(),
                      static_cast<unsigned long long>(r.bytes), r.md5.c_str(), r.maps, r.keyframes, r.map_points, r.seconds);
    else
        std::snprintf(buf, sizeof(buf), "failed: %s", r.reason.c_str());
    mPriorStatus.last_save_result = buf;
}

} // namespace ORB_SLAM3
