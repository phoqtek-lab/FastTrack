#include <cstdio>
#include <mutex>
#include <set>
namespace ORB_SLAM3 {
}
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

#include "KeyFrame.h"
#include "Converter.h"
#include "ImuTypes.h"
#include "CalculationLogger.h"
#include<mutex>

namespace ORB_SLAM3
{

bool TreeParentAllowed(KeyFrame* pChild, KeyFrame* pParent, bool bAllowMergeTransition)
{
    if(!pChild || !pParent || pChild == pParent || pChild->isBad() || pParent->isBad())
        return false;

    Map* pChildMap = pChild->GetMap();
    Map* pParentMap = pParent->GetMap();
    if(!pChildMap || !pParentMap || (!bAllowMergeTransition && pChildMap != pParentMap))
        return false;
    if(!bAllowMergeTransition && pChild->mnId == pChildMap->GetInitKFid())
        return false; // The map's original root is only demoted by map merge.

    std::set<KeyFrame*> visited;
    KeyFrame* pWalk = pParent;
    while(pWalk)
    {
        if(pWalk == pChild || !visited.insert(pWalk).second || pWalk->isBad())
            return false;
        if(!bAllowMergeTransition && pWalk->GetMap() != pChildMap)
            return false;
        pWalk = pWalk->mpParent; // Caller holds KeyFrame::SpanningTreeMutex().
    }

    KeyFrame* pRoot = pParent;
    while(pRoot->mpParent)
        pRoot = pRoot->mpParent;
    Map* pRootMap = pRoot->GetMap();
    if(!pRootMap || pRoot->mnId != pRootMap->GetInitKFid())
        return false;
    return bAllowMergeTransition || pRootMap == pChildMap;
}

bool ChangeParentUnderTreeLock(KeyFrame* pChild, KeyFrame* pParent, bool bAllowMergeTransition)
{
    if(!TreeParentAllowed(pChild, pParent, bAllowMergeTransition))
        return false;

    if(pChild->mpParent == pParent)
    {
        pParent->mspChildrens.insert(pChild);
        return true;
    }

    if(pChild->mpParent)
        pChild->mpParent->mspChildrens.erase(pChild);
    pChild->mpParent = pParent;
    pParent->mspChildrens.insert(pChild);
    return true;
}

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame():
        mnFrameId(0),  mTimeStamp(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
        mfGridElementWidthInv(0), mfGridElementHeightInv(0),
        mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0),
        mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnMergeQuery(0), mnMergeWords(0), mnBAGlobalForKF(0),
        fx(0), fy(0), cx(0), cy(0), invfx(0), invfy(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
        mbf(0), mb(0), mThDepth(0), N(0), mvKeys(static_cast<vector<cv::KeyPoint> >(NULL)), mvKeysUn(static_cast<vector<cv::KeyPoint> >(NULL)),
        mvuRight(static_cast<vector<float> >(NULL)), mvDepth(static_cast<vector<float> >(NULL)), mnScaleLevels(0), mfScaleFactor(0),
        mfLogScaleFactor(0), mvScaleFactors(0), mvLevelSigma2(0), mvInvLevelSigma2(0), mnMinX(0), mnMinY(0), mnMaxX(0),
        mnMaxY(0), mPrevKF(static_cast<KeyFrame*>(NULL)), mNextKF(static_cast<KeyFrame*>(NULL)), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
        mbToBeErased(false), mbBad(false), mHalfBaseline(0), mbCurrentPlaceRecognition(false), mnMergeCorrectedForKF(0),
        NLeft(0),NRight(0), mnNumberOfOpt(0), mbHasVelocity(false)
{
    FT_CALC_SCOPE();
}

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    bImu(pMap->isImuInitialized()), mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK_(F.mK_), mPrevKF(NULL), mNextKF(NULL), mpImuPreintegrated(F.mpImuPreintegrated),
    mImuCalib(F.mImuCalib), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mDistCoef(F.mDistCoef), mbNotErase(false), mnDataset(F.mnDataset),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap), mbCurrentPlaceRecognition(false), mNameFile(F.mNameFile), mnMergeCorrectedForKF(0),
    mpCamera(F.mpCamera), mpCamera2(F.mpCamera2),
    mvLeftToRightMatch(F.mvLeftToRightMatch),mvRightToLeftMatch(F.mvRightToLeftMatch), mTlr(F.GetRelativePoseTlr()),
    mvKeysRight(F.mvKeysRight), NLeft(F.Nleft), NRight(F.Nright), mTrl(F.GetRelativePoseTrl()), mnNumberOfOpt(0), mbHasVelocity(false)
{
    FT_CALC_SCOPE();
    mnId=nNextId++;
    FT_CALC_LOG("keyframe.construct id=%llu frame=%llu map=%llu features=%d imu=%d",
                static_cast<unsigned long long>(mnId), static_cast<unsigned long long>(mnFrameId),
                static_cast<unsigned long long>(pMap->GetId()), N, bImu ? 1 : 0);

    mGrid.resize(mnGridCols);
    if(F.Nleft != -1)  mGridRight.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        if(F.Nleft != -1) mGridRight[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++){
            mGrid[i][j] = F.mGrid[i][j];
            if(F.Nleft != -1){
                mGridRight[i][j] = F.mGridRight[i][j];
            }
        }
    }



    if(!F.HasVelocity()) {
        mVw.setZero();
        mbHasVelocity = false;
    }
    else
    {
        mVw = F.GetVelocity();
        mbHasVelocity = true;
    }

    mImuBias = F.mImuBias;
    SetPose(F.GetPose());

    mnOriginMapId = pMap->GetId();
}

void KeyFrame::ComputeBoW()
{
    FT_CALC_SCOPE();
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
        FT_CALC_LOG("keyframe.bow.computed id=%llu features=%d words=%llu nodes=%llu",
                    static_cast<unsigned long long>(mnId), N,
                    static_cast<unsigned long long>(mBowVec.size()),
                    static_cast<unsigned long long>(mFeatVec.size()));
    }
}

void KeyFrame::SetPose(const Sophus::SE3f &Tcw)
{
    FT_CALC_SCOPE();
    unique_lock<mutex> lock(mMutexPose);

    mTcw = Tcw;
    mRcw = mTcw.rotationMatrix();
    mTwc = mTcw.inverse();
    mRwc = mTwc.rotationMatrix();

    if (mImuCalib.mbIsSet) // TODO Use a flag instead of the OpenCV matrix
    {
        mOwb = mRwc * mImuCalib.mTcb.translation() + mTwc.translation();
    }
}

void KeyFrame::SetVelocity(const Eigen::Vector3f &Vw)
{
    FT_CALC_SCOPE();
    unique_lock<mutex> lock(mMutexPose);
    mVw = Vw;
    mbHasVelocity = true;
}

Sophus::SE3f KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw;
}

Sophus::SE3f KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc;
}

Eigen::Vector3f KeyFrame::GetCameraCenter(){
    unique_lock<mutex> lock(mMutexPose);
    return mTwc.translation();
}

Eigen::Vector3f KeyFrame::GetImuPosition()
{
    unique_lock<mutex> lock(mMutexPose);
    return mOwb;
}

Eigen::Matrix3f KeyFrame::GetImuRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return (mTwc * mImuCalib.mTcb).rotationMatrix();
}

Sophus::SE3f KeyFrame::GetImuPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc * mImuCalib.mTcb;
}

Eigen::Matrix3f KeyFrame::GetRotation(){
    unique_lock<mutex> lock(mMutexPose);
    return mRcw;
}

Eigen::Vector3f KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw.translation();
}

Eigen::Vector3f KeyFrame::GetVelocity()
{
    unique_lock<mutex> lock(mMutexPose);
    return mVw;
}

bool KeyFrame::isVelocitySet()
{
    unique_lock<mutex> lock(mMutexPose);
    return mbHasVelocity;
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    FT_CALC_SCOPE();
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        if(!vPairs[i].second->isBad())
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
    {
        return vector<KeyFrame*>();
    }

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);

    if(it==mvOrderedWeights.end() && mvOrderedWeights.back() < w)
    {
        return vector<KeyFrame*>();
    }
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

int KeyFrame::GetNumberMPs()
{
    unique_lock<mutex> lock(mMutexFeatures);
    int numberMPs = 0;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        numberMPs++;
    }
    return numberMPs;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const int &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    tuple<size_t,size_t> indexes = pMP->GetIndexInKeyFrame(this);
    size_t leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    if(leftIndex != -1)
        mvpMapPoints[leftIndex]=static_cast<MapPoint*>(NULL);
    if(rightIndex != -1)
        mvpMapPoints[rightIndex]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const int &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections(bool upParent)
{
    FT_CALC_SCOPE();
    map<KeyFrame*,int> KFcounter;

    vector<MapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId || mit->first->isBad() || mit->first->GetMap() != mpMap)
                continue;
            KFcounter[mit->first]++;

        }
    }

    // This should not happen
    if(KFcounter.empty())
    {
        Map* pMap = GetMap();
        bool parentAssigned = false;
        if(pMap && mnId != pMap->GetInitKFid())
        {
            std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
            if(mbFirstConnection && mPrevKF)
            {
                parentAssigned = ChangeParentUnderTreeLock(this, mPrevKF, false);
                if(parentAssigned)
                    mbFirstConnection = false;
            }
        }
        if(pMap && mnId!=pMap->GetInitKFid() && !parentAssigned)
        {
            int nMP = 0; for(MapPoint* pMP : vpMP) if(pMP && !pMP->isBad()) nMP++;
            FT_CALC_LOG("research.kf.parentless reason=no_covisibility_no_valid_temporal_parent kf=%lu map=%lu valid_mps=%d t=%f",
                        mnId, pMap->GetId(), nMP, mTimeStamp);
        }
        return;
    }

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }
    FT_CALC_LOG("keyframe.connections.updated id=%llu candidates=%llu accepted=%llu parent_update=%d",
                static_cast<unsigned long long>(mnId), static_cast<unsigned long long>(KFcounter.size()),
                static_cast<unsigned long long>(vPairs.size()), upParent ? 1 : 0);

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    std::vector<KeyFrame*> parentCandidates;
    bool shouldAssignParent = false;
    {
        unique_lock<mutex> lockCon(mMutexConnections);

        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());


        parentCandidates = mvpOrderedConnectedKeyFrames;
    }

    Map* pMap = GetMap();
    if(pMap)
    {
        std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
        shouldAssignParent = mbFirstConnection && mnId != pMap->GetInitKFid();
        if(shouldAssignParent)
        {
            KeyFrame* pAssignedParent = NULL;
            bool usedTemporalFallback = false;
            for(KeyFrame* pCandidate : parentCandidates)
            {
                if(ChangeParentUnderTreeLock(this, pCandidate, false))
                {
                    pAssignedParent = pCandidate;
                    break;
                }
            }
            if(!pAssignedParent && mPrevKF && ChangeParentUnderTreeLock(this, mPrevKF, false))
            {
                pAssignedParent = mPrevKF;
                usedTemporalFallback = true;
            }

            if(pAssignedParent)
            {
                mbFirstConnection = false;
                FT_CALC_LOG("research.kf.parent_assigned id=%llu parent=%llu temporal_fallback=%d",
                            static_cast<unsigned long long>(mnId),
                            static_cast<unsigned long long>(pAssignedParent->mnId),
                            usedTemporalFallback ? 1 : 0);
            }
            else
            {
                FT_CALC_LOG("research.kf.parent_rejected id=%llu candidates=%llu reason=no_rooted_acyclic_parent",
                            static_cast<unsigned long long>(mnId),
                            static_cast<unsigned long long>(parentCandidates.size()));
            }
        }
    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    if(pKF)
        pKF->ChangeParent(this);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    if(!pKF)
        return;
    std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
    mspChildrens.erase(pKF);
    if(pKF->mpParent == this)
        pKF->mpParent = NULL;
}

bool KeyFrame::ChangeParent(KeyFrame *pKF)
{
    std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
    const bool ok = ChangeParentUnderTreeLock(this, pKF, false);
    FT_CALC_LOG("keyframe.parent_change id=%llu parent=%llu accepted=%d mode=normal",
                static_cast<unsigned long long>(mnId),
                static_cast<unsigned long long>(pKF ? pKF->mnId : 0UL), ok ? 1 : 0);
    return ok;
}

bool KeyFrame::ChangeParentForMerge(KeyFrame *pKF)
{
    std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
    const bool ok = ChangeParentUnderTreeLock(this, pKF, true);
    FT_CALC_LOG("keyframe.parent_change id=%llu parent=%llu accepted=%d mode=merge",
                static_cast<unsigned long long>(mnId),
                static_cast<unsigned long long>(pKF ? pKF->mnId : 0UL), ok ? 1 : 0);
    return ok;
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
    // mpParent remains as a historical reconstruction link after culling;
    // it is not an active spanning-tree edge for bad keyframes.
    return mbBad ? NULL : mpParent;
}

bool KeyFrame::ResolveTrajectoryReference(KeyFrame*& pRoot, Sophus::SE3f& TReferenceToRoot)
{
    std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
    std::set<KeyFrame*> visited;
    KeyFrame* pWalk = this;
    TReferenceToRoot = Sophus::SE3f();

    while(pWalk && pWalk->mbBad)
    {
        if(!visited.insert(pWalk).second)
            return false;

        Map* pWalkMap = pWalk->GetMap();
        KeyFrame* pParent = pWalk->mpParent;
        if(!pWalkMap || !pParent || pParent->GetMap() != pWalkMap)
            return false;

        TReferenceToRoot = TReferenceToRoot * pWalk->mTcp;
        pWalk = pParent;
    }

    // A reset detaches live keyframes from the cleared map. Do not combine an
    // old reference pose with a newly initialized map's origin. A live terminal
    // is valid only when it is the declared root used by TreeParentAllowed;
    // a disconnected non-root must not be serialized as an additional root.
    Map* pRootMap = pWalk ? pWalk->GetMap() : NULL;
    if(!pRootMap)
        return false;

    // A frame may legitimately reference any live keyframe on the map's
    // rooted tree, not only the declared root. Validate that active ancestry
    // reaches the declared root before exposing the live terminal. This still
    // rejects disconnected parentless components and malformed merge/reset
    // links without dropping ordinary non-root trajectory references.
    std::set<KeyFrame*> activeAncestors;
    KeyFrame* pActive = pWalk;
    while(pActive)
    {
        if(!activeAncestors.insert(pActive).second || pActive->mbBad || pActive->GetMap() != pRootMap)
            return false;
        if(pActive->mnId == pRootMap->GetInitKFid())
        {
            if(pActive->mpParent)
                return false;
            pRoot = pWalk;
            return true;
        }

        KeyFrame* pParent = pActive->mpParent;
        if(!pParent || pParent->mbBad || pParent->GetMap() != pRootMap ||
           pParent->mspChildrens.count(pActive) == 0)
            return false;
        pActive = pParent;
    }
    return false;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
    return mspChildrens.count(pKF);
}

void KeyFrame::SetFirstConnection(bool bFirst)
{
    std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
    mbFirstConnection=bFirst;
}

std::mutex& KeyFrame::SpanningTreeMutex()
{
    static std::mutex mutex;
    return mutex;
}

void KeyFrame::DetachTreeRelations()
{
    std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
    if(mpParent)
        mpParent->mspChildrens.erase(this);
    mpParent = NULL;
    for(KeyFrame* pChild : mspChildrens)
    {
        if(pChild && pChild->mpParent == this)
            pChild->mpParent = NULL;
    }
    mspChildrens.clear();
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::AddMergeEdge(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspMergeEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetMergeEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspMergeEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

bool KeyFrame::SetBadFlag(KeyFrame* pFallbackParent, bool bMergeInertialLinks)
{
    return SetBadFlagImpl(pFallbackParent, bMergeInertialLinks, NULL);
}

bool KeyFrame::SetBadFlagMapUpdateHeld(Map* pHeldMap)
{
    return SetBadFlagImpl(NULL, false, pHeldMap);
}

bool KeyFrame::SetBadFlagImpl(KeyFrame* pFallbackParent, bool bMergeInertialLinks, Map* pHeldMap)
{
    FT_CALC_SCOPE();
    Map* pMap = NULL;
    std::vector<KeyFrame*> vConnections;
    std::vector<MapPoint*> vMapPoints;

    pMap = GetMap();
    if(!pMap)
        return false;
    // Match the map-update -> spanning-tree lock order used by optimization,
    // merge membership migration, map clear and save snapshots.
    std::unique_lock<std::mutex> mapUpdateLock(pMap->mMutexMapUpdate, std::defer_lock);
    if(pMap != pHeldMap)
        mapUpdateLock.lock();
    // One tree lock covers validation, child reparenting and reciprocal edge
    // updates. Readers use the same lock through GetParent/GetChilds/hasChild.
    std::unique_lock<std::mutex> treeLock(SpanningTreeMutex());
    if(GetMap() != pMap)
        return false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mbBad || mnId == pMap->GetInitKFid())
            return false;
        if(mbNotErase)
        {
            mbToBeErased = true;
            return false;
        }
        for(map<KeyFrame*,int>::const_iterator it=mConnectedKeyFrameWeights.begin(); it!=mConnectedKeyFrameWeights.end(); ++it)
            vConnections.push_back(it->first);
    }
    {
        unique_lock<mutex> lock(mMutexFeatures);
        vMapPoints = mvpMapPoints;
    }

    if(bMergeInertialLinks && (!mPrevKF || !mNextKF || !mpImuPreintegrated || !mNextKF->mpImuPreintegrated))
        return false;

    KeyFrame* pReplacementParent = mpParent ? mpParent : pFallbackParent;
    const bool usedParentlessFallback = !mpParent && pFallbackParent && !mspChildrens.empty();
    if(!mspChildrens.empty())
    {
        if(!pReplacementParent || !TreeParentAllowed(this, pReplacementParent, false))
        {
            FT_CALC_LOG("research.cull.refused kf=%llu children=%llu reason=no_same_map_rooted_parent",
                        static_cast<unsigned long long>(mnId),
                        static_cast<unsigned long long>(mspChildrens.size()));
            return false;
        }
        for(KeyFrame* pChild : mspChildrens)
        {
            if(pChild && !pChild->isBad() && !TreeParentAllowed(pChild, pReplacementParent, false))
            {
                FT_CALC_LOG("research.cull.refused kf=%llu child=%llu reason=unsafe_reparent",
                            static_cast<unsigned long long>(mnId),
                            static_cast<unsigned long long>(pChild->mnId));
                return false;
            }
        }
    }

    // The existing parent (or the caller's temporal predecessor when the
    // target was parentless) is the explicit structural fallback. We never
    // create an additional map root to hide a disconnected component.
    if(!mspChildrens.empty())
    {
        if(usedParentlessFallback)
            FT_CALC_LOG("research.cull.parentless_fallback.begin kf=%llu parent=%llu children=%llu",
                        static_cast<unsigned long long>(mnId),
                        static_cast<unsigned long long>(pReplacementParent->mnId),
                        static_cast<unsigned long long>(mspChildrens.size()));
        std::vector<KeyFrame*> vChildren(mspChildrens.begin(), mspChildrens.end());
        for(KeyFrame* pChild : vChildren)
        {
            if(pChild && !pChild->isBad() && !ChangeParentUnderTreeLock(pChild, pReplacementParent, false))
                return false; // Prevalidation makes this unreachable without concurrent lifetime corruption.
        }
        for(KeyFrame* pChild : vChildren)
            if(pChild && pChild->mpParent == this)
                pChild->mpParent = NULL;
    }

    if(bMergeInertialLinks)
    {
        KeyFrame* pPrevKF = mPrevKF;
        KeyFrame* pNextKF = mNextKF;
        pNextKF->mpImuPreintegrated->MergePrevious(mpImuPreintegrated);
        pNextKF->mPrevKF = pPrevKF;
        pPrevKF->mNextKF = pNextKF;
        mNextKF = NULL;
        mPrevKF = NULL;
    }

    if(pReplacementParent)
    {
        if(mpParent)
            mpParent->mspChildrens.erase(this);
        // Trajectory reconstruction follows mTcp for culled keyframes. If the
        // original keyframe was parentless, preserve the same relative-pose
        // invariant using the validated explicit fallback parent.
        mTcp = mTcw * pReplacementParent->GetPoseInverse();
        // This remains available only to trajectory reconstruction. The bad
        // keyframe is removed from the active parent's reciprocal child set.
        mpParent = pReplacementParent;
    }
    mspChildrens.clear();
    {
        unique_lock<mutex> lock(mMutexConnections);
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();
        mbBad = true;
    }
    treeLock.unlock();

    for(KeyFrame* pConnection : vConnections)
        if(pConnection)
            pConnection->EraseConnection(this);
    for(MapPoint* pMP : vMapPoints)
        if(pMP)
            pMP->EraseObservation(this);
    pMap->EraseKeyFrame(this);
    if(mpKeyFrameDB)
        mpKeyFrameDB->erase(this);
    if(usedParentlessFallback)
        FT_CALC_LOG("research.cull.parentless_fallback.complete kf=%llu parent=%llu",
                    static_cast<unsigned long long>(mnId),
                    static_cast<unsigned long long>(pReplacementParent->mnId));
    FT_CALC_LOG("keyframe.cull.complete id=%llu map=%llu", static_cast<unsigned long long>(mnId),
                static_cast<unsigned long long>(pMap->GetId()));
    return true;
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}


vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (NLeft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

bool KeyFrame::UnprojectStereo(int i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        x3D = mRwc * x3Dc + mTwc.translation();
        return true;
    }
    else
        return false;
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    if(N==0)
        return -1.0;

    vector<MapPoint*> vpMapPoints;
    Eigen::Matrix3f Rcw;
    Eigen::Vector3f tcw;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        tcw = mTcw.translation();
        Rcw = mRcw;
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    Eigen::Matrix<float,1,3> Rcw2 = Rcw.row(2);
    float zcw = tcw(2);
    for(int i=0; i<N; i++) {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            Eigen::Vector3f x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw) + zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

void KeyFrame::SetNewBias(const IMU::Bias &b)
{
    unique_lock<mutex> lock(mMutexPose);
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

Eigen::Vector3f KeyFrame::GetGyroBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}

Eigen::Vector3f KeyFrame::GetAccBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bax, mImuBias.bay, mImuBias.baz);
}

IMU::Bias KeyFrame::GetImuBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return mImuBias;
}

Map* KeyFrame::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void KeyFrame::UpdateMap(Map* pMap)
{
    std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void KeyFrame::PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP, set<GeometricCamera*>& spCam)
{
    // Save the id of each MapPoint in this KF, there can be null pointer in the vector
    mvBackupMapPointsId.clear();
    mvBackupMapPointsId.reserve(N);
    for(int i = 0; i < N; ++i)
    {

        if(mvpMapPoints[i] && spMP.find(mvpMapPoints[i]) != spMP.end()) // Checks if the element is not null
            mvBackupMapPointsId.push_back(mvpMapPoints[i]->mnId);
        else // If the element is null his value is -1 because all the id are positives
            mvBackupMapPointsId.push_back(-1);
    }
    // Save the id of each connected KF with it weight
    mBackupConnectedKeyFrameIdWeights.clear();
    for(std::map<KeyFrame*,int>::const_iterator it = mConnectedKeyFrameWeights.begin(), end = mConnectedKeyFrameWeights.end(); it != end; ++it)
    {
        if(spKF.find(it->first) != spKF.end())
            mBackupConnectedKeyFrameIdWeights[it->first->mnId] = it->second;
    }

    // Save the parent id
    mBackupParentId = -1;
    if(mpParent && spKF.find(mpParent) != spKF.end())
        mBackupParentId = mpParent->mnId;

    // Save the id of the childrens KF
    mvBackupChildrensId.clear();
    mvBackupChildrensId.reserve(mspChildrens.size());
    for(KeyFrame* pKFi : mspChildrens)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupChildrensId.push_back(pKFi->mnId);
    }

    // Save the id of the loop edge KF
    mvBackupLoopEdgesId.clear();
    mvBackupLoopEdgesId.reserve(mspLoopEdges.size());
    for(KeyFrame* pKFi : mspLoopEdges)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupLoopEdgesId.push_back(pKFi->mnId);
    }

    // Save the id of the merge edge KF
    mvBackupMergeEdgesId.clear();
    mvBackupMergeEdgesId.reserve(mspMergeEdges.size());
    for(KeyFrame* pKFi : mspMergeEdges)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupMergeEdgesId.push_back(pKFi->mnId);
    }

    //Camera data
    mnBackupIdCamera = -1;
    if(mpCamera && spCam.find(mpCamera) != spCam.end())
        mnBackupIdCamera = mpCamera->GetId();

    mnBackupIdCamera2 = -1;
    if(mpCamera2 && spCam.find(mpCamera2) != spCam.end())
        mnBackupIdCamera2 = mpCamera2->GetId();

    //Inertial data
    mBackupPrevKFId = -1;
    if(mPrevKF && spKF.find(mPrevKF) != spKF.end())
        mBackupPrevKFId = mPrevKF->mnId;

    mBackupNextKFId = -1;
    if(mNextKF && spKF.find(mNextKF) != spKF.end())
        mBackupNextKFId = mNextKF->mnId;

    if(mpImuPreintegrated)
        mBackupImuPreintegrated.CopyFrom(mpImuPreintegrated);
}

void KeyFrame::PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid, map<unsigned int, GeometricCamera*>& mpCamId){
    FT_CALC_SCOPE();
    // Rebuild the empty variables

    // Pose
    SetPose(mTcw);

    mTrl = mTlr.inverse();

    // Reference reconstruction
    // Each MapPoint sight from this KeyFrame
    mvpMapPoints.clear();
    mvpMapPoints.resize(N);
    for(int i=0; i<N; ++i)
    {
        if(mvBackupMapPointsId[i] != -1)
            mvpMapPoints[i] = mpMPid[mvBackupMapPointsId[i]];
        else
            mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
    }

    // Conected KeyFrames with him weight
    mConnectedKeyFrameWeights.clear();
    for(map<long unsigned int, int>::const_iterator it = mBackupConnectedKeyFrameIdWeights.begin(), end = mBackupConnectedKeyFrameIdWeights.end();
        it != end; ++it)
    {
        KeyFrame* pKFi = mpKFid[it->first];
        mConnectedKeyFrameWeights[pKFi] = it->second;
    }

    {
        std::lock_guard<std::mutex> treeLock(SpanningTreeMutex());
        // Restore parent KeyFrame
        mpParent = (mBackupParentId>=0) ? mpKFid[mBackupParentId] : NULL;

        // KeyFrame childrens
        mspChildrens.clear();
        for(vector<long unsigned int>::const_iterator it = mvBackupChildrensId.begin(), end = mvBackupChildrensId.end(); it!=end; ++it)
        {
            mspChildrens.insert(mpKFid[*it]);
        }
    }

    // Loop edge KeyFrame
    mspLoopEdges.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupLoopEdgesId.begin(), end = mvBackupLoopEdgesId.end(); it != end; ++it)
    {
        mspLoopEdges.insert(mpKFid[*it]);
    }

    // Merge edge KeyFrame
    mspMergeEdges.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupMergeEdgesId.begin(), end = mvBackupMergeEdgesId.end(); it != end; ++it)
    {
        mspMergeEdges.insert(mpKFid[*it]);
    }

    //Camera data
    if(mnBackupIdCamera >= 0)
    {
        mpCamera = mpCamId[mnBackupIdCamera];
    }
    else
    {
        FT_CALC_NOTICE("keyframe.load.missing_camera id=%llu",
                       static_cast<unsigned long long>(mnId));
    }
    if(mnBackupIdCamera2 >= 0)
    {
        mpCamera2 = mpCamId[mnBackupIdCamera2];
    }

    //Inertial data
    if(mBackupPrevKFId != -1)
    {
        mPrevKF = mpKFid[mBackupPrevKFId];
    }
    if(mBackupNextKFId != -1)
    {
        mNextKF = mpKFid[mBackupNextKFId];
    }
    mpImuPreintegrated = &mBackupImuPreintegrated;


    // Remove all backup container
    mvBackupMapPointsId.clear();
    mBackupConnectedKeyFrameIdWeights.clear();
    mvBackupChildrensId.clear();
    mvBackupLoopEdgesId.clear();

    UpdateBestCovisibles();
}

bool KeyFrame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{
    FT_CALC_SCOPE();

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mRcw * P + mTcw.translation();
    float &PcX = Pc(0);
    float &PcY = Pc(1);
    float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        FT_CALC_LOG("keyframe.project_distort.reject id=%llu reason=negative_depth z=%f",
                    static_cast<unsigned long long>(mnId), PcZ);
        return false;
    }

    // Project in image and check it is not outside
    float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    // cout << "c";

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    float u_distort = x_distort * fx + cx;
    float v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

bool KeyFrame::ProjectPointUnDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{
    FT_CALC_SCOPE();

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mRcw * P + mTcw.translation();
    float &PcX = Pc(0);
    float &PcY= Pc(1);
    float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        FT_CALC_LOG("keyframe.project_undistort.reject id=%llu reason=negative_depth z=%f",
                    static_cast<unsigned long long>(mnId), PcZ);
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u = fx * PcX * invz + cx;
    v = fy * PcY * invz + cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    kp = cv::Point2f(u, v);

    return true;
}

Sophus::SE3f KeyFrame::GetRelativePoseTrl()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTrl;
}

Sophus::SE3f KeyFrame::GetRelativePoseTlr()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTlr;
}

Sophus::SE3<float> KeyFrame::GetRightPose() {
    unique_lock<mutex> lock(mMutexPose);

    return mTrl * mTcw;
}

Sophus::SE3<float> KeyFrame::GetRightPoseInverse() {
    unique_lock<mutex> lock(mMutexPose);

    return mTwc * mTlr;
}

Eigen::Vector3f KeyFrame::GetRightCameraCenter() {
    unique_lock<mutex> lock(mMutexPose);

    return (mTwc * mTlr).translation();
}

Eigen::Matrix<float,3,3> KeyFrame::GetRightRotation() {
    unique_lock<mutex> lock(mMutexPose);

    return (mTrl.so3() * mTcw.so3()).matrix();
}

Eigen::Vector3f KeyFrame::GetRightTranslation() {
    unique_lock<mutex> lock(mMutexPose);
    return (mTrl * mTcw).translation();
}

void KeyFrame::SetORBVocabulary(ORBVocabulary* pORBVoc)
{
    mpORBvocabulary = pORBVoc;
}

void KeyFrame::SetKeyFrameDatabase(KeyFrameDatabase* pKFDB)
{
    mpKeyFrameDB = pKFDB;
}

} //namespace ORB_SLAM
