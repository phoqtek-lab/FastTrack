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



#include "System.h"
#include "Converter.h"
#include "Kernels/StereoMatchKernel.h"
#include <csignal>
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <openssl/md5.h>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <csignal>
#include <iostream>
#include <cstdlib> // For std::exit
#include "Kernels/KernelController.h"
#include "CalculationLogger.h"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>

namespace ORB_SLAM3
{

Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NORMAL;

// void signalHandler(int signum) {
//     std::cout << "Interrupt signal (" << signum << ") received.\n";

//     // Release resources here
//     KernelController::shutdownKernels();
    
//     // Exit the program
//     std::exit(signum);
// }

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer, const int initFr, const string &strSequence):
    System(strVocFile, strSettingsFile, sensor, PriorMapOptions(), bUseViewer, initFr, strSequence)
{
}

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const PriorMapOptions &prior, const bool bUseViewer, const int initFr, const string &strSequence):
    mSensor(sensor), mpVocabulary(nullptr), mpKeyFrameDatabase(nullptr), mpAtlas(nullptr),
    mpTracker(nullptr), mpLocalMapper(nullptr), mpLoopCloser(nullptr), mpViewer(nullptr),
    mpFrameDrawer(nullptr), mpMapDrawer(nullptr), mptLocalMapping(nullptr),
    mptLoopClosing(nullptr), mptViewer(nullptr), mbReset(false), mbResetActiveMap(false),
    mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false), mbShutDown(false),
    mTrackingState(Tracking::SYSTEM_NOT_READY), settings_(nullptr)
{
    CalculationLog::initialize();
    FT_CALC_SCOPE();
    mPriorOpt = prior;
    mbResetToPriorMap = false;

    // std::signal(SIGINT, signalHandler);

    // Output welcome message
    FT_CALC_LOG("system.license orbslam3");
    /*
    "ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
    FT_CALC_LOG("system.license warranty=none");
    */

    FT_CALC_LOG("system.sensor.selected value=%d", static_cast<int>(mSensor));

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       FT_CALC_LOG("system.abort reason=settings_open_failed path=%s", strSettingsFile.c_str());
       exit(-1);
    }

    cv::FileNode node = fsSettings["File.version"];
    if(!node.empty() && node.isString() && node.string() == "1.0"){
        settings_ = new Settings(strSettingsFile,mSensor);

        mStrLoadAtlasFromFile = settings_->atlasLoadFile();
        mStrSaveAtlasToFile = settings_->atlasSaveFile();

        FT_CALC_LOG("system.settings.loaded version=1.0 atlas_load=%s atlas_save=%s",
                    mStrLoadAtlasFromFile.c_str(), mStrSaveAtlasToFile.c_str());
    }
    else{
        settings_ = nullptr;
        cv::FileNode node = fsSettings["System.LoadAtlasFromFile"];
        if(!node.empty() && node.isString())
        {
            mStrLoadAtlasFromFile = (string)node;
        }

        node = fsSettings["System.SaveAtlasToFile"];
        if(!node.empty() && node.isString())
        {
            mStrSaveAtlasToFile = (string)node;
        }
    }

    node = fsSettings["loopClosing"];
    bool activeLC = true;
    if(!node.empty())
    {
        activeLC = static_cast<int>(fsSettings["loopClosing"]) != 0;
    }

    mStrVocabularyFilePath = strVocFile;

    bool loadedAtlas = false;

    // Prior-map start: the options name the file; the legacy settings key is ignored then.
    const bool bPriorLoad = mPriorOpt.enable && !mPriorOpt.load_path.empty();
    PriorMapStatus priorLoad;
    bool bPriorLoaded = false;
    if(bPriorLoad && !mStrLoadAtlasFromFile.empty())
    {
        FT_CALC_LOG("system.prior_map.ignore_settings_load path=%s", mStrLoadAtlasFromFile.c_str());
        mStrLoadAtlasFromFile.clear();
    }

    if(mStrLoadAtlasFromFile.empty())
    {
        //Load ORB Vocabulary
        FT_CALC_LOG("system.vocabulary.load path=%s", strVocFile.c_str());

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if(!bVocLoad)
        {
            FT_CALC_LOG("system.abort reason=vocabulary_load_failed path=%s", strVocFile.c_str());
            exit(-1);
        }
        FT_CALC_LOG("system.vocabulary.loaded");

        //Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        //Create the Atlas
        FT_CALC_LOG("system.atlas.initialize mode=new");
        mpAtlas = new Atlas(0);

        if(bPriorLoad)
        {
            const auto tLoad0 = std::chrono::steady_clock::now();
            bPriorLoaded = LoadAtlasFile(mPriorOpt.load_path, priorLoad);
            const double loadS = std::chrono::duration<double>(std::chrono::steady_clock::now() - tLoad0).count();
            FT_CALC_LOG("system.prior_map.load ok=%d seconds=%f reason=%s", bPriorLoaded ? 1 : 0, loadS, priorLoad.reason.c_str());
            if(!bPriorLoaded)
            {
                // Normal start. The partially filled database (if any) is replaced.
                mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
                mpAtlas = new Atlas(0);
            }
        }
    }
    else
    {
        //Load ORB Vocabulary
        FT_CALC_LOG("system.vocabulary.load path=%s", strVocFile.c_str());

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if(!bVocLoad)
        {
            FT_CALC_LOG("system.abort reason=vocabulary_load_failed path=%s", strVocFile.c_str());
            exit(-1);
        }
        FT_CALC_LOG("system.vocabulary.loaded");

        //Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        FT_CALC_LOG("system.atlas.load path=%s", mStrLoadAtlasFromFile.c_str());

        // Load the file with an earlier session
        //clock_t start = clock();
        bool isRead = LoadAtlas(FileType::BINARY_FILE);

        if(!isRead)
        {
            FT_CALC_LOG("system.abort reason=atlas_load_failed path=%s", mStrLoadAtlasFromFile.c_str());
            exit(-1);
        }
        //mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);


        //cout << "KF in DB: " << mpKeyFrameDatabase->mnNumKFs << "; words: " << mpKeyFrameDatabase->mnNumWords << endl;

        loadedAtlas = true;

        mpAtlas->CreateNewMap();

        //clock_t timeElapsed = clock() - start;
        //unsigned msElapsed = timeElapsed / (CLOCKS_PER_SEC / 1000);
        //cout << "Binary file read in " << msElapsed << " ms" << endl;

        //usleep(10*1000*1000);
    }


    if (mSensor==IMU_STEREO || mSensor==IMU_MONOCULAR || mSensor==IMU_RGBD)
        mpAtlas->SetInertialSensor();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpAtlas);
    mpMapDrawer = new MapDrawer(mpAtlas, strSettingsFile, settings_);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    FT_CALC_LOG("system.sequence name=%s", strSequence.c_str());
    
    Tracking::sbInstallSigintHandler = !mPriorOpt.keep_host_sigint;
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpAtlas, mpKeyFrameDatabase, strSettingsFile, mSensor, settings_, strSequence);
    // Before the mapping threads exist: choose / validate the prior map against this session's calibration.
    mpTracker->ConfigurePriorMap(mPriorOpt, priorLoad, bPriorLoaded);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(this, mpAtlas, mSensor==MONOCULAR || mSensor==IMU_MONOCULAR,
                                     mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD, strSequence);
    mptLocalMapping = new thread(&ORB_SLAM3::LocalMapping::Run,mpLocalMapper);
    mpLocalMapper->mInitFr = initFr;
    if(settings_)
        mpLocalMapper->mThFarPoints = settings_->thFarPoints();
    else
        mpLocalMapper->mThFarPoints = fsSettings["thFarPoints"];
    if(mpLocalMapper->mThFarPoints!=0)
    {
        FT_CALC_LOG("system.local_mapping.far_point_threshold=%f", mpLocalMapper->mThFarPoints);
        mpLocalMapper->mbFarPoints = true;
    }
    else
        mpLocalMapper->mbFarPoints = false;

    //Initialize the Loop Closing thread and launch
    // mSensor!=MONOCULAR && mSensor!=IMU_MONOCULAR
    mpLoopCloser = new LoopClosing(mpAtlas, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR, activeLC); // mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM3::LoopClosing::Run, mpLoopCloser);

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    //usleep(10*1000*1000);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    //if(false) // TODO
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile,settings_);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
        mpLoopCloser->mpViewer = mpViewer;
        mpViewer->both = mpFrameDrawer->both;
    }

    // Fix verbosity
    Verbose::SetTh(Verbose::VERBOSITY_QUIET);

}


Sophus::SE3f System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.track.stereo.begin timestamp=%f left=%dx%d right=%dx%d imu_samples=%llu",
                timestamp, imLeft.cols, imLeft.rows, imRight.cols, imRight.rows,
                static_cast<unsigned long long>(vImuMeas.size()));
    if(mSensor!=STEREO && mSensor!=IMU_STEREO)
    {
        FT_CALC_LOG("system.track.stereo.abort reason=sensor_mismatch sensor=%d", static_cast<int>(mSensor));
        exit(-1);
    }

    cv::Mat imLeftToFeed, imRightToFeed;
    if(settings_ && settings_->needToRectify()){
        cv::Mat M1l = settings_->M1l();
        cv::Mat M2l = settings_->M2l();
        cv::Mat M1r = settings_->M1r();
        cv::Mat M2r = settings_->M2r();

        cv::remap(imLeft, imLeftToFeed, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(imRight, imRightToFeed, M1r, M2r, cv::INTER_LINEAR);
    }
    else if(settings_ && settings_->needToResize()){
        cv::resize(imLeft,imLeftToFeed,settings_->newImSize());
        cv::resize(imRight,imRightToFeed,settings_->newImSize());
    }
    else{
        imLeftToFeed = imLeft.clone();
        imRightToFeed = imRight.clone();
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    if (mSensor == System::IMU_STEREO)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    Sophus::SE3f Tcw = mpTracker->GrabImageStereo(imLeftToFeed,imRightToFeed,timestamp,filename);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    FT_CALC_LOG("system.track.stereo.end state=%d tracked_points=%llu keypoints=%llu",
                mTrackingState, static_cast<unsigned long long>(mTrackedMapPoints.size()),
                static_cast<unsigned long long>(mTrackedKeyPointsUn.size()));

    return Tcw;
}

Sophus::SE3f System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.track.rgbd.begin timestamp=%f image=%dx%d depth=%dx%d imu_samples=%llu",
                timestamp, im.cols, im.rows, depthmap.cols, depthmap.rows,
                static_cast<unsigned long long>(vImuMeas.size()));
    if(mSensor!=RGBD  && mSensor!=IMU_RGBD)
    {
        FT_CALC_LOG("system.track.rgbd.abort reason=sensor_mismatch sensor=%d", static_cast<int>(mSensor));
        exit(-1);
    }

    cv::Mat imToFeed = im.clone();
    cv::Mat imDepthToFeed = depthmap.clone();
    if(settings_ && settings_->needToResize()){
        cv::Mat resizedIm;
        cv::resize(im,resizedIm,settings_->newImSize());
        imToFeed = resizedIm;

        cv::resize(depthmap,imDepthToFeed,settings_->newImSize());
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    if (mSensor == System::IMU_RGBD)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    Sophus::SE3f Tcw = mpTracker->GrabImageRGBD(imToFeed,imDepthToFeed,timestamp,filename);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    FT_CALC_LOG("system.track.rgbd.end state=%d tracked_points=%llu keypoints=%llu",
                mTrackingState, static_cast<unsigned long long>(mTrackedMapPoints.size()),
                static_cast<unsigned long long>(mTrackedKeyPointsUn.size()));
    return Tcw;
}

Sophus::SE3f System::TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.track.monocular.begin timestamp=%f image=%dx%d imu_samples=%llu",
                timestamp, im.cols, im.rows, static_cast<unsigned long long>(vImuMeas.size()));

    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbShutDown)
            return Sophus::SE3f();
    }

    if(mSensor!=MONOCULAR && mSensor!=IMU_MONOCULAR)
    {
        FT_CALC_LOG("system.track.monocular.abort reason=sensor_mismatch sensor=%d", static_cast<int>(mSensor));
        exit(-1);
    }

    cv::Mat imToFeed = im.clone();
    if(settings_ && settings_->needToResize()){
        cv::Mat resizedIm;
        cv::resize(im,resizedIm,settings_->newImSize());
        imToFeed = resizedIm;
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            FT_CALC_LOG("system.track.monocular.reset_active_map");
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
        if(mbResetToPriorMap)
        {
            mbResetToPriorMap = false;
            std::string why;
            if(mpTracker->ResetToPriorMap(why))
                FT_CALC_LOG("system.track.monocular.reset_to_prior_map ok=1 detail=%s", why.c_str());
            else
            {
                // A ground reset must complete: without a usable prior map (or if keeping it fails) reset everything.
                FT_CALC_LOG("system.track.monocular.reset_to_prior_map ok=0 reason=%s action=full_reset", why.c_str());
                mpTracker->Reset();
            }
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    Sophus::SE3f Tcw = mpTracker->GrabImageMonocular(imToFeed,timestamp,filename);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    FT_CALC_LOG("system.track.monocular.end state=%d tracked_points=%llu keypoints=%llu",
                mTrackingState, static_cast<unsigned long long>(mTrackedMapPoints.size()),
                static_cast<unsigned long long>(mTrackedKeyPointsUn.size()));

    return Tcw;
}



void System::SetExternalReference(const bool bHasYaw, const double yawDeg, const bool bHasAlt, const double altM)
{
    mpTracker->SetExternalReference(bHasYaw, yawDeg, bHasAlt, altM);
}

void System::AddAltitudeReference(double t, double h)
{
    if(mpTracker)
        mpTracker->AddAltitudeReference(t, h);
}

void System::AddGravityReference(double t, const Eigen::Matrix3f &Rwb)
{
    if(mpLocalMapper)
        mpLocalMapper->AddGravityReference(t, Rwb);
}

void System::SetGravityFromReference(bool b)
{
    if(mpLocalMapper)
        mpLocalMapper->SetGravityFromReference(b);
}

void System::SetRestGyroBiasPrior(const Eigen::Vector3f &bg, const float sigma)
{
    mpLocalMapper->SetRestGyroBiasPrior(bg, sigma);
}

void System::ClearRestGyroBiasPrior()
{
    mpLocalMapper->ClearRestGyroBiasPrior();
}

void System::SetAccBiasPrior(const Eigen::Vector3f &ba, const float sigma)
{
    mpLocalMapper->SetAccBiasPrior(ba, sigma);
}

void System::ClearAccBiasPrior()
{
    mpLocalMapper->ClearAccBiasPrior();
}

void System::SetPoseOptimizationMode(int mode)
{
    mpTracker->SetPoseOptimizationMode(mode);
}

void System::SetPoseOptimizationTurn(double rate_dps, double hold_s)
{
    mpTracker->SetPoseOptimizationTurn(rate_dps, hold_s);
}

bool System::GetBaroScaleObservation(double &t, double &h, double &d, double &s, double &z, double &tilt, int &nI,
                                     int &nP, unsigned long &mapId, unsigned long &seq)
{
    LocalMapping::BaroScaleObs o;
    if (!mpLocalMapper->GetBaroScaleObs(o))
        return false;
    t = o.t; h = o.h; d = o.d; s = o.s; z = o.z; tilt = o.tilt; nI = o.nI; nP = o.nP; mapId = o.mapId; seq = o.seq;
    return true;
}

bool System::GetInitProfileState(Eigen::Vector3f &ba, Eigen::Vector3f &bg, int &nKF)
{
    Map* pMap = mpAtlas->GetCurrentMap();
    if (!pMap || !pMap->isImuInitialized() || !pMap->GetIniertialBA2())
        return false;
    KeyFrame* pKF = mpTracker->GetLastKeyFrame();
    if (!pKF || pKF->isBad() || pKF->GetMap() != pMap)
        return false;
    ba = pKF->GetAccBias();
    bg = pKF->GetGyroBias();
    nKF = static_cast<int>(pMap->KeyFramesInMap());
    return ba.allFinite() && bg.allFinite();
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpAtlas->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::ResetActiveMap()
{
    unique_lock<mutex> lock(mMutexReset);
    mbResetActiveMap = true;
}

void System::ResetToPriorMap()
{
    unique_lock<mutex> lock(mMutexReset);
    mbResetToPriorMap = true;
}

void System::AllowTimeGapOnce(double gapS)
{
    if(mpTracker)
        mpTracker->AllowTimeGapOnce(gapS);
}

bool System::HasUsablePriorMap()
{
    return mpTracker && mpTracker->HasUsablePriorMap();
}

PriorMapStatus System::GetPriorMapStatus()
{
    if(!mpTracker)
        return PriorMapStatus();
    return mpTracker->GetPriorMapStatus();
}

SaveMapResult System::SaveMap(const std::string &pathIn)
{
    FT_CALC_SCOPE();
    SaveMapResult r;
    r.path = pathIn.empty() ? mPriorOpt.save_path : pathIn;
    if(r.path.empty())
        r.reason = "no save path (prior_map save_path empty)";
    else if(isShutDown())
        r.reason = "system shut down";
    else
    {
        Map* pActive = mpAtlas->GetCurrentMap();
        if(!pActive || !pActive->isImuInitialized() || !pActive->GetIniertialBA2())
            r.reason = "active map is not inertially initialized (IMU init + VIBA2)";
        else if(static_cast<int>(pActive->KeyFramesInMap()) < mPriorOpt.min_keyframes)
            r.reason = "active map has " + std::to_string(pActive->KeyFramesInMap()) + " keyframes, minimum " +
                       std::to_string(mPriorOpt.min_keyframes);
    }
    if(!r.reason.empty())
    {
        FT_CALC_LOG("system.save_map.refused reason=%s", r.reason.c_str());
        mpTracker->NotePriorSave(r);
        return r;
    }
    std::string why;
    if(!mpTracker->StopMappingThreads(false, mPriorOpt.gba_wait_s, why))
    {
        r.reason = why;
        FT_CALC_LOG("system.save_map.refused reason=%s", r.reason.c_str());
        mpTracker->NotePriorSave(r);
        return r;
    }
    const std::string path = r.path;
    r = WriteAtlasFile(path);
    mpTracker->ReleaseMappingThreads();
    if(r.ok)
        mpTracker->AdoptActiveMapAsPrior(r);
    mpTracker->NotePriorSave(r);
    return r;
}

void System::Shutdown()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbShutDown)
            return;
        mbShutDown = true;
    }

    FT_CALC_NOTICE("system.shutdown.requested");

    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();

    if(mpViewer)
    {
        mpViewer->RequestFinish();
    }

    // NanoLog finalization is deliberately delayed until every producer has
    // stopped. Joining also makes the previous best-effort shutdown ordering
    // explicit without adding work to the frame path.
    if(mptLocalMapping != nullptr && mptLocalMapping->joinable())
        mptLocalMapping->join();
    if(mptLoopClosing != nullptr && mptLoopClosing->joinable())
        mptLoopClosing->join();
    if(mptViewer != nullptr && mptViewer->joinable())
        mptViewer->join();

    // LoopClosing may own a detached GBA worker.  Wait for its terminal state
    // before finalizing NanoLog so no producer can enter the runtime during
    // finalization.
    while(mpLoopCloser != nullptr && mpLoopCloser->isRunningGBA())
        usleep(1000);
    FT_CALC_LOG("system.shutdown.workers_stopped");

    KernelController::shutdownKernels();

    if(mPriorOpt.save_on_shutdown && !mPriorOpt.save_path.empty())
    {
        const SaveMapResult r = WriteAtlasFile(mPriorOpt.save_path);
        FT_CALC_LOG("system.shutdown.save_map ok=%d path=%s reason=%s", r.ok ? 1 : 0, r.path.c_str(), r.reason.c_str());
        std::fprintf(stderr, "[prior_map] shutdown save %s: %s (%llu bytes) %s\n", r.ok ? "ok" : "FAILED", r.path.c_str(),
                     static_cast<unsigned long long>(r.bytes), r.reason.c_str());
    }

    if(!mStrSaveAtlasToFile.empty())
    {
        Verbose::PrintMess("Atlas saving to file " + mStrSaveAtlasToFile, Verbose::VERBOSITY_NORMAL);
        SaveAtlas(FileType::BINARY_FILE);
    }

    /*if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");*/

#ifdef REGISTER_TIMES
    mpTracker->PrintTimeStats();
#endif

    CalculationLog::shutdown();

}

bool System::isShutDown() {
    unique_lock<mutex> lock(mMutexReset);
    return mbShutDown;
}

void System::SaveTrajectoryTUM(const string &filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.trajectory.save format=tum kind=camera path=%s", filename.c_str());
    if(mSensor==MONOCULAR)
    {
        FT_CALC_LOG("system.trajectory.abort format=tum reason=monocular_sensor");
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
    if(vpKFs.empty())
    {
        FT_CALC_LOG("system.trajectory.abort format=tum reason=no_keyframes");
        return;
    }
    Map* pExportMap = vpKFs[0]->GetMap();
    if(!pExportMap)
    {
        FT_CALC_LOG("system.trajectory.abort format=tum reason=detached_export_map");
        return;
    }

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;
        Sophus::SE3f Trw;

        // Culling retains a separate historical path. A reset-invalidated or
        // malformed path must not be dereferenced or stitched to a new map.
        if(!pKF || !pKF->ResolveTrajectoryReference(pKF, Trw))
            continue;
        if(pKF->GetMap() != pExportMap)
            continue;

        Trw = Trw * pKF->GetPose() * Two;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();

        Eigen::Vector3f twc = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    f.close();
    // cout << endl << "trajectory saved!" << endl;
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.trajectory.save format=tum kind=keyframe path=%s", filename.c_str());

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        Eigen::Vector3f t = Twc.translation();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t(0) << " " << t(1) << " " << t(2)
          << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

    }

    f.close();
}

void System::SaveTrajectoryEuRoC(const string &filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.trajectory.save format=euroc kind=frame path=%s", filename.c_str());
    /*if(mSensor==MONOCULAR)
    {
        FT_CALC_LOG("system.trajectory.abort format=euroc reason=monocular_sensor");
        return;
    }*/

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    int numMaxKFs = 0;
    Map* pBiggerMap = nullptr;
    FT_CALC_LOG("system.trajectory.maps count=%llu", static_cast<unsigned long long>(vpMaps.size()));
    for(Map* pMap :vpMaps)
    {
        FT_CALC_LOG("system.trajectory.map id=%llu keyframes=%llu",
                    static_cast<unsigned long long>(pMap->GetId()),
                    static_cast<unsigned long long>(pMap->GetAllKeyFrames().size()));
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    if(!pBiggerMap || pBiggerMap->GetAllKeyFrames().empty())
    {
        FT_CALC_LOG("system.trajectory.abort format=euroc reason=no_keyframes");
        return;
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b depending on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose();
    else
        Twb = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


    for(auto lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //cout << "1" << endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //cout << "2.5" << endl;

        if(!pKF->ResolveTrajectoryReference(pKF, Trw))
            continue;

        if(!pKF || pKF->GetMap() != pBiggerMap)
        {
            //cout << "--Parent KF is from another map" << endl;
            continue;
        }

        //cout << "3" << endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;

        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f Twc = ((*lit)*Trw).inverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f twc = Twc.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    //cout << "end saving trajectory" << endl;
    f.close();
    FT_CALC_LOG("system.trajectory.saved format=euroc kind=frame path=%s", filename.c_str());
}

void System::SaveTrajectoryEuRoC(const string &filename, Map* pMap)
{
    FT_CALC_SCOPE();
    if(!pMap)
    {
        FT_CALC_LOG("system.trajectory.abort format=euroc reason=null_map");
        return;
    }
    FT_CALC_LOG("system.trajectory.save format=euroc kind=frame map_id=%llu path=%s",
                static_cast<unsigned long long>(pMap->GetId()), filename.c_str());
    /*if(mSensor==MONOCULAR)
    {
        FT_CALC_LOG("system.trajectory.abort format=euroc reason=monocular_sensor");
        return;
    }*/

    int numMaxKFs = 0;

    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
    if(vpKFs.empty())
    {
        FT_CALC_LOG("system.trajectory.abort format=euroc reason=no_keyframes map_id=%llu",
                    static_cast<unsigned long long>(pMap->GetId()));
        return;
    }
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose();
    else
        Twb = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


    for(auto lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //cout << "1" << endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //cout << "2.5" << endl;

        if(!pKF->ResolveTrajectoryReference(pKF, Trw))
            continue;
        if(!pKF || pKF->GetMap() != pMap)
        {
            //cout << "--Parent KF is from another map" << endl;
            continue;
        }

        //cout << "3" << endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;

        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f Twc = ((*lit)*Trw).inverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f twc = Twc.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    //cout << "end saving trajectory" << endl;
    f.close();
    FT_CALC_LOG("system.trajectory.saved format=euroc kind=frame map_id=%llu path=%s",
                static_cast<unsigned long long>(pMap->GetId()), filename.c_str());
}

/*void System::SaveTrajectoryEuRoC(const string &filename)
{

    cout << endl << "Saving trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose_();
    else
        Twb = vpKFs[0]->GetPoseInverse_();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //cout << "1" << endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //cout << "2.5" << endl;

        if(!pKF->ResolveTrajectoryReference(pKF, Trw))
            continue;

        if(!pKF || pKF->GetMap() != pBiggerMap)
        {
            //cout << "--Parent KF is from another map" << endl;
            continue;
        }

        //cout << "3" << endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;


        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Tbw = pKF->mImuCalib.Tbc_ * (*lit) * Trw;
            Sophus::SE3f Twb = Tbw.inverse();

            Eigen::Vector3f twb = Twb.translation();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f Tcw = (*lit) * Trw;
            Sophus::SE3f Twc = Tcw.inverse();

            Eigen::Vector3f twc = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    //cout << "end saving trajectory" << endl;
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}*/


/*void System::SaveKeyFrameTrajectoryEuRoC_old(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            cv::Mat R = pKF->GetImuRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat twb = pKF->GetImuPosition();
            f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb.at<float>(0) << " " << twb.at<float>(1) << " " << twb.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        }
        else
        {
            cv::Mat R = pKF->GetRotation();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }
    }
    f.close();
}*/

void System::SaveKeyFrameTrajectoryEuRoC(const string &filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.trajectory.save format=euroc kind=keyframe path=%s", filename.c_str());

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap = nullptr;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap && pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    if(!pBiggerMap)
    {
        FT_CALC_LOG("system.trajectory.abort format=euroc reason=no_map");
        return;
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(!pKF || pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = pKF->GetImuPose();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

        }
        else
        {
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
    }
    f.close();
}

void System::SaveKeyFrameTrajectoryEuRoC(const string &filename, Map* pMap)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.trajectory.save format=euroc kind=keyframe map_id=%llu path=%s",
                static_cast<unsigned long long>(pMap->GetId()), filename.c_str());

    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

        if(!pKF || pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = pKF->GetImuPose();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

        }
        else
        {
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
    }
    f.close();
}

/*void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM3::KeyFrame* pKF = *lRit;

        Sophus::SE3f TReferenceToRoot;
        if(!pKF || !pKF->ResolveTrajectoryReference(pKF, TReferenceToRoot))
            continue;
        cv::Mat Trw = Converter::toCvMat(TReferenceToRoot.matrix());

        Trw = Trw * pKF->GetPoseCv() * Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
}*/

void System::SaveTrajectoryKITTI(const string &filename)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.trajectory.save format=kitti kind=camera path=%s", filename.c_str());
    if(mSensor==MONOCULAR)
    {
        FT_CALC_LOG("system.trajectory.abort format=kitti reason=monocular_sensor");
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
    if(vpKFs.empty())
    {
        FT_CALC_LOG("system.trajectory.abort format=kitti reason=no_keyframes");
        return;
    }
    Map* pExportMap = vpKFs[0]->GetMap();
    if(!pExportMap)
    {
        FT_CALC_LOG("system.trajectory.abort format=kitti reason=detached_export_map");
        return;
    }

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Tow = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM3::KeyFrame* pKF = *lRit;

        Sophus::SE3f Trw;

        if(!pKF)
            continue;

        if(!pKF->ResolveTrajectoryReference(pKF, Trw))
            continue;
        if(pKF->GetMap() != pExportMap)
            continue;

        Trw = Trw * pKF->GetPose() * Tow;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Matrix3f Rwc = Twc.rotationMatrix();
        Eigen::Vector3f twc = Twc.translation();

        f << setprecision(9) << Rwc(0,0) << " " << Rwc(0,1)  << " " << Rwc(0,2) << " "  << twc(0) << " " <<
             Rwc(1,0) << " " << Rwc(1,1)  << " " << Rwc(1,2) << " "  << twc(1) << " " <<
             Rwc(2,0) << " " << Rwc(2,1)  << " " << Rwc(2,2) << " "  << twc(2) << endl;
    }
    f.close();
}


void System::SaveDebugData(const int &initIdx)
{
    // 0. Save initialization trajectory
    SaveTrajectoryEuRoC("init_FrameTrajectoy_" +to_string(mpLocalMapper->mInitSect)+ "_" + to_string(initIdx)+".txt");

    // 1. Save scale
    ofstream f;
    f.open("init_Scale_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mScale << endl;
    f.close();

    // 2. Save gravity direction
    f.open("init_GDir_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mRwg(0,0) << "," << mpLocalMapper->mRwg(0,1) << "," << mpLocalMapper->mRwg(0,2) << endl;
    f << mpLocalMapper->mRwg(1,0) << "," << mpLocalMapper->mRwg(1,1) << "," << mpLocalMapper->mRwg(1,2) << endl;
    f << mpLocalMapper->mRwg(2,0) << "," << mpLocalMapper->mRwg(2,1) << "," << mpLocalMapper->mRwg(2,2) << endl;
    f.close();

    // 3. Save computational cost
    f.open("init_CompCost_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mCostTime << endl;
    f.close();

    // 4. Save biases
    f.open("init_Biases_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mbg(0) << "," << mpLocalMapper->mbg(1) << "," << mpLocalMapper->mbg(2) << endl;
    f << mpLocalMapper->mba(0) << "," << mpLocalMapper->mba(1) << "," << mpLocalMapper->mba(2) << endl;
    f.close();

    // 5. Save covariance matrix
    f.open("init_CovMatrix_" +to_string(mpLocalMapper->mInitSect)+ "_" +to_string(initIdx)+".txt", ios_base::app);
    f << fixed;
    for(int i=0; i<mpLocalMapper->mcovInertial.rows(); i++)
    {
        for(int j=0; j<mpLocalMapper->mcovInertial.cols(); j++)
        {
            if(j!=0)
                f << ",";
            f << setprecision(15) << mpLocalMapper->mcovInertial(i,j);
        }
        f << endl;
    }
    f.close();

    // 6. Save initialization time
    f.open("init_Time_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mInitTime << endl;
    f.close();
}


System::ActiveMapFrameInfo System::GetActiveMapFrameInfo()
{
    ActiveMapFrameInfo info;
    if(!mpAtlas)
        return info;
    Map* pMap = mpAtlas->GetCurrentMap();
    if(!pMap)
        return info;
    info.valid = true;
    info.map_id = pMap->GetId();
    info.imu_initialized = pMap->isImuInitialized();
    info.inertial_ba1 = pMap->GetIniertialBA1();
    info.inertial_ba2 = pMap->GetIniertialBA2();
    info.change_index = pMap->GetMapChangeIndex();
    info.change_index_notified = pMap->GetLastMapChange();
    info.frame_epoch = pMap->GetFrameEpoch();
    if(mpTracker)
    {
        info.pose_epoch_recorded = mpTracker->mbPoseEpochRecorded;
        info.pose_map_id = mpTracker->mnPoseMapId;
        info.pose_frame_epoch = mpTracker->mnPoseFrameEpoch;
    }
    return info;
}

std::vector<Map::FrameTransform> System::GetActiveMapFrameTransforms()
{
    if(!mpAtlas)
        return {};
    Map* pMap = mpAtlas->GetCurrentMap();
    if(!pMap)
        return {};
    return pMap->GetFrameTransforms();
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

double System::GetTimeFromIMUInit()
{
    double aux = mpLocalMapper->GetCurrKFTime()-mpLocalMapper->mFirstTs;
    if ((aux>0.) && mpAtlas->isImuInitialized())
        return mpLocalMapper->GetCurrKFTime()-mpLocalMapper->mFirstTs;
    else
        return 0.f;
}

bool System::isLost()
{
    if (!mpAtlas->isImuInitialized())
        return false;
    else
    {
        if ((mpTracker->mState==Tracking::LOST)) //||(mpTracker->mState==Tracking::RECENTLY_LOST))
            return true;
        else
            return false;
    }
}


bool System::isFinished()
{
    return (GetTimeFromIMUInit()>0.1);
}

void System::ChangeDataset()
{
    if(mpAtlas->GetCurrentMap()->KeyFramesInMap() < 12)
    {
        mpTracker->ResetActiveMap();
    }
    else
    {
        mpTracker->CreateMapInAtlas();
    }

    mpTracker->NewDataset();
}

float System::GetImageScale()
{
    return mpTracker->GetImageScale();
}

#ifdef REGISTER_TIMES
void System::InsertRectTime(double& time)
{
    mpTracker->vdRectStereo_ms.push_back(time);
}

void System::InsertResizeTime(double& time)
{
    mpTracker->vdResizeImage_ms.push_back(time);
}

void System::InsertTrackTime(double& time)
{
    mpTracker->vdTrackTotal_ms.push_back(time);
}
#endif

void System::SaveAtlas(int type){
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.atlas.save.begin type=%d configured=%d", type, mStrSaveAtlasToFile.empty() ? 0 : 1);
    if(!mStrSaveAtlasToFile.empty())
    {
        //clock_t start = clock();

        // Save the current session
        mpAtlas->PreSave();

        string pathSaveFileName = "./";
        pathSaveFileName = pathSaveFileName.append(mStrSaveAtlasToFile);
        pathSaveFileName = pathSaveFileName.append(".osa");

        string strVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);
        std::size_t found = mStrVocabularyFilePath.find_last_of("/\\");
        string strVocabularyName = mStrVocabularyFilePath.substr(found+1);

        if(type == TEXT_FILE) // File text
        {
            FT_CALC_LOG("system.atlas.save.file type=text path=%s", pathSaveFileName.c_str());
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::text_oarchive oa(ofs);

            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            FT_CALC_LOG("system.atlas.save.file_complete type=text");
        }
        else if(type == BINARY_FILE) // File binary
        {
            FT_CALC_LOG("system.atlas.save.file type=binary path=%s", pathSaveFileName.c_str());
            const SaveMapResult r = WriteAtlasFile(pathSaveFileName);  // PreSave again: idempotent
            FT_CALC_LOG("system.atlas.save.file_complete type=binary ok=%d reason=%s", r.ok ? 1 : 0, r.reason.c_str());
        }
    }
}

bool System::LoadAtlas(int type)
{
    FT_CALC_SCOPE();
    string strFileVoc, strVocChecksum;
    bool isRead = false;

    string pathLoadFileName = "./";
    pathLoadFileName = pathLoadFileName.append(mStrLoadAtlasFromFile);
    pathLoadFileName = pathLoadFileName.append(".osa");

    if(type == TEXT_FILE) // File text
    {
        FT_CALC_LOG("system.atlas.load.file type=text path=%s", pathLoadFileName.c_str());
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if(!ifs.good())
        {
            FT_CALC_LOG("system.atlas.load.abort reason=file_not_found path=%s", pathLoadFileName.c_str());
            return false;
        }
        boost::archive::text_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        FT_CALC_LOG("system.atlas.load.file_complete type=text");
        isRead = true;
    }
    else if(type == BINARY_FILE) // File binary
    {
        FT_CALC_LOG("system.atlas.load.file type=binary path=%s", pathLoadFileName.c_str());
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if(!ifs.good())
        {
            FT_CALC_LOG("system.atlas.load.abort reason=file_not_found path=%s", pathLoadFileName.c_str());
            return false;
        }
        boost::archive::binary_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        FT_CALC_LOG("system.atlas.load.file_complete type=binary");
        isRead = true;
    }

    if(isRead)
    {
        //Check if the vocabulary is the same
        string strInputVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);

        if(strInputVocabularyChecksum.compare(strVocChecksum) != 0)
        {
            FT_CALC_LOG("system.atlas.load.abort reason=vocabulary_checksum_mismatch vocabulary=%s",
                        strFileVoc.c_str());
            return false; // Both are differents
        }

        mpAtlas->SetKeyFrameDababase(mpKeyFrameDatabase);
        mpAtlas->SetORBVocabulary(mpVocabulary);
        mpAtlas->PostLoad();

        return true;
    }
    return false;
}

namespace
{
bool FsyncPath(const std::string &path, bool bDirectory)
{
    const int fd = ::open(path.c_str(), bDirectory ? (O_RDONLY | O_DIRECTORY) : O_RDONLY);
    if(fd < 0)
        return false;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

std::string DirName(const std::string &path)
{
    const std::size_t p = path.find_last_of('/');
    if(p == std::string::npos)
        return ".";
    if(p == 0)
        return "/";
    return path.substr(0, p);
}

// Value of "key":"..." in a small flat JSON object written by WriteAtlasFile.
std::string JsonString(const std::string &json, const std::string &key)
{
    const std::string k = "\"" + key + "\":\"";
    const std::size_t a = json.find(k);
    if(a == std::string::npos)
        return "";
    const std::size_t b = json.find('"', a + k.size());
    if(b == std::string::npos)
        return "";
    return json.substr(a + k.size(), b - a - k.size());
}
}  // namespace

bool System::LoadAtlasFile(const std::string &path, PriorMapStatus &st)
{
    FT_CALC_SCOPE();
    st = PriorMapStatus();
    st.state = PriorMapStatus::kLoadFailed;
    st.path = path;
    struct stat sb;
    if(::stat(path.c_str(), &sb) != 0 || !S_ISREG(sb.st_mode))
    {
        st.reason = "file not found: " + path;
        return false;
    }
    st.file_bytes = static_cast<uint64_t>(sb.st_size);
    if(sb.st_size <= 0)
    {
        st.reason = "file is empty: " + path;
        return false;
    }
    if(static_cast<double>(sb.st_size) > mPriorOpt.max_file_mb * 1024.0 * 1024.0)
    {
        char b[200];
        std::snprintf(b, sizeof(b), "file is %.1f MB, limit %.1f MB", sb.st_size / 1048576.0, mPriorOpt.max_file_mb);
        st.reason = b;
        return false;
    }
    st.file_md5 = CalculateCheckSum(path, BINARY_FILE);
    {
        std::ifstream js(path + ".json");
        if(js.good())
        {
            const std::string json((std::istreambuf_iterator<char>(js)), std::istreambuf_iterator<char>());
            const std::string md5 = JsonString(json, "md5");
            if(!md5.empty() && md5 != st.file_md5)
            {
                st.reason = "file md5 " + st.file_md5 + " does not match its sidecar (" + md5 + "): corrupt or replaced";
                return false;
            }
        }
    }
    Atlas* pLoaded = nullptr;
    try
    {
        std::ifstream ifs(path, std::ios::binary);
        if(!ifs.good())
        {
            st.reason = "cannot open: " + path;
            return false;
        }
        boost::archive::binary_iarchive ia(ifs);
        string strFileVoc, strVocChecksum;
        ia >> strFileVoc;
        ia >> strVocChecksum;
        const string strInputVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath, TEXT_FILE);
        if(strInputVocabularyChecksum.empty() || strInputVocabularyChecksum != strVocChecksum)
        {
            st.reason = "vocabulary differs from the one the map was built with (" + strFileVoc + ")";
            return false;
        }
        ia >> pLoaded;
    }
    catch(const std::exception &e)
    {
        st.reason = std::string("cannot deserialize: ") + e.what();
        return false;
    }
    if(!pLoaded)
    {
        st.reason = "no atlas in the file";
        return false;
    }
    try
    {
        pLoaded->SetKeyFrameDababase(mpKeyFrameDatabase);
        pLoaded->SetORBVocabulary(mpVocabulary);
        pLoaded->PostLoad();
    }
    catch(const std::exception &e)
    {
        st.reason = std::string("post-load failed: ") + e.what();
        return false;
    }
    // The deserialized atlas has no current map; one is needed before SetInertialSensor and the tracker.
    Map* pLargest = nullptr;
    for(Map* pMi : pLoaded->GetAllMaps())
        if(pMi && (!pLargest || pMi->KeyFramesInMap() > pLargest->KeyFramesInMap()))
            pLargest = pMi;
    if(!pLargest)
    {
        st.reason = "the file holds no map";
        return false;
    }
    pLoaded->ChangeMap(pLargest);
    mpAtlas = pLoaded;  // the empty Atlas(0) created before is not used
    st.reason = "loaded";
    return true;
}

SaveMapResult System::WriteAtlasFile(const std::string &path)
{
    FT_CALC_SCOPE();
    SaveMapResult r;
    r.path = path;
    const auto t0 = std::chrono::steady_clock::now();
    {
        struct statvfs vfs;
        if(::statvfs(DirName(path).c_str(), &vfs) == 0)
        {
            const double freeMb = static_cast<double>(vfs.f_bavail) * vfs.f_frsize / 1048576.0;
            if(freeMb < 256.0)
            {
                char b[160];
                std::snprintf(b, sizeof(b), "only %.0f MB free in %s", freeMb, DirName(path).c_str());
                r.reason = b;
                return r;
            }
        }
        else
        {
            r.reason = "directory not accessible: " + DirName(path) + " (" + std::strerror(errno) + ")";
            return r;
        }
    }

    mpAtlas->PreSave();
    for(Map* pMi : mpAtlas->GetAllMaps())
    {
        if(!pMi || pMi->IsBad())
            continue;
        r.maps++;
        r.keyframes += pMi->KeyFramesInMap();
        r.map_points += pMi->MapPointsInMap();
    }

    const string strVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath, TEXT_FILE);
    const std::size_t found = mStrVocabularyFilePath.find_last_of("/\\");
    const string strVocabularyName = mStrVocabularyFilePath.substr(found + 1);
    const std::string tmp = path + ".tmp." + std::to_string(::getpid());
    try
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if(!ofs.good())
        {
            r.reason = "cannot create " + tmp + " (" + std::strerror(errno) + ")";
            return r;
        }
        {
            boost::archive::binary_oarchive oa(ofs);
            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
        }
        ofs.flush();
        if(!ofs.good())
        {
            ofs.close();
            std::remove(tmp.c_str());
            r.reason = "write error on " + tmp;
            return r;
        }
        ofs.close();
    }
    catch(const std::exception &e)
    {
        std::remove(tmp.c_str());
        r.reason = std::string("serialization failed: ") + e.what();
        return r;
    }
    if(!FsyncPath(tmp, false) || std::rename(tmp.c_str(), path.c_str()) != 0)
    {
        r.reason = "cannot commit " + path + " (" + std::strerror(errno) + ")";
        std::remove(tmp.c_str());
        return r;
    }
    FsyncPath(DirName(path), true);

    struct stat sb;
    if(::stat(path.c_str(), &sb) == 0)
        r.bytes = static_cast<uint64_t>(sb.st_size);
    r.md5 = CalculateCheckSum(path, BINARY_FILE);

    // Sidecar: identity and content summary (LoadAtlasFile checks the md5).
    char tbuf[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    char json[1024];
    std::snprintf(json, sizeof(json),
                  "{\"format\":\"fasttrack-atlas-binary\",\"file\":\"%s\",\"md5\":\"%s\",\"bytes\":%llu,\"created_utc\":\"%s\","
                  "\"vocabulary\":\"%s\",\"vocabulary_md5\":\"%s\",\"maps\":%lu,\"keyframes\":%lu,\"map_points\":%lu,"
                  "\"active_map\":%lu}\n",
                  path.c_str(), r.md5.c_str(), static_cast<unsigned long long>(r.bytes), tbuf, strVocabularyName.c_str(),
                  strVocabularyChecksum.c_str(), r.maps, r.keyframes, r.map_points,
                  static_cast<unsigned long>(mpAtlas->GetCurrentMap() ? mpAtlas->GetCurrentMap()->GetId() : 0));
    const std::string jtmp = path + ".json.tmp." + std::to_string(::getpid());
    {
        std::ofstream js(jtmp, std::ios::trunc);
        js << json;
    }
    if(!FsyncPath(jtmp, false) || std::rename(jtmp.c_str(), (path + ".json").c_str()) != 0)
    {
        std::remove(jtmp.c_str());
        r.reason = "map saved, sidecar not written (" + std::string(std::strerror(errno)) + ")";
    }
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    r.ok = true;
    FT_CALC_LOG("system.save_map.ok path=%s bytes=%llu md5=%s maps=%lu keyframes=%lu points=%lu seconds=%f",
                path.c_str(), static_cast<unsigned long long>(r.bytes), r.md5.c_str(), r.maps, r.keyframes, r.map_points,
                r.seconds);
    return r;
}

string System::CalculateCheckSum(string filename, int type)
{
    FT_CALC_SCOPE();
    FT_CALC_LOG("system.checksum.begin path=%s type=%d", filename.c_str(), type);
    string checksum = "";

    unsigned char c[MD5_DIGEST_LENGTH];

    std::ios_base::openmode flags = std::ios::in;
    if(type == BINARY_FILE) // Binary file
        flags = std::ios::in | std::ios::binary;

    ifstream f(filename.c_str(), flags);
    if ( !f.is_open() )
    {
        FT_CALC_LOG("system.checksum.abort reason=file_open_failed path=%s", filename.c_str());
        return checksum;
    }

    MD5_CTX md5Context;
    char buffer[1024];

    MD5_Init (&md5Context);
    while ( int count = f.readsome(buffer, sizeof(buffer)))
    {
        MD5_Update(&md5Context, buffer, count);
    }

    f.close();

    MD5_Final(c, &md5Context );

    for(int i = 0; i < MD5_DIGEST_LENGTH; i++)
    {
        char aux[10];
        sprintf(aux,"%02x", c[i]);
        checksum = checksum + aux;
    }

    FT_CALC_LOG("system.checksum.end path=%s length=%llu", filename.c_str(),
                static_cast<unsigned long long>(checksum.size()));
    return checksum;
}

} //namespace ORB_SLAM
