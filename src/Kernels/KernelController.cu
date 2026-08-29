#include "Kernels/KernelController.h"
#include "CalculationLogApi.h"

bool KernelController::orbExtractionKernelRunStatus;
bool KernelController::stereoMatchKernelRunStatus;
bool KernelController::searchLocalPointsKernelRunStatus;
bool KernelController::poseEstimationKernelRunStatus;
bool KernelController::poseOptimizationRunStatus;
bool KernelController::memory_is_initialized = false;
bool KernelController::stereoMatchDataHasMovedForward = false;
bool KernelController::cuda_utils_loaded_setting = false;
std::unique_ptr<SearchLocalPointsKernel> KernelController::mpSearchLocalPointsKernel = std::make_unique<SearchLocalPointsKernel>();
std::unique_ptr<PoseEstimationKernel> KernelController::mpPoseEstimationKernel = std::make_unique<PoseEstimationKernel>();
std::unique_ptr<StereoMatchKernel> KernelController::mpStereoMatchKernel = std::make_unique<StereoMatchKernel>();
DATA_WRAPPER::CudaFrame* KernelController::cudaFramePtr;
DATA_WRAPPER::CudaFrame* KernelController::cudaLastFramePtr;

void KernelController::setCUDADevice(int deviceID) {
    cudaSetDevice(deviceID);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    ORB_SLAM3::CalculationLog::textf("cuda.device.selected id=%d name=%s", deviceID, deviceProp.name);
}

void KernelController::setGPURunMode(bool orbExtractionStatus, bool stereoMatchStatus, bool searchLocalPointsStatus, bool poseEstimationStatus, bool poseOptimizationStatus) {
    ORB_SLAM3::CalculationLog::textf("cuda.run_mode orb=%d stereo=%d search_local=%d pose_estimation=%d pose_optimization=%d",
                                     orbExtractionStatus ? 1 : 0, stereoMatchStatus ? 1 : 0,
                                     searchLocalPointsStatus ? 1 : 0, poseEstimationStatus ? 1 : 0,
                                     poseOptimizationStatus ? 1 : 0);
    orbExtractionKernelRunStatus = orbExtractionStatus;
    stereoMatchKernelRunStatus = stereoMatchStatus;
    searchLocalPointsKernelRunStatus = searchLocalPointsStatus;
    poseEstimationKernelRunStatus = poseEstimationStatus;
    poseOptimizationRunStatus = poseOptimizationStatus;
}

void KernelController::initializeKernels(){
    ORB_SLAM3::CalculationLog::text("cuda.kernels.initialize.begin");
    cudaFramePtr = new DATA_WRAPPER::CudaFrame();
    cudaLastFramePtr = new DATA_WRAPPER::CudaFrame();
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point SM_start = std::chrono::steady_clock::now();
#endif
    if(stereoMatchKernelRunStatus == 1)
        mpStereoMatchKernel->initialize();
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point SM_end = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point SLP_start = std::chrono::steady_clock::now();
#endif
    if(searchLocalPointsKernelRunStatus == 1)
        mpSearchLocalPointsKernel->initialize();
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point SLP_end = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point PE_start = std::chrono::steady_clock::now();
#endif
    if(poseEstimationKernelRunStatus == 1)
        mpPoseEstimationKernel->initialize();
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point PE_end = std::chrono::steady_clock::now();
    TrackingStats::getInstance().stereoMatch_init_time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(SM_end - SM_start).count();
    TrackingStats::getInstance().searchLocalPoints_init_time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(SLP_end - SLP_start).count();
    TrackingStats::getInstance().poseEstimation_init_time = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(PE_end - PE_start).count();
#endif
    checkCudaError(cudaDeviceSynchronize(), "[Kernel Controller:] Failed to initialize kernels.");
    memory_is_initialized = true;
    ORB_SLAM3::CalculationLog::text("cuda.kernels.initialize.end");
}

void KernelController::shutdownKernels(){
    ORB_SLAM3::CalculationLog::text("cuda.kernels.shutdown.begin");
    if (memory_is_initialized) {
        cudaFramePtr->freeMemory();
        delete cudaFramePtr;
        if(stereoMatchKernelRunStatus == 1)
            mpStereoMatchKernel->shutdown();
        if(searchLocalPointsKernelRunStatus == 1)
            mpSearchLocalPointsKernel->shutdown();
        if(poseEstimationKernelRunStatus == 1)
            mpPoseEstimationKernel->shutdown();
    }
    if (cuda_utils_loaded_setting) {
        CudaUtils::shutdown();
    }
    ORB_SLAM3::CalculationLog::text("cuda.kernels.shutdown.end");
}

void KernelController::saveKernelsStats(const std::string &file_path){
    mpStereoMatchKernel->saveStats(file_path);
    mpSearchLocalPointsKernel->saveStats(file_path);
    mpPoseEstimationKernel->saveStats(file_path);
}

void KernelController::launchStereoMatchKernel(std::vector<std::vector<int>> &vRowIndices, uchar* d_imagePyramidL, uchar* d_imagePyramidR, 
                                               std::vector<cv::Mat> &mvImagePyramid, std::vector<cv::Mat> &mvImagePyramidRight,
                                               std::vector<cv::KeyPoint> &mvKeys, std::vector<cv::KeyPoint> &mvKeysRight, 
                                               cv::Mat mDescriptors, cv::Mat mDescriptorsRight, const float minD, const float maxD, const int thOrbDist, 
                                               const float mbf, const bool mvImagePyramidOnGpu,
                                               std::vector<std::pair<int, int>> &vDistIdx, std::vector<float> &mvuRight, std::vector<float> &mvDepth) {
    
    ORB_SLAM3::CalculationLog::textf("cuda.launch.stereo frame_features=%llu threshold=%d min_depth=%f max_depth=%f",
                                     static_cast<unsigned long long>(mvKeys.size()), thOrbDist, minD, maxD);

    mpStereoMatchKernel->launch(vRowIndices, d_imagePyramidL, d_imagePyramidR, mvImagePyramid, mvImagePyramidRight, mvKeys, mvKeysRight, 
                                mDescriptors, mDescriptorsRight, minD, maxD, thOrbDist, mbf, mvImagePyramidOnGpu, vDistIdx, mvuRight, mvDepth);

    if (!stereoMatchDataHasMovedForward) {
        cudaFramePtr->setMvKeys( mpStereoMatchKernel->get_d_gpuKeypointsL() );
        cudaFramePtr->setMvKeysRight( mpStereoMatchKernel->get_d_gpuKeypointsR() );    
        cudaFramePtr->setMDescriptors( mpStereoMatchKernel->get_d_descriptorsL() );
        stereoMatchDataHasMovedForward = true;
    }
}

void KernelController::launchFisheyeStereoMatchKernel(const int N, const int Nr, cv::Mat mDescriptors, cv::Mat mDescriptorsRight, int* matches) {

    ORB_SLAM3::CalculationLog::textf("cuda.launch.fisheye_stereo left=%d right=%d", N, Nr);

    mpStereoMatchKernel->launch(N, Nr, mDescriptors, mDescriptorsRight, matches);

    if (!stereoMatchDataHasMovedForward) {
        cudaFramePtr->setMDescriptors( mpStereoMatchKernel->get_d_descriptorsAll() );
        stereoMatchDataHasMovedForward = true;
    }
}

void KernelController::launchSearchLocalPointsKernel(ORB_SLAM3::Frame &F, const vector<ORB_SLAM3::MapPoint*> &vmp,
                                                    const float th, const bool bFarPoints, const float thFarPoints,
                                                    int* h_bestLevel, int* h_bestLevel2, int* h_bestDist, int* h_bestDist2, int* h_bestIdx,
                                                    int* h_bestLevelR, int* h_bestLevelR2, int* h_bestDistR, int* h_bestDistR2, int* h_bestIdxR) {
         
    ORB_SLAM3::CalculationLog::textf("cuda.launch.search_local frame=%llu map_points=%llu threshold=%f far_points=%d far_threshold=%f",
                                     static_cast<unsigned long long>(F.mnId),
                                     static_cast<unsigned long long>(vmp.size()), th,
                                     bFarPoints ? 1 : 0, thFarPoints);
    // if(cudaFramePtr->mnId == NULL || cudaFramePtr->mnId != F.mnId) {
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point startFrameWrap = std::chrono::steady_clock::now();
#endif
        cudaFramePtr->setMemory(F);
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endFrameWrap = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startFrameTransfer = std::chrono::steady_clock::now();
#endif
        mpSearchLocalPointsKernel->setFrame(cudaFramePtr);
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endFrameTransfer = std::chrono::steady_clock::now();
    double frameWrap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endFrameWrap - startFrameWrap).count();
    double frameTransfer = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endFrameTransfer - startFrameTransfer).count();
    mpSearchLocalPointsKernel->frame_wrap_time.emplace_back(F.mnId, frameWrap);
    mpSearchLocalPointsKernel->frame_data_transfer_time.emplace_back(F.mnId, frameTransfer);
#endif
    // }
    
    mpSearchLocalPointsKernel->launch(F, vmp, th, bFarPoints, thFarPoints, 
                                    h_bestLevel, h_bestLevel2, h_bestDist, h_bestDist2, h_bestIdx,
                                    h_bestLevelR, h_bestLevelR2, h_bestDistR, h_bestDistR2, h_bestIdxR);
}

void KernelController::launchPoseEstimationKernel(ORB_SLAM3::Frame &CurrentFrame, const ORB_SLAM3::Frame &LastFrame,
                                                const float th, const bool bForward, const bool bBackward, Eigen::Matrix4f transform_matrix,
                                                int* h_bestDist, int* h_bestIdx2, int* h_bestDistR, int* h_bestIdxR2){

    ORB_SLAM3::CalculationLog::textf("cuda.launch.pose_estimation current=%llu last=%llu threshold=%f forward=%d backward=%d",
                                     static_cast<unsigned long long>(CurrentFrame.mnId),
                                     static_cast<unsigned long long>(LastFrame.mnId), th,
                                     bForward ? 1 : 0, bBackward ? 1 : 0);
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point startFrameWrap = std::chrono::steady_clock::now();
#endif
    cudaLastFramePtr->setMemory(LastFrame);
    cudaFramePtr->setMemory(CurrentFrame);
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endFrameWrap = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point startFrameTransfer = std::chrono::steady_clock::now();
#endif
    mpPoseEstimationKernel->setLastFrame(cudaLastFramePtr);
    mpPoseEstimationKernel->setCurrentFrame(cudaFramePtr);
#ifdef REGISTER_STATS
    std::chrono::steady_clock::time_point endFrameTransfer = std::chrono::steady_clock::now();
    double frameWrap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endFrameWrap - startFrameWrap).count();
    double frameTransfer = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(endFrameTransfer - startFrameTransfer).count();
    mpPoseEstimationKernel->data_wrap_time.emplace_back(CurrentFrame.mnId, frameWrap);
    mpPoseEstimationKernel->input_data_transfer_time.emplace_back(CurrentFrame.mnId, frameTransfer);
#endif

    mpPoseEstimationKernel->launch(CurrentFrame, LastFrame, th, bForward, bBackward, transform_matrix, h_bestDist, h_bestIdx2, h_bestDistR, h_bestIdxR2);
}
