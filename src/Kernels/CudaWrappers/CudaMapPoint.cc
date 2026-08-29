#include "Kernels/CudaWrappers/CudaMapPoint.h"
#include "CalculationLogger.h"

namespace DATA_WRAPPER
{
    CudaMapPoint::CudaMapPoint() {
        FT_CALC_SCOPE();
        isEmpty = true;
    }

    CudaMapPoint::CudaMapPoint(ORB_SLAM3::MapPoint* mp) {
        FT_CALC_SCOPE();
        isEmpty = false;
        mnId = mp->mnId;
        mbTrackInView = mp->mbTrackInView;
        mbTrackInViewR = mp->mbTrackInViewR;
        mTrackDepth = mp->mTrackDepth;
        mnTrackScaleLevel = mp->mnTrackScaleLevel;
        mTrackViewCos = mp->mTrackViewCos;
        mTrackProjX = mp->mTrackProjX;
        mTrackProjY = mp->mTrackProjY;
        mnTrackScaleLevelR = mp->mnTrackScaleLevelR;
        mTrackViewCosR = mp->mTrackViewCosR;
        mTrackProjXR = mp->mTrackProjXR;
        mTrackProjYR = mp->mTrackProjYR;
        mnLastFrameSeen = mp->mnLastFrameSeen;
        nObs = mp->Observations();
        const cv::Mat& descriptor = mp->GetDescriptor();
        std::memcpy(mDescriptor, descriptor.ptr<uint8_t>(0), descriptor.cols * sizeof(uint8_t));
        mWorldPos = mp->GetWorldPos();
        FT_CALC_LOG("cuda_map_point.wrapped id=%llu observations=%d depth=%f scale=%d",
                    static_cast<unsigned long long>(mnId), nObs, mTrackDepth, mnTrackScaleLevel);
    }
}
