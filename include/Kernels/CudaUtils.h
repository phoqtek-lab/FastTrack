#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <vector>
#include <opencv2/opencv.hpp>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

// Flag to activate the measurement of time in each kernel.
// #define REGISTER_STATS

#define N_FEATURES_TH 20
#define DESCRIPTOR_SIZE 32

class CudaUtils {
    public:
        static void loadSetting(int _nFeatures, int _nLevels, bool _isMonocular, float _scaleFactor, int _nCols, int _nRows, bool _cameraIsFisheye);
        static void shutdown();

    public:
        static int nFeatures_with_th;
        static int nLevels; 
        static bool isMonocular; 
        static float scaleFactor; 
        static int nCols;
        static int nRows;
        static int keypointsPerCell;
        static int maxNumOfMapPoints;
        static int ORBmatcher_TH_HIGH;
        static int ORBmatcher_TH_LOW;
        static int ORBmatcher_HISTO_LENGTH;
        static float* d_mvScaleFactors;
        static bool cameraIsFisheye;
        // Per-level FAST candidate capacity for the GPU corner buffers.
        // Replaces the old pixel-count sizing (w*h per level), which over-allocated
        // multiple GB per frame on 720p. The budget derives from the configured
        // image resolution only (max(w*h/16, 16384), set in allocMemory) so it is
        // identical across the primary and 5*nfeatures map-init extractors;
        // corner buffers are sized level*budget.
        static int maxCandidatesPerLevel;
};

void checkCudaError(cudaError_t err, const char* msg);
__device__ int DescriptorDistance(const uint8_t *a, const uint8_t *b);

#endif // CUDA_UTILS_H