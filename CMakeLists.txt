cmake_minimum_required(VERSION 3.10) # Upgraded for native target-based CUDA support
project(ORB_SLAM3 LANGUAGES CUDA CXX C)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE(STATUS "Configuring ORB_SLAM3 Build Type: " ${CMAKE_BUILD_TYPE})

# Avoid warning floods in the terminal output
add_compile_options(-w)

# =============================================================================
# JETSON ORIN NANO HARDWARE ACCELERATION
# =============================================================================
# Hyper-optimize the ARM Cortex-A78AE CPU cores (enables vector pipelines, unrolling, and fast math)
# =============================================================================
# JETSON ORIN NANO HARDWARE ACCELERATION
# =============================================================================
set(ORIN_NANO_CPU_FLAGS "-march=armv8.2-a+fp16+rcpc+dotprod+crypto -mtune=cortex-a78 -funroll-loops")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3 ${OpenMP_CXX_FLAGS}")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ORIN_NANO_CPU_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ORIN_NANO_CPU_FLAGS}")

# Hyper-optimize the 1024-core Ampere GPU and loosen strict host-header type-matching conflicts
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "Orin Nano GPU target" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xcompiler -w --relaxed-constexpr" CACHE STRING "Optimized GPU arithmetic" FORCE)
# =============================================================================

# =============================================================================

# Enforce standard C++14 compliance cleanly
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DCOMPILEDWITHC11)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4.4 REQUIRED)
MESSAGE(STATUS "Found OpenCV Version: ${OpenCV_VERSION}")

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(realsense2)

set(FASTTRACK_CUDA_INCLUDE_DIR "/home/fci/Desktop/jetson_cuda_includes" CACHE PATH
    "Fallback CUDA host headers copied from the Jetson when toolkit headers are unavailable")
if(EXISTS "${FASTTRACK_CUDA_INCLUDE_DIR}/cuda.h")
  message(STATUS "Using FastTrack CUDA host headers from ${FASTTRACK_CUDA_INCLUDE_DIR}")
  include_directories("${FASTTRACK_CUDA_INCLUDE_DIR}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${FASTTRACK_CUDA_INCLUDE_DIR}")
endif()

set(GARUDANAV_SYSTEM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../garudanav_system" CACHE PATH
    "garudanav_system checkout providing the reviewed NanoLog runtime")
set(GARUDANAV_NANOLOG_RUNTIME_DIR "${GARUDANAV_SYSTEM_DIR}/third_party/nanolog/runtime")
if(NOT EXISTS "${GARUDANAV_NANOLOG_RUNTIME_DIR}/NanoLogCpp17.h")
  message(FATAL_ERROR
    "NanoLog runtime not found under ${GARUDANAV_NANOLOG_RUNTIME_DIR}; "
    "set GARUDANAV_SYSTEM_DIR to the garudanav_system checkout")
endif()

add_library(fasttrack_nanolog_runtime STATIC
  "${GARUDANAV_NANOLOG_RUNTIME_DIR}/Log.cc"
  "${GARUDANAV_NANOLOG_RUNTIME_DIR}/RuntimeLogger.cc"
  "${GARUDANAV_NANOLOG_RUNTIME_DIR}/NanoLog.cc"
  "${GARUDANAV_NANOLOG_RUNTIME_DIR}/Cycles.cc"
  "${GARUDANAV_NANOLOG_RUNTIME_DIR}/Util.cc"
  "${GARUDANAV_NANOLOG_RUNTIME_DIR}/TimeTrace.cc"
  "${GARUDANAV_NANOLOG_RUNTIME_DIR}/Perf.cc"
  "${GARUDANAV_NANOLOG_RUNTIME_DIR}/PerfHelper.cc"
  "${GARUDANAV_SYSTEM_DIR}/src/nanolog_stubs.cc"
)
set_target_properties(fasttrack_nanolog_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fasttrack_nanolog_runtime PUBLIC "${GARUDANAV_NANOLOG_RUNTIME_DIR}")
target_compile_features(fasttrack_nanolog_runtime PUBLIC cxx_std_17)
target_compile_options(fasttrack_nanolog_runtime PRIVATE
  -Wno-sign-compare
  -Wno-unused-parameter
  -Wno-reorder
  -Wno-format-truncation
  -Wno-volatile
)
target_link_libraries(fasttrack_nanolog_runtime PUBLIC Threads::Threads rt)

add_library(fasttrack_calculation_logger STATIC src/CalculationLogger.cc)
set_target_properties(fasttrack_calculation_logger PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fasttrack_calculation_logger PUBLIC include)
target_link_libraries(fasttrack_calculation_logger PUBLIC fasttrack_nanolog_runtime)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/CameraModels
  ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Explicitly list FastTrack CUDA and C++ source frames
add_library(${PROJECT_NAME} SHARED
  src/System.cc
  src/Tracking.cc
  src/LocalMapping.cc
  src/LoopClosing.cc
  src/ORBextractor.cc
  src/fast.cu
  src/orientation.cu
  src/resize.cu
  src/gaussian_blur.cu
  src/descriptor.cu
  src/ORBmatcher.cc
  src/FrameDrawer.cc
  src/Converter.cc
  src/MapPoint.cc
  src/KeyFrame.cc
  src/Atlas.cc
  src/Map.cc
  src/MapDrawer.cc
  src/Optimizer.cc
  src/Frame.cc
  src/KeyFrameDatabase.cc
  src/Sim3Solver.cc
  src/Viewer.cc
  src/ImuTypes.cc
  src/G2oTypes.cc
  src/CameraModels/Pinhole.cpp
  src/CameraModels/KannalaBrandt8.cpp
  src/OptimizableTypes.cpp
  src/MLPnPsolver.cpp
  src/GeometricTools.cc
  src/TwoViewReconstruction.cc
  src/Config.cc
  src/Settings.cc
  src/Stats/TrackingStats.cc
  src/Kernels/CudaUtils.cu
  src/Kernels/CudaWrappers/CudaMapPoint.cc
  src/Kernels/CudaWrappers/CudaFrame.cu
  src/Kernels/StereoMatchKernel.cu
  src/Kernels/SearchLocalPointsKernel.cu
  src/Kernels/PoseEstimationKernel.cu
  src/Kernels/KernelController.cu
)

# Mandatory separate compilation rule for custom FastTrack device kernels
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
  fasttrack_calculation_logger
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
  -lboost_serialization
  -lcrypto
  OpenMP::OpenMP_CXX
)

if(realsense2_FOUND)
    include_directories(${realsense_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY})
endif()

# =============================================================================
# EXECUTABLE BINARY GENERATION MACRO
# =============================================================================
macro(add_orb_executable EXECUTABLE_NAME SOURCE_PATH)
    add_executable(${EXECUTABLE_NAME} ${SOURCE_PATH})
    target_link_libraries(${EXECUTABLE_NAME} ${PROJECT_NAME})
endmacro()

# RGB-D Executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
add_orb_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc)
if(realsense2_FOUND)
    add_orb_executable(rgbd_realsense_D435i Examples/RGB-D/rgbd_realsense_D435i.cc)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)
    add_orb_executable(rgbd_inertial_realsense_D435i Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
endif()

# Stereo Executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)
add_orb_executable(stereo_kitti Examples/Stereo/stereo_kitti.cc)
add_orb_executable(stereo_euroc Examples/Stereo/stereo_euroc.cc)
add_orb_executable(stereo_tum_vi Examples/Stereo/stereo_tum_vi.cc)
if(realsense2_FOUND)
    add_orb_executable(stereo_realsense_t265 Examples/Stereo/stereo_realsense_t265.cc)
    add_orb_executable(stereo_realsense_D435i Examples/Stereo/stereo_realsense_D435i.cc)
endif()

# Monocular Executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
add_orb_executable(mono_tum Examples/Monocular/mono_tum.cc)
add_orb_executable(mono_kitti Examples/Monocular/mono_kitti.cc)
add_orb_executable(mono_euroc Examples/Monocular/mono_euroc.cc)
add_orb_executable(mono_tum_vi Examples/Monocular/mono_tum_vi.cc)
if(realsense2_FOUND)
    add_orb_executable(mono_realsense_t265 Examples/Monocular/mono_realsense_t265.cc)
    add_orb_executable(mono_realsense_D435i Examples/Monocular/mono_realsense_D435i.cc)
endif()

# Monocular Inertial Executables (Crucial For Your Drone Setup)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)
add_orb_executable(mono_inertial_euroc Examples/Monocular-Inertial/mono_inertial_euroc.cc)
add_orb_executable(mono_inertial_tum_vi Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
if(realsense2_FOUND)
    add_orb_executable(mono_inertial_realsense_t265 Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    add_orb_executable(mono_inertial_realsense_D435i Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
endif()

# Stereo Inertial Executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)
add_orb_executable(stereo_inertial_euroc Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
add_orb_executable(stereo_inertial_tum_vi Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
if(realsense2_FOUND)
    add_orb_executable(stereo_inertial_realsense_t265 Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    add_orb_executable(stereo_inertial_realsense_D435i Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
endif()

# Calibration Recorders
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Calibration)
if(realsense2_FOUND)
    add_orb_executable(recorder_realsense_D435i Examples/Calibration/recorder_realsense_D435i.cc)
    add_orb_executable(recorder_realsense_T265 Examples/Calibration/recorder_realsense_T265.cc)
endif()
