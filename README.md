# FastTrack

FastTrack is an optimized tracking for ORB-SLAM3 that leverages GPU computing power to accelerate the time-consuming components of tracking and enhance the overall performance. These components include ORB extraction, stereo feature matching, and local map tracking. We implemented CUDA kernels to speed up these components. Our results demonstrate a significant reduction in tracking times, achieving up to 2.8× faster performance on desktop and up to 2.7× speedup on Xavier NX. We evaluated FastTrack on a mix of sequences from the EuRoC and TUM-VI datasets using the stereo-inertial configuration.

🚀 FastTrack has been accepted for publication at [IROS 2025](http://iros25.org/).

<p align="center"><strong>Figure 1: Tracking In FastTrack</strong></p>
<p align="center">
  <img src="https://github.com/sfu-rsl/FastTrack/blob/main/Tracking_in_FastTrack.png" alt="Tracking In FastTrack">
</p>

<div align="center">

<strong>Table 1: Machine Specifications</strong>
<br>

<table>
  <thead>
    <tr>
      <th>Machine</th>
      <th>Specs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Desktop</td>
      <td>
        20-core Intel Core i7-12700K CPU @ 5.0 GHz <br>
        NVIDIA RTX 3090 GPU (10496-core) <br>
        64 GB RAM
      </td>
    </tr>
    <tr>
      <td>Xavier NX</td>
      <td>
        6-core ARM Carmel CPU @ 1.4 GHz <br>
        NVIDIA Volta GPU (384-core) <br>
        8 GB RAM
      </td>
    </tr>
  </tbody>
</table>

</div>


# 2. Prerequisites
We have tested the library in **Ubuntu 22.04** and **20.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++14 or C++0x Compiler
We use the new thread and chrono functionalities of C++14.

## Cuda
We have tested the library with **Cuda 12.2**. Download and install instructions can be found at: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 4.4.0**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## Pangolin (optional)
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Download and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

Pangolin is only needed for the map viewer, which is off by default. Configure with `-DFASTTRACK_WITH_VIEWER=ON` to build it; otherwise FastTrack builds and runs headless without Pangolin.

Pangolin avoids to use Eigen in CUDA. To compile this project, guards in line 475 of glsl.hpp and 47 of glsl.h should be commented. Checkout [this issue](https://github.com/stevenlovegrove/Pangolin/issues/814) in ORB-SLAM3 github issues. 

## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## Python
Required to calculate the alignment of the trajectory with the ground truth. **Required Numpy module**.

* (win) http://www.python.org/downloads/windows
* (deb) `sudo apt install libpython2.7-dev`
* (mac) preinstalled with osx

# 3. Building FastTrack 

Clone the repository:
```
git clone git@github.com:sfu-rsl/FastTrack.git
```

Our system is based on ORB-SLAM3 and ORB-SLAM3 provides a script `build.sh` to build the *Thirdparty* libraries and *ORB-SLAM3*. Please make sure you have installed all required dependencies (see section 2). Execute:
```
cd FastTrack
chmod +x build.sh
sudo ./build.sh
```

This will create **libORB_SLAM3.so**  at *lib* folder and the executables in *Examples* folder.

# 4. Running FastTrack
## EuRoC Examples
[EuRoC dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) was recorded with two pinhole cameras and an inertial sensor. We provide an example script to launch EuRoC sequences in all the sensor configurations.

1. Download a sequence (ASL format) from http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

2. Open the script "euroc_eval_examples.sh" in the **Examples** directory of the project. Change **pathDatasetEuroc** variable to point to the directory where the dataset has been uncompressed. 

3. Execute the following script to process all the sequences with all sensor configurations:
```
./run_script $dataset_name 1
```

## TUM-VI Examples
[TUM-VI dataset](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) was recorded with two fisheye cameras and an inertial sensor.

1. Download a sequence from https://vision.in.tum.de/data/datasets/visual-inertial-dataset and uncompress it.

2. Open the script "tum_vi_examples.sh" in the root of the project. Change **pathDatasetTUM_VI** variable to point to the directory where the dataset has been uncompressed. 

3. Execute the following script to process all the sequences with all sensor configurations:
```
./run_script $dataset_name 1
```
