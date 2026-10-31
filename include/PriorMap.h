/**
 * Prior-map start (2026-10-16): initialize a stationary IMU-monocular system by relocalizing into a saved,
 * inertially initialized map of the same site, instead of the motion-based visual-inertial initialization.
 * Map save/load hardening and the session reset that keeps the prior map live here as well.
 * Off unless PriorMapOptions::enable is set; the legacy System.LoadAtlasFromFile / SaveAtlasToFile keys are unchanged.
 */

#ifndef PRIORMAP_H
#define PRIORMAP_H

#include <cstdint>
#include <mutex>
#include <string>

namespace ORB_SLAM3
{

struct PriorMapOptions
{
    bool enable = false;               // relocalization start from load_path
    std::string load_path;             // absolute or cwd-relative .osa file; empty = start without a map
    std::string save_path;             // target of SaveMap() requests and of save_on_shutdown
    bool save_on_shutdown = false;     // save to save_path in Shutdown() (needs an orderly shutdown)
    bool keep_host_sigint = false;     // do not install FastTrack's SIGINT handler (it calls std::exit and skips Shutdown)
    bool keep_only_best_map = true;    // discard the other maps of the loaded atlas (no merges with fragments)
    bool require_stationary = true;    // relocalize only while the ZUPT detector reports the vehicle stationary
    bool allow_hand_init = true;       // non-stationary frames run the normal (motion) initialization in a session map
    bool adopt_after_save = true;      // after SaveMap() the saved map becomes the prior map of this process
    int confirm_frames = 3;            // consecutive stationary relocalizations that must agree before accepting
    double confirm_max_translation_m = 0.05;
    double confirm_max_rotation_deg = 1.0;
    double max_tilt_error_deg = 5.0;   // map gravity vs measured specific force at the relocalized pose
    // 2026-10-22: on the pad the camera sees a defocused marker cell cm away (not recognizable); the mapped scene is
    // visible while the vehicle is lifted / held / carried. Relocalization is therefore also attempted at low rotation
    // rates, from a session map that is already tracking (motion initialization done), until it succeeds.
    bool reloc_while_moving = true;
    int reloc_min_inliers = 30;              // optimized inliers into the prior map (upstream 50)
    bool reloc_mutual_check = false;         // cross-check BoW matches when relocalizing into the prior map
    int relaxed_confirm_frames = 5;          // confirmations when the inliers are below 50
    double moving_max_rate_rad_s = 0.5;     // frame rotation rate limit for a moving attempt (motion blur)
    double quasi_still_rate_rad_s = 0.3;    // without an inertial session map: accept only below this (velocity = 0)
    int moving_probe_every = 5;             // frames between moving attempts until one succeeds
    double moving_confirm_max_rotation_deg = 1.5;  // consecutive relocalized rotation vs the gyro
    double moving_confirm_max_step_m = 0.15;       // consecutive relocalized positions
    int min_keyframes = 20;            // smallest prior map accepted
    double max_file_mb = 512.0;        // larger files are refused (memory)
    double camera_tolerance = 1e-3;    // relative tolerance of the intrinsics / distortion comparison
    double tbc_tolerance_m = 1e-3;     // camera-IMU extrinsics comparison (translation, m)
    double tbc_tolerance_deg = 0.1;    // camera-IMU extrinsics comparison (rotation, deg)
    double gba_wait_s = 60.0;          // SaveMap / ResetToPriorMap wait for a running global BA, then refuse
};

struct PriorMapStatus
{
    enum State
    {
        kDisabled = 0,       // no prior map configured
        kLoadFailed,         // file missing / unreadable / corrupt / vocabulary mismatch / too large
        kRejected,           // loaded, but unusable (not inertial, too small, other camera or IMU calibration)
        kWaiting,            // usable prior map, not initialized: waiting for a stationary frame
        kConfirming,         // stationary relocalization found; waiting for confirm_frames agreeing frames
        kHandInit,           // vehicle moving: normal initialization in a session map (prior map kept)
        kRelocalized,        // tracking in the prior map
        kDiscarded,          // the prior map was reset or removed in this process
    };
    State state = kDisabled;
    std::string reason;
    std::string path;
    uint64_t file_bytes = 0;
    std::string file_md5;
    unsigned long map_id = 0;
    unsigned long keyframes = 0;      // keyframes of the prior map (at load / last reset)
    unsigned long map_points = 0;
    unsigned long attempts = 0;       // stationary relocalization attempts since the last (re)start
    unsigned long successes = 0;      // successful relocalizations (before confirmation) since the last (re)start
    int last_bow_matches = 0;         // best SearchByBoW matches of the last attempt
    int best_bow_matches = 0;         // best since the last (re)start
    int last_inliers = 0;             // inliers of the last successful relocalization
    double last_tilt_error_deg = -1.0;
    double reloc_time = 0.0;          // frame timestamp of the accepted relocalization
    unsigned long relocalizations = 0;// accepted relocalizations in this process
    unsigned long resets = 0;         // ResetToPriorMap executions
    unsigned long saves = 0;          // successful SaveMap calls
    std::string last_save_result;     // "" or "ok ..." / "failed: ..."
};

struct SaveMapResult
{
    bool ok = false;
    std::string reason;
    std::string path;
    uint64_t bytes = 0;
    std::string md5;
    unsigned long maps = 0;
    unsigned long keyframes = 0;
    unsigned long map_points = 0;
    double seconds = 0.0;
};

const char* PriorMapStateName(PriorMapStatus::State s);

} // namespace ORB_SLAM3

#endif // PRIORMAP_H
