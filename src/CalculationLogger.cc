/**
 * FastTrack calculation tracing backed by the reviewed NanoLog runtime in
 * garudanav_system.
 */
#include "CalculationLogger.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
#include <unistd.h>

namespace
{
std::atomic<bool> g_enabled{false};
std::mutex g_lifecycle_mutex;
std::string g_log_path;

std::string defaultLogPath()
{
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::string("/tmp/fasttrack-calculations-") + std::to_string(static_cast<long long>(::getpid())) +
         "-" + std::to_string(static_cast<long long>(ticks)) + ".nlog";
}

std::string configuredLogPath()
{
  const char *configured = std::getenv("FASTTRACK_CALCULATION_LOG");
  if (configured != nullptr && configured[0] != '\0') {
    return std::string(configured);
  }
  return defaultLogPath();
}
}  // namespace

namespace ORB_SLAM3
{
namespace CalculationLog
{

void initialize()
{
  std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
  if (g_enabled.load(std::memory_order_acquire)) {
    return;
  }

  const std::string path = configuredLogPath();
  try {
    // The order is deliberate: no producer can observe enabled until the
    // output file, thread-local staging storage, and DEBUG level are ready.
    NanoLog::setLogFile(path.c_str());
    NanoLog::preallocate();
    NanoLog::setLogLevel(NanoLog::DEBUG);
    g_log_path = path;
    g_enabled.store(true, std::memory_order_release);
    NANO_LOG(NOTICE, "fasttrack.calculation_logger.start path=%s", g_log_path.c_str());
  } catch (...) {
    g_enabled.store(false, std::memory_order_release);
    try {
      NanoLog::finalize();
    } catch (...) {
      // There is no second logging sink for a failed NanoLog initialization.
    }
  }
}

void preallocate() noexcept
{
  if (!enabled()) {
    return;
  }
  try {
    NanoLog::preallocate();
  } catch (...) {
  }
}

void shutdown()
{
  std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
  if (!g_enabled.load(std::memory_order_acquire)) {
    return;
  }

  // The caller owns the worker-stop precondition. Emit the terminal marker
  // before publishing disabled, then finalize without any producer racing it.
  try {
    NANO_LOG(NOTICE, "fasttrack.calculation_logger.stop");
  } catch (...) {
    // Finalization still runs so a partially written stream is closed.
  }
  g_enabled.store(false, std::memory_order_release);
  try {
    NanoLog::finalize();
  } catch (...) {
    // The runtime has no alternate sink; leave the stream disabled.
  }
  g_log_path.clear();
}

bool enabled() noexcept { return g_enabled.load(std::memory_order_acquire); }

void text(const char *message) noexcept
{
  if (!enabled() || message == nullptr) {
    return;
  }
  try {
    NANO_LOG(NOTICE, "fasttrack.message %s", message);
  } catch (...) {
    // Diagnostics must never alter estimator control flow.
  }
}

void textf(const char *format, ...) noexcept
{
  if (!enabled() || format == nullptr) {
    return;
  }
  try {
    char message[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    NANO_LOG(DEBUG, "fasttrack.message %s", message);
  } catch (...) {
  }
}

void pointDistribution(const char *owner, int left, int right) noexcept
{
  if (!enabled()) {
    return;
  }
  try {
    NANO_LOG(DEBUG, "frame.point_distribution owner=%s left=%d right=%d", owner, left, right);
  } catch (...) {
  }
}

void matrixCheckFailure(const char *kind, int rows, int cols) noexcept
{
  if (!enabled()) {
    return;
  }
  try {
    NANO_LOG(WARNING, "calculation.matrix_check_failure kind=%s rows=%d cols=%d", kind, rows, cols);
  } catch (...) {
  }
}

void imuMeasurementTimestamp(double timestamp) noexcept
{
  if (!enabled()) {
    return;
  }
  try {
    NANO_LOG(DEBUG, "imu.measurement.timestamp t=%f", timestamp);
  } catch (...) {
  }
}

void missingParameter(const char *name, bool required) noexcept
{
  if (!enabled()) {
    return;
  }
  try {
    NANO_LOG(WARNING, "settings.parameter_missing name=%s required=%d", name, required ? 1 : 0);
  } catch (...) {
  }
}

Scope::Scope(const char *name) noexcept : name_(name), active_(enabled())
{
  if (!active_) {
    return;
  }
  try {
    NANO_LOG(DEBUG, "calc.begin %s", name_);
  } catch (...) {
    active_ = false;
  }
}

Scope::~Scope() noexcept
{
  if (!active_ || !enabled()) {
    return;
  }
  try {
    NANO_LOG(DEBUG, "calc.end %s", name_);
  } catch (...) {
    // Scope tracing is observational and cannot throw into estimator code.
  }
}

}  // namespace CalculationLog
}  // namespace ORB_SLAM3
