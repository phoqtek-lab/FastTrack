// Host-side declaration-only bridge for inline FastTrack diagnostics.
//
// This header intentionally does not include NanoLog.  Frame/KeyFrame and
// CUDA-facing headers can therefore report rare diagnostics without pulling
// the host-only NanoLog implementation into device compilation.
#ifndef FASTTRACK_CALCULATION_LOG_API_H
#define FASTTRACK_CALCULATION_LOG_API_H

namespace ORB_SLAM3
{
namespace CalculationLog
{
void text(const char *message) noexcept;
void textf(const char *format, ...) noexcept;
void pointDistribution(const char *owner, int left, int right) noexcept;
void matrixCheckFailure(const char *kind, int rows, int cols) noexcept;
void imuMeasurementTimestamp(double timestamp) noexcept;
void missingParameter(const char *name, bool required) noexcept;
}  // namespace CalculationLog
}  // namespace ORB_SLAM3

#endif  // FASTTRACK_CALCULATION_LOG_API_H
