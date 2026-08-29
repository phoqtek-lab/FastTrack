/**
 * FastTrack calculation tracing backed by the reviewed NanoLog runtime in
 * garudanav_system.
 */
#ifndef FASTTRACK_CALCULATION_LOGGER_H
#define FASTTRACK_CALCULATION_LOGGER_H

#include <atomic>

#include "CalculationLogApi.h"
#include "NanoLogCpp17.h"

namespace ORB_SLAM3
{

namespace CalculationLog
{

// Starts one process-wide NanoLog stream. The output path comes from
// FASTTRACK_CALCULATION_LOG or a unique /tmp path when unset.
void initialize();

// Warm the NanoLog thread-local staging buffer before a worker enters a hot
// path. This keeps the first traced operation from paying the allocation cost.
void preallocate() noexcept;

// Stops accepting events and finalizes the NanoLog stream. Call only after
// all estimator worker threads have stopped producing events.
void shutdown();

// This acquire load is the only guard on the producer path. It avoids any
// stream formatting or argument evaluation when tracing is unavailable.
bool enabled() noexcept;

class Scope
{
public:
  explicit Scope(const char *name) noexcept;
  ~Scope() noexcept;

  Scope(const Scope &) = delete;
  Scope & operator=(const Scope &) = delete;

private:
  const char *name_;
  bool active_;
};

}  // namespace CalculationLog

}  // namespace ORB_SLAM3

// NanoLog's format string is compile-time parsed and its producer path writes
// directly into thread-local staging storage. No expression is evaluated
// unless the stream is active.
#define FT_CALC_LOG(format, ...)                                                                  \
  do {                                                                                            \
    if (::ORB_SLAM3::CalculationLog::enabled()) {                                                \
      NANO_LOG(DEBUG, format, ##__VA_ARGS__);                                                     \
    }                                                                                             \
  } while (false)

#define FT_CALC_NOTICE(format, ...)                                                               \
  do {                                                                                            \
    if (::ORB_SLAM3::CalculationLog::enabled()) {                                                \
      NANO_LOG(NOTICE, format, ##__VA_ARGS__);                                                    \
    }                                                                                             \
  } while (false)

#define FT_CALC_SCOPE_JOIN_IMPL(left, right) left##right
#define FT_CALC_SCOPE_JOIN(left, right) FT_CALC_SCOPE_JOIN_IMPL(left, right)
#define FT_CALC_SCOPE()                                                                          \
  ::ORB_SLAM3::CalculationLog::Scope FT_CALC_SCOPE_JOIN(ft_calculation_scope_, __LINE__)(__func__)

#endif  // FASTTRACK_CALCULATION_LOGGER_H
