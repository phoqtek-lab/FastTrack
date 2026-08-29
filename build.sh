#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"
sanitize_build_dir="${script_dir}/build-sanitize"
tsan_build_dir="${script_dir}/build-tsan"
leak_build_dir="${script_dir}/build-leak"
benchmark_build_dir="${script_dir}/build-benchmark"
lint_quick_build_dir="${script_dir}/build-lint-quick"
shared_system_dir="${GARUDANAV_SYSTEM_DIR:-}"
cuda_include_dir="${JETSON_CUDA_INCLUDES:-}"

usage() {
  echo "Usage: $0 [build|test|lint-quick|lint|sanitize|tsan|leak-check|benchmark|all|clean]" >&2
  echo "  GARUDANAV_BUILD_JOBS=<n>  override the compile job count (default: memory-aware)" >&2
  echo "  GARUDANAV_SYSTEM_DIR=<path>  garudanav_system checkout providing NanoLog" >&2
  echo "  JETSON_CUDA_INCLUDES=<path>  fallback CUDA host include directory" >&2
}

build_jobs() {
  if [[ -n "${GARUDANAV_BUILD_JOBS:-}" ]]; then
    echo "$GARUDANAV_BUILD_JOBS"
    return
  fi

  local cores
  cores="$(nproc)"
  local avail_kb
  avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  local jobs="$cores"
  if [[ -n "$avail_kb" ]]; then
    local mem_jobs=$((avail_kb / 2097152))
    if [[ "$mem_jobs" -lt "$jobs" ]]; then
      jobs="$mem_jobs"
    fi
  fi
  if [[ "$jobs" -lt 1 ]]; then
    jobs=1
  fi
  echo "$jobs"
}

parallel_jobs="$(build_jobs)"

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "build.sh: required tool '$1' not found on PATH" >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "build.sh: required file not found: $1" >&2
    exit 1
  fi
}

configure_build() {
  local target_dir="$1"
  shift
  local cmake_args=(
    -S "$script_dir"
    -B "$target_dir"
    -G Ninja
    "-DCMAKE_BUILD_TYPE=${FASTTRACK_BUILD_TYPE:-RelWithDebInfo}"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  )
  if [[ -n "$shared_system_dir" ]]; then
    cmake_args+=("-DGARUDANAV_SYSTEM_DIR=$shared_system_dir")
  fi
  if [[ -n "$cuda_include_dir" ]]; then
    cmake_args+=("-DFASTTRACK_CUDA_INCLUDE_DIR=$cuda_include_dir")
  fi
  cmake "${cmake_args[@]}" "$@"
}

build_project() {
  local target_dir="$1"
  shift
  require_tool cmake
  require_tool ninja
  configure_build "$target_dir" "$@"
  cmake --build "$target_dir" --parallel "$parallel_jobs"
}

production_sources() {
  find "${script_dir}/src" "${script_dir}/include" -type f \
    \( -name '*.cc' -o -name '*.cpp' -o -name '*.cu' -o -name '*.h' -o -name '*.hpp' \
       -o -name '*.cuh' \) -print0
}

cpp_sources_from_compile_database() {
  python3 - "$1" "$script_dir" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as database:
    entries = json.load(database)
root = sys.argv[2] + "/src/"
for entry in sorted(entries, key=lambda item: item["file"]):
    path = entry["file"]
    if path.startswith(root) and path.endswith((".cc", ".cpp")):
        sys.stdout.buffer.write(path.encode() + b"\0")
PY
}

cppcheck_project_for_sources() {
  python3 - "$1" "$2" "$script_dir" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as database:
    entries = json.load(database)
root = sys.argv[3] + "/src/"
filtered = [
    entry for entry in entries
    if entry["file"].startswith(root) and entry["file"].endswith((".cc", ".cpp"))
]
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump(filtered, output)
PY
}

cppcheck_arch_define() {
  case "$(uname -m)" in
    x86_64) echo "__x86_64__" ;;
    aarch64) echo "__aarch64__" ;;
    *)
      echo "build.sh: unsupported architecture '$(uname -m)' for cppcheck" >&2
      exit 1
      ;;
  esac
}

cmd_build() {
  build_project "$build_dir" -DBUILD_TESTING=OFF
  require_file "${build_dir}/compile_commands.json"
}

cmd_test() {
  build_project "$build_dir" -DBUILD_TESTING=ON
  if [[ ! -f "${build_dir}/CTestTestfile.cmake" ]]; then
    echo "build.sh: FastTrack has no registered CTest tests; test gate unresolved" >&2
    exit 1
  fi
  require_tool ctest
  ctest --test-dir "$build_dir" --output-on-failure
}

cmd_lint_quick() {
  require_tool clang-format
  require_tool clang-tidy
  require_tool cppcheck
  require_tool python3

  local status=0
  local sources=()
  while IFS= read -r -d '' source; do
    sources+=("$source")
  done < <(production_sources)

  echo "== clang-format (production sources) =="
  if [[ ${#sources[@]} -gt 0 ]]; then
    clang-format --dry-run --Werror "${sources[@]}" || status=1
  fi

  build_project "$lint_quick_build_dir" -DBUILD_TESTING=OFF
  local compile_database="${lint_quick_build_dir}/compile_commands.json"
  require_file "$compile_database"

  echo "== clang-tidy (production C++ sources) =="
  local cpp_sources=()
  while IFS= read -r -d '' source; do
    cpp_sources+=("$source")
  done < <(cpp_sources_from_compile_database "$compile_database")
  if [[ ${#cpp_sources[@]} -gt 0 ]]; then
    clang-tidy -p "$lint_quick_build_dir" "${cpp_sources[@]}" || status=1
  fi

  echo "== cppcheck (production C++ sources) =="
  local cppcheck_project="${lint_quick_build_dir}/cppcheck-project.json"
  cppcheck_project_for_sources "$compile_database" "$cppcheck_project"
  cppcheck --project="$cppcheck_project" --enable=all --inconclusive \
    --error-exitcode=1 -D"$(cppcheck_arch_define)" || status=1
  rm -f "$cppcheck_project"

  exit "$status"
}

cmd_lint() {
  cmd_lint_quick
}

cmd_sanitize() {
  local sanitizer_flags="-fsanitize=address,undefined -fno-omit-frame-pointer"
  build_project "$sanitize_build_dir" -DBUILD_TESTING=ON \
    -DCMAKE_C_FLAGS="$sanitizer_flags" \
    -DCMAKE_CXX_FLAGS="$sanitizer_flags" \
    -DCMAKE_EXE_LINKER_FLAGS="$sanitizer_flags" \
    -DCMAKE_SHARED_LINKER_FLAGS="$sanitizer_flags"
  if [[ ! -f "${sanitize_build_dir}/CTestTestfile.cmake" ]]; then
    echo "build.sh: FastTrack has no registered CTest tests; sanitize gate unresolved" >&2
    exit 1
  fi
  ASAN_OPTIONS="detect_leaks=0:halt_on_error=1" \
    UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
    ctest --test-dir "$sanitize_build_dir" --output-on-failure
}

cmd_tsan() {
  local sanitizer_flags="-fsanitize=thread -fno-omit-frame-pointer"
  build_project "$tsan_build_dir" -DBUILD_TESTING=ON \
    -DCMAKE_C_FLAGS="$sanitizer_flags" \
    -DCMAKE_CXX_FLAGS="$sanitizer_flags" \
    -DCMAKE_EXE_LINKER_FLAGS="$sanitizer_flags" \
    -DCMAKE_SHARED_LINKER_FLAGS="$sanitizer_flags"
  if [[ ! -f "${tsan_build_dir}/CTestTestfile.cmake" ]]; then
    echo "build.sh: FastTrack has no registered CTest tests; TSan gate unresolved" >&2
    exit 1
  fi
  TSAN_OPTIONS="halt_on_error=1:verbosity=0" \
    ctest --test-dir "$tsan_build_dir" --output-on-failure
}

cmd_leak_check() {
  require_tool valgrind
  echo "build.sh: FastTrack has no registered test or benchmark executable for leak-check" >&2
  exit 1
}

cmd_benchmark() {
  build_project "$benchmark_build_dir" -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=ON
  echo "build.sh: FastTrack has no benchmark target; benchmark gate unresolved" >&2
  exit 1
}

cmd_all() {
  echo "== all: build =="
  "${script_dir}/build.sh" build
  echo "== all: test =="
  "${script_dir}/build.sh" test
  echo "== all: lint =="
  "${script_dir}/build.sh" lint
  echo "== all: sanitize =="
  "${script_dir}/build.sh" sanitize
  echo "== all: tsan =="
  "${script_dir}/build.sh" tsan
  echo "== all: leak-check =="
  "${script_dir}/build.sh" leak-check
  echo "== all: benchmark =="
  "${script_dir}/build.sh" benchmark
  echo "== all: PASSED =="
}

cmd_clean() {
  local dirs=(
    "$build_dir" "$sanitize_build_dir" "$tsan_build_dir"
    "$leak_build_dir" "$benchmark_build_dir" "$lint_quick_build_dir"
  )
  for directory in "${dirs[@]}"; do
    if [[ "$directory" != "$script_dir"/* ]]; then
      echo "build.sh: refusing to clean path outside package root: $directory" >&2
      exit 1
    fi
    if [[ -d "$directory" ]]; then
      rm -rf "$directory"
    fi
  done
}

mode="${1:-build}"
case "$mode" in
  build) cmd_build ;;
  test) cmd_test ;;
  lint-quick) cmd_lint_quick ;;
  lint) cmd_lint ;;
  sanitize) cmd_sanitize ;;
  tsan) cmd_tsan ;;
  leak-check) cmd_leak_check ;;
  benchmark) cmd_benchmark ;;
  all) cmd_all ;;
  clean) cmd_clean ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 1
    ;;
esac
