#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="fix"

while (($#)); do
  case "$1" in
    --fix) mode="fix" ;;
    --check) mode="check" ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./lint.sh [--fix | --check]

By default, applies clang-format to C and C++ source and header files under
research/joint_shard, excluding egg-c. --check checks formatting without
modifying files.

Uses the cached Bazel LLVM 18 formatter when available, otherwise a compatible
formatter on PATH. Set CLANG_FORMAT to select an executable explicitly.
The parent XLA formatting configuration requires clang-format 16 or newer.
USAGE
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      printf 'Run ./lint.sh --help for usage.\n' >&2
      exit 2
      ;;
  esac
  shift
done

if ! command -v rg >/dev/null 2>&1; then
  printf 'Required tool not found: rg\n' >&2
  exit 127
fi

# Prefer the project's toolchain over an older system formatter. Discover it
# through Bazel's workspace symlinks without starting Bazel or downloading tools.
workspace_dir="$(cd -- "$root_dir/../.." && pwd)"
if [[ -n "${CLANG_FORMAT:-}" ]]; then
  candidates=("$CLANG_FORMAT")
else
  candidates=(
    "$workspace_dir"/bazel-*/external/*llvm18*/bin/clang-format
    clang-format-18 clang-format-17 clang-format-16 clang-format
  )
fi
formatter=""
for candidate in "${candidates[@]}"; do
  if ! command -v "$candidate" >/dev/null 2>&1; then
    continue
  fi
  if version="$("$candidate" --version 2>/dev/null)" &&
      [[ "$version" =~ version[[:space:]]+([0-9]+) ]] &&
      ((BASH_REMATCH[1] >= 16)); then
    formatter="$candidate"
    break
  fi
done
if [[ -z "$formatter" ]]; then
  printf 'A working clang-format 16 or newer is required by the XLA formatting configuration.\n' >&2
  printf 'Install clang-format-18, build the Bazel LLVM toolchain, or set CLANG_FORMAT to a compatible executable.\n' >&2
  exit 127
fi

mapfile -t format_files < <(
  cd "$root_dir"
  rg --files \
    -g '*.c' -g '*.cc' -g '*.cpp' -g '*.cxx' -g '*.h' -g '*.hh' -g '*.hpp' -g '*.hxx' \
    -g '!egg-c/**' | sort
)

if ((${#format_files[@]} == 0)); then
  printf 'No C or C++ source/header files found outside egg-c.\n' >&2
  exit 1
fi

if [[ "$mode" == "fix" ]]; then
  printf 'Applying clang-format to %d files (excluding egg-c)...\n' "${#format_files[@]}"
  "$formatter" -i "${format_files[@]/#/${root_dir}/}"
else
  printf 'Checking clang-format on %d files (excluding egg-c)...\n' "${#format_files[@]}"
  "$formatter" --dry-run --Werror "${format_files[@]/#/${root_dir}/}"
fi
