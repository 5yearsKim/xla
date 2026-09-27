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

if ! command -v clang-format >/dev/null 2>&1; then
  printf 'Required tool not found: clang-format\n' >&2
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
  clang-format -i "${format_files[@]/#/${root_dir}/}"
else
  printf 'Checking clang-format on %d files (excluding egg-c)...\n' "${#format_files[@]}"
  clang-format --dry-run --Werror "${format_files[@]/#/${root_dir}/}"
fi
