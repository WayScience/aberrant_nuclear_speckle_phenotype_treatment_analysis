#!/usr/bin/env bash

set -u

sources=(
  source_1
  source_10
  source_11
  source_13
  source_15
  source_2
  source_3
  source_4
  source_5
  source_6
  source_7
  source_8
  source_9
)

successful_sources=()
failed_sources=()

script_dir="$(dirname "$0")"
download_script="$script_dir/download_and_merge_cpg0016_source_profiles.py"

for source in "${sources[@]}"; do
  printf 'Running %s\n' "$source"

  if python "$download_script" --source "$source" "$@"; then
    successful_sources+=("$source")
    printf 'Completed %s\n' "$source"
  else
    exit_code=$?
    failed_sources+=("$source")
    printf 'Failed %s with exit code %s\n' "$source" "$exit_code"
  fi
done

printf '\nRun summary\n'
printf 'Total sources: %s\n' "${#sources[@]}"
printf 'Successful sources: %s\n' "${#successful_sources[@]}"
for source in "${successful_sources[@]}"; do
  printf '  %s\n' "$source"
done

printf 'Failed sources: %s\n' "${#failed_sources[@]}"
for source in "${failed_sources[@]}"; do
  printf '  %s\n' "$source"
done

if [ "${#failed_sources[@]}" -gt 0 ]; then
  exit 1
fi

exit 0
