#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 <num_runs> <wrapped command...>"
  echo "Example: $0 5 nsys python myscript.py arg1"
  exit 1
fi

# Number of times to run
num_runs=$1
shift

# Array to hold elapsed times
declare -a times

for ((i=1; i<=num_runs; i++)); do
  echo
  echo ">>> Run #$i of $num_runs"

  # capture output to a temp log
  logfile=$(mktemp /tmp/profile-log.XXXXXX)

  # run the wrapped command
  start_marker=""
  # start time comes from script’s own START_PROFILE line; end time we record
  "$@" 2>&1 | tee "$logfile"
  exit_code=${PIPESTATUS[0]}

  # get end timestamp
  end_ts=$(date +%s.%N)

  # extract the start timestamp from the first START_PROFILE line
  start_line=$(grep -m1 '^START_PROFILE: ' "$logfile" || true)
  if [[ -z "$start_line" ]]; then
    echo "ERROR: no START_PROFILE found in run #$i" >&2
    rm -f "$logfile"
    exit 1
  fi
  start_ts=${start_line#START_PROFILE:\ }

  # compute elapsed
  elapsed=$(echo "$end_ts - $start_ts" | bc)

  # store
  times+=("$elapsed")

  # report this run
  printf "Run %2d: %s seconds\n" "$i" "$elapsed"

  rm -f "$logfile"

  # if the wrapped command failed, stop early
  if [[ $exit_code -ne 0 ]]; then
    echo "Wrapped command exited with code $exit_code. Aborting."
    exit $exit_code
  fi
done

# Summary: compute min, max, avg via bc
min=${times[0]}
max=${times[0]}
sum=0
for t in "${times[@]}"; do
  # compare floats: use bc
  sleep 5
  is_less=$(echo "$t < $min" | bc)
  (( is_less )) && min=$t
  is_greater=$(echo "$t > $max" | bc)
  (( is_greater )) && max=$t
  sum=$(echo "$sum + $t" | bc)
done

avg=$(echo "$sum / $num_runs" | bc -l)

echo
echo "=== SUMMARY over $num_runs runs ==="
echo "  Min elapsed : $min seconds"
echo "  Max elapsed : $max seconds"
echo "  Avg elapsed : $avg seconds"
echo "==================================="

exit 0

