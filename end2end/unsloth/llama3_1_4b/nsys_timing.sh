#!/usr/bin/env bash
set -euo pipefail

# Tell bash’s built-in time to output only the real time in seconds
export TIMEFORMAT='%R'
ALL_ELAPSED=()
echo "Running 5 trials; printing elapsed (real) time for each:"
for i in {1..5}; do
  # Run the profiling command, suppress its stdout, capture the time output
  elapsed=$( { time nsys profile --trace=cuda --sample=none --cpuctxsw=none python ./llama3_1.py > /dev/null; } 2>&1 )
  # elapsed=$( { time python ./llama3_1.py > /dev/null; } 2>&1 )
  echo "Trial $i: ${elapsed}s"
  ALL_ELAPSED+=($elapsed)
done
  
# Print all elapsed times
echo "All elapsed times: ${ALL_ELAPSED[@]}"
  
# Calculate and print the average elapsed time

