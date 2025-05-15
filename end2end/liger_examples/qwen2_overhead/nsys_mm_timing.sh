#!/usr/bin/env bash
./profile_wrapper nsys profile --trace=cuda --sample=none --cpuctxsw=none python training.py
#
# set -euo pipefail
# 
# # Tell bash's built-in time to output only the real time in seconds
# export TIMEFORMAT='%R'
# 
# echo "Running 5 trials for nsys mm; printing elapsed (real) time for each:"
# for i in {1..5}; do
#   echo "Trial nsys $i:"
#   elapsed=$( { time nsys profile --trace=cuda --sample=none --cpuctxsw=none python training.py > /dev/null; } 2>&1 | tail -n 1 )
#   echo "Trial $i: ${elapsed}s"
# done
