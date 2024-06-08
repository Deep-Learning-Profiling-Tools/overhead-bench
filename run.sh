#!/bin/bash

PROFILERS=("nsys" "proton")
MODES=("cpu_bound" "gpu_bound")
SCRIPTS=("bench_torch.py" "bench_triton.py")

for profiler in "${PROFILERS[@]}"
do
  for mode in "${MODES[@]}"
  do
    for script in "${SCRIPTS[@]}"
    do
      echo "Running $profiler $script $mode"
      if [ "$profiler" == "nsys" ]
      then
        time nsys profile python "$script" "$mode"
      elif [ "$profiler" == "proton" ]
      then
        if [ "$script" == "bench_triton.py" ]
        then
          time proton -k triton "$script" "$mode"
        else
          time proton "$script" "$mode"
        fi
      fi
    done
  done
done
