#!/bin/bash

PROFILERS=("nsys" "proton")
WORKLOADS=("cpu_bound" "gpu_bound")
KERNELS=("torch" "triton")

for profiler in "${PROFILERS[@]}"
do
  for workload in "${WORKLOADS[@]}"
  do
    for kernel in "${KERNELS[@]}"
    do
      echo "Running $profiler-$kernel-$workload"
      if [ "$profiler" == "nsys" ];
      then
        time nsys profile python microbench.py --workload "$workload" --profiler "$profiler" --kernel "$kernel"
      elif [ "$profiler" == "proton" ];
      then
        if [ "$kernel" == "triton" ]
        then
          time proton -k triton microbench.py --workload "$workload" --profiler "$profiler" --kernel "$kernel"
        else
          time proton microbench.py --workload "$workload" --profiler "$profiler" --kernel "$kernel"
        fi
      fi
    done
  done
done
