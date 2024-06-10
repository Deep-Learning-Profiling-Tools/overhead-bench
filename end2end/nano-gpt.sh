#!/bin/bash

cd nanoGPT
python data/shakespeare_char/prepare.py

PROFILERS=("nsys" "proton")
WORKLOADS=("train" "sample")
KERNELS=("torch" "triton")

for profiler in "${PROFILERS[@]}"
do
  for workload in "${WORKLOADS[@]}"
  do
    for kernel in "${KERNELS[@]}"
    do
      if [ "$profiler" == "nsys" ];
      then
        if [ "$workload" == "train" ]
          cmd="time nsys profile python train.py config/train_shakespeare_char.py --max_iters=1000"
        then
          cmd="time python sample.py --out_dir=out-shakespeare-char"
        else
        fi
      elif [ "$profiler" == "proton" ];
      then
        if [ "$kernel" == "triton" ]
        then
          cmd="time proton -k triton microbench.py --workload $workload --profiler $profiler --kernel $kernel"
        else
          cmd="time proton microbench.py --workload $workload --profiler $profiler --kernel $kernel"
        fi
      fi
      echo "$cmd"
      eval "$cmd"
    done
  done
done

