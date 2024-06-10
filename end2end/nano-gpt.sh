#!/bin/bash

cd nanoGPT || exit
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
      profiler_cmd=""
      workload_cmd=""
      kernel_cmd=""
      
      if [ "$profiler" == "nsys" ]; then
        profiler_cmd="time nsys profile python"
      elif [ "$profiler" == "proton" ]; then
        if [ "$kernel" == "torch" ]; then
          profiler_cmd="time proton"  
        elif [ "$kernel" == "triton" ]; then
          profiler_cmd="time proton -k triton"
        fi
      fi

      if [ "$workload" == "train" ]; then
        workload_cmd="train.py config/train_shakespeare_char.py --max_iters=1000"
      elif [ "$workload" == "sample" ]; then
        workload_cmd="sample.py --out_dir=out-shakespeare-char"
      fi

      if [ "$kernel" == "torch" ]; then
        kernel_cmd="--compile=False"
      elif [ "$kernel" == "triton" ]; then
        kernel_cmd="--compile=True"
      fi

      cmd="$profiler_cmd $workload_cmd $kernel_cmd"
      echo "-------------------------------------------"
      echo "$cmd"
      echo "-------------------------------------------"
      eval "$cmd"
    done
  done
done

rm -rf *.nsys-rep
rm -rf *.hatchet
