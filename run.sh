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
						echo "Running $profiler $mode $script"
						./"$profiler" "$mode" "$script"
				done
		done
done