#!/bin/bash

PROFILERS=("nsys profile" "proton")
MODES=("cpu_bound" "gpu_bound")
SCRIPTS=("bench_torch.py" "python bench_triton.py")

for profiler in "${PROFILERS[@]}"
do
		for mode in "${MODES[@]}"
		do
				for script in "${SCRIPTS[@]}"
				do
						echo "Running $profiler $script $mode"
						$profiler "$script" "$mode" 
				done
		done
done