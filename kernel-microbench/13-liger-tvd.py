import argparse
import triton.profiler as proton

from typing import Optional

import torch

from liger_kernel.ops.tvd import tv_distance_forward_triton

"""
the header defined in liger_kernel
def tv_distance_forward_triton(p, q, shift_labels, reduction, ignore_index, has_label)
"""

def simple_benchmark_tvd():
    sizes = [2 ** i for i in range(10, 16)]
    device = "cuda"
    for size in sizes:
        batch = size
        num_classes = size // 2
        for _ in range(500):
            p = torch.rand((batch, num_classes), device=device, dtype=torch.float32)
            q = torch.rand((batch, num_classes), device=device, dtype=torch.float32)
            shift_labels = False
            reduction = 1  # 0: none, 1: mean, 2: sum (typical torch convention)
            ignore_index = -100
            has_label = False
            tv_distance_forward_triton(
                p,
                q,
                shift_labels,
                reduction,
                ignore_index,
                has_label,
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_tvd"):
                simple_benchmark_tvd()
        with open("tvd_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        proton.start(name="proton_tvd", context="shadow", backend="cupti")
        simple_benchmark_tvd()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_tvd()

if __name__ == "__main__":
    main()
