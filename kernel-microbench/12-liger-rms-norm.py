import argparse
import triton.profiler as proton

from typing import Optional

import torch

from liger_kernel.ops.rms_norm import rms_norm_forward
"""
the header defined in liger_kernel
def rms_norm_forward(X, W, eps, offset, casting_mode):
"""

def simple_benchmark_rms_norm():
    sizes = [2 ** i for i in range(10, 16)]
    device = "cuda"
    for size in sizes:
        batch = size
        num_features = size // 2
        for _ in range(500):
            X = torch.rand((batch, num_features), device=device, dtype=torch.float32)
            W = torch.rand((num_features,), device=device, dtype=torch.float32)
            eps = 1e-5
            offset = 0  # default offset
            casting_mode = 0  # default casting mode, update if needed
            rms_norm_forward(X, W, eps, offset, casting_mode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_rms_norm"):
                simple_benchmark_rms_norm()
        with open("rms_norm_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        proton.start(name="proton_rms_norm", context="shadow", backend="cupti")
        simple_benchmark_rms_norm()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_rms_norm()

if __name__ == "__main__":
    main()

