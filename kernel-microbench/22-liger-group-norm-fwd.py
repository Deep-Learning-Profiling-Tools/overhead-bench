import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.group_norm import group_norm_forward

"""
the header defined in liger_kernel
def group_norm_forward(X, num_channels, num_groups, W, B, eps):

"""

def simple_benchmark_group_norm_fwd():
    sizes = [2 ** i for i in range(8, 14)]
    device = "cuda"
    for size in sizes:
        batch = size
        num_channels = size // 2
        num_groups = max(1, num_channels // 8)
        seq_len = 32
        shape = (batch, num_channels, seq_len)
        for _ in range(500):
            X = torch.randn(shape, device=device, dtype=torch.float32)
            W = torch.randn(num_channels, device=device, dtype=torch.float32)
            B = torch.randn(num_channels, device=device, dtype=torch.float32)
            eps = 1e-5
            group_norm_forward(X, num_channels, num_groups, W, B, eps)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_group_norm_fwd"):
                simple_benchmark_group_norm_fwd()
        with open("group_norm_fwd_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        import triton.profiler as proton
        proton.start(name="proton_group_norm_fwd", context="shadow", backend="cupti")
        simple_benchmark_group_norm_fwd()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_group_norm_fwd()

if __name__ == "__main__":
    main()

