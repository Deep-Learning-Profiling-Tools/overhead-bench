import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import ensure_contiguous

from liger_kernel.ops.geglu import geglu_forward

"""
the header defined in liger_kernel
def geglu_forward(a, b):
"""

def simple_benchmark_geglu_fwd():
    sizes = [2 ** i for i in range(8, 14)]
    device = "cuda"
    for size in sizes:
        batch = size
        hidden = size // 2
        for _ in range(500):
            a = torch.randn((batch, hidden), device=device, dtype=torch.float32)
            b = torch.randn((batch, hidden), device=device, dtype=torch.float32)
            geglu_forward(a, b)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_geglu_fwd"):
                simple_benchmark_geglu_fwd()
        with open("geglu_fwd_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        import triton.profiler as proton
        proton.start(name="proton_geglu_fwd", context="shadow", backend="cupti")
        simple_benchmark_geglu_fwd()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_geglu_fwd()

if __name__ == "__main__":
    main()
