import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.geglu import geglu_backward

"""
the header defined in liger_kernel
def geglu_backward(a, b, dc):
"""

def simple_benchmark_geglu_bwd():
    sizes = [2 ** i for i in range(8, 14)]
    device = "cuda"
    for size in sizes:
        batch = size
        hidden = size // 2
        for _ in range(500):
            a = torch.randn((batch, hidden), device=device, dtype=torch.float32)
            b = torch.randn((batch, hidden), device=device, dtype=torch.float32)
            dc = torch.randn((batch, hidden), device=device, dtype=torch.float32)
            geglu_backward(a, b, dc)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_geglu_bwd"):
                simple_benchmark_geglu_bwd()
        with open("geglu_bwd_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        import triton.profiler as proton
        proton.start(name="proton_geglu_bwd", context="shadow", backend="cupti")
        simple_benchmark_geglu_bwd()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_geglu_bwd()

if __name__ == "__main__":
    main()
