import torch
import triton
import triton.language as tl

from liger_kernel.ops.sparsemax import LigerSparsemaxFunction

"""
the header defined in liger_kernel
class LigerSparsemaxFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, dim: int):
"""

def simple_benchmark_sparsemax_fwd():
    sizes = [2 ** i for i in range(8, 14)]
    device = "cuda"
    for size in sizes:
        batch = size
        dim = size // 2
        for _ in range(500):  
            x = torch.randn((batch, dim), device=device, dtype=torch.float32)
            LigerSparsemaxFunction.apply(x, 1)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_sparsemax_fwd"):
                simple_benchmark_sparsemax_fwd()
        with open("sparsemax_fwd_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        import triton.profiler as proton
        proton.start(name="proton_sparsemax_fwd", context="shadow", backend="cupti")
        simple_benchmark_sparsemax_fwd()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_sparsemax_fwd()

if __name__ == "__main__":
    main()
