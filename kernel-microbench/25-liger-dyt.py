import torch
import triton
import triton.language as tl

from liger_kernel.ops.dyt import liger_dyt_fwd

"""
the header defined in liger_kernel
def liger_dyt_fwd(x, alpha, gamma, beta):

"""

def simple_benchmark_liger_dyt_fwd():
    sizes = [2 ** i for i in range(8, 14)]
    device = "cuda"
    for size in sizes:
        batch = size
        dim = size // 2
        for _ in range(500):  
            x = torch.randn((batch, dim), device=device, dtype=torch.float32)
            alpha = torch.randn(dim, device=device, dtype=torch.float32)
            gamma = torch.randn(dim, device=device, dtype=torch.float32)
            beta = torch.randn(dim, device=device, dtype=torch.float32)
            liger_dyt_fwd(x, alpha, gamma, beta)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_liger_dyt_fwd"):
                simple_benchmark_liger_dyt_fwd()
        with open("liger_dyt_fwd_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        import triton.profiler as proton
        proton.start(name="proton_liger_dyt_fwd", context="shadow", backend="cupti")
        simple_benchmark_liger_dyt_fwd()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_liger_dyt_fwd()

if __name__ == "__main__":
    main()
