from typing import Literal

import torch
import triton
import triton.language as tl
import triton.profiler as proton

from liger_kernel.ops.kl_div import kldiv_backward_triton

"""
the header defined in liger_kernel
def kldiv_backward_triton(target, grad_output, new_grads, log_target):

"""

def simple_benchmark_kldiv_bwd():
    sizes = [2 ** i for i in range(8, 14)]
    device = "cuda"
    for size in sizes:
        batch = size
        vocab_size = size // 2
        for _ in range(500):  
            target = torch.randn((batch, vocab_size), device=device, dtype=torch.float32)
            grad_output = torch.randn((batch, vocab_size), device=device, dtype=torch.float32)
            new_grads = torch.empty((batch, vocab_size), device=device, dtype=torch.float32)
            log_target = False
            kldiv_backward_triton(target, grad_output, new_grads, log_target)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_kldiv_bwd"):
                simple_benchmark_kldiv_bwd()
        with open("kldiv_bwd_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        import triton.profiler as proton
        proton.start(name="proton_kldiv_bwd", context="shadow", backend="cupti")
        simple_benchmark_kldiv_bwd()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_kldiv_bwd()

if __name__ == "__main__":
    main()
