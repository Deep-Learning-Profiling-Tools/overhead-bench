from typing import Literal

import torch
import triton
import triton.language as tl
import triton.profiler as proton

from liger_kernel.ops.kl_div import kldiv_forward_triton

"""
the header defined in liger_kernel
def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):  # [BT, V]

"""

def simple_benchmark_kldiv():
    sizes = [2 ** i for i in range(8, 14)]
    device = "cuda"
    for size in sizes:
        batch = size
        vocab_size = size // 2
        for _ in range(500):
            y_pred = torch.randn((batch, vocab_size), device=device, dtype=torch.float32)
            y_true = torch.randn((batch, vocab_size), device=device, dtype=torch.float32)
            log_target = False
            reduction = 1  # 0: none, 1: mean, 2: sum
            eps = 1e-6
            kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_kldiv"):
                simple_benchmark_kldiv()
        with open("kldiv_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        proton.start(name="proton_kldiv", context="shadow", backend="cupti")
        simple_benchmark_kldiv()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_kldiv()

if __name__ == "__main__":
    main()

