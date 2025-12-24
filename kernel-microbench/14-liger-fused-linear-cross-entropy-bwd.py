import argparse
import triton.profiler as proton

from typing import Optional

import torch

from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward

"""
the header defined in liger_kernel
def cross_entropy_forward(
    _input,
    target,
    weight,
    ignore_index,
    lse_square_scale,
    label_smoothing,
    reduction,
    softcap,
    return_z_loss,
):
"""

def simple_benchmark_cross_entropy():
    # Example parameter sets for cross_entropy_forward
    sizes = [2 ** i for i in range(8, 14)]
    device = "cuda"
    for size in sizes:
        batch = size
        num_classes = size // 2
        for _ in range(500): 
            _input = torch.randn((batch, num_classes), device=device, dtype=torch.float32, requires_grad=True)
            target = torch.randint(0, num_classes, (batch,), device=device, dtype=torch.int64)
            weight = torch.rand(num_classes, device=device, dtype=torch.float32)
            ignore_index = -100
            lse_square_scale = 0.0
            label_smoothing = 0.0
            reduction = 1  # 0: none, 1: mean, 2: sum (typical torch convention)
            softcap = 0.0
            return_z_loss = False
            # Call the kernel
            fused_linear_cross_entropy_forward(
                _input,
                target,
                weight,
                ignore_index,
                lse_square_scale,
                label_smoothing,
                reduction,
                softcap,
                return_z_loss,
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_cross_entropy"):
                simple_benchmark_cross_entropy()
        with open("cross_entropy_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        proton.start(name="proton_cross_entropy", context="shadow", backend="cupti")
        simple_benchmark_cross_entropy()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_cross_entropy()

if __name__ == "__main__":
    main()

