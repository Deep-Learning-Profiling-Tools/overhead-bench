import argparse
import triton.profiler as proton

from typing import Optional

import torch

from liger_kernel.ops.grpo_loss import GrpoLossFunction

"""
the header defined in liger_kernel
class GrpoLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
        inplace,
    ):
"""

def simple_benchmark_grpo_loss():
    sizes = [2 ** i for i in range(8, 14)]
    device = "cuda"
    vocab_size = 32000
    for size in sizes:
        batch = size
        seq_len = size // 2
        for _ in range(5):
            logits = torch.randn((batch, seq_len, vocab_size), device=device, dtype=torch.float32, requires_grad=True)
            old_logp = torch.randn((batch, seq_len), device=device, dtype=torch.float32)
            ref_logp = torch.randn((batch, seq_len), device=device, dtype=torch.float32)
            completion_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device, dtype=torch.int64)
            advantages = torch.randn((batch, seq_len), device=device, dtype=torch.float32)
            completion_mask = torch.randint(0, 2, (batch, seq_len), device=device, dtype=torch.int32)
            temperature = 1.0
            beta = 1.0
            eps_low = 1e-6
            eps_high = 1e-2
            inplace = False
            # Call the kernel
            GrpoLossFunction.apply(
                logits,
                old_logp,
                ref_logp,
                completion_ids,
                advantages,
                completion_mask,
                temperature,
                beta,
                eps_low,
                eps_high,
                inplace,
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_grpo_loss"):
                simple_benchmark_grpo_loss()
        with open("grpo_loss_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        proton.start(name="proton_grpo_loss", context="shadow", backend="cupti")
        simple_benchmark_grpo_loss()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_grpo_loss()

if __name__ == "__main__":
    main()

