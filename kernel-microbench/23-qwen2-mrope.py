import torch
import triton
import triton.language as tl

from liger_kernel.ops.qwen2vl_mrope import qwen2vl_mrope_forward

"""
the header defined in liger_kernel
def qwen2vl_mrope_forward(q, k, cos, sin, mrope_section):

"""

def simple_benchmark_qwen2_mrope():
    sizes = [2 ** i for i in range(7, 11)]
    device = "cuda"
    for size in sizes:
        batch = size
        seq_len = size
        n_q_head = 16
        n_kv_head = 8
        head_dim = 128
        q = torch.randn((batch, seq_len, n_q_head, head_dim), device=device, dtype=torch.float32)
        k = torch.randn((batch, seq_len, n_kv_head, head_dim), device=device, dtype=torch.float32)
        cos = torch.randn((seq_len, head_dim), device=device, dtype=torch.float32)
        sin = torch.randn((seq_len, head_dim), device=device, dtype=torch.float32)
        mrope_section = (0, seq_len)
        for _ in range(500):
            qwen2vl_mrope_forward(q, k, cos, sin, mrope_section)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_qwen2_mrope"):
                simple_benchmark_qwen2_mrope()
        with open("qwen2_mrope_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        import triton.profiler as proton
        proton.start(name="proton_qwen2_mrope", context="shadow", backend="cupti")
        simple_benchmark_qwen2_mrope()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_qwen2_mrope()

if __name__ == "__main__":
    main()
