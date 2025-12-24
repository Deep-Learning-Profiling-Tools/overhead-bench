import torch
from triton_kernels.compaction import compaction
from triton_kernels import Bitmatrix


def simple_benchmark_compaction():
    sizes = [(2 ** i, 2**i) for i in range(8, 14)]
    device = "cuda"
    k = 16  # or another reasonable value for k
    p = 0.5  # probability to keep an index
    for n_tokens, n_cols in sizes:
        for _ in range(100):
            yi = torch.rand((n_tokens, n_cols), device=device).argsort(dim=-1)
            yi = yi[:, :k].to(torch.int32)
            yv = torch.randn((n_tokens, k), dtype=torch.bfloat16, device=device)
            mask = torch.zeros((n_tokens, n_cols), dtype=torch.int32, device=device)
            keep = (torch.rand(yi.shape, device=device) < p)
            if keep.any():
                rows = torch.arange(yi.size(0), device=device).unsqueeze(1).expand_as(yi)
                mask[rows[keep], yi[keep]] = 1
            chunks = mask.view(*mask.shape[:-1], -1, 32)
            weights = (1 << torch.arange(32, dtype=torch.int32, device=device))
            bitmask = (chunks.int() * weights).sum(dim=-1)
            compaction(yv, yi, bitmask)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_compaction"):
                simple_benchmark_compaction()
        with open("compaction_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        import triton.profiler as proton
        proton.start(name="proton_compaction", context="shadow", backend="cupti")
        simple_benchmark_compaction()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_compaction()

if __name__ == "__main__":
    main()


