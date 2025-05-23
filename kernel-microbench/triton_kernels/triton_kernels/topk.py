import torch
from .topk_details._topk import _topk
from triton_kernels import Bitmatrix
import argparse
import triton.profiler as proton
from triton_kernels.topk import topk

def topk(x, k, dim=1, return_bitmatrix=True):
    cdiv = lambda a, b: (a + b - 1) // b
    BLOCK_M = 8
    BLOCK_N = 128
    assert x.ndim == 2
    assert x.shape[-1] < 32768
    assert dim == 1
    assert return_bitmatrix
    n_rows, n_cols = x.shape
    dev = x.device
    n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    # scratchpad tensors
    # NOTE: these are not returned
    y_vals = torch.empty((n_rows, k), dtype=x.dtype, device=dev)
    y_indx = torch.empty((n_rows, k), dtype=torch.int16, device=dev)
    bitmatrix = torch.empty((n_rows, n_cols_words), dtype=torch.uint32, device=dev)
    _topk[(cdiv(n_rows, BLOCK_M), )](
        x, x.stride(0),  # inputs
        y_vals, y_indx, y_vals.stride(0),  # output [topk]
        bitmatrix, bitmatrix.stride(0),  # output [bitmatrix]
        n_rows, n_cols,  # shapes
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  # tunable parameter
        N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
    )
    return y_vals, y_indx, Bitmatrix(bitmatrix, [n_rows, n_cols])



def simple_benchmark_topk():
    sizes = [128 * i for i in range(2, 6)]
    ks = [4, 8, 16]
    device = "cuda"
    for size in sizes:
        for k in ks:
            x = torch.randn((size, size), device=device, dtype=torch.float16)
            for _ in range(5):
                # Triton topk
                topk(x, k)
                # Optionally, compare to torch's topk for correctness or reference
                # torch.topk(x, k, dim=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_topk"):
                simple_benchmark_topk()
        prof.export_chrome_trace("topk_trace.json")
    elif args.profiler == "proton":
        print("Profiling with proton")
        proton.start(name="proton_topk", context="shadow", backend="cupti")
        simple_benchmark_topk()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_topk()

if __name__ == "__main__":
    main()