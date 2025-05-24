import torch
# from topk_details._topk import _topk
# from . import Bitmatrix
import argparse
import triton.profiler as proton
# from triton_kernels.topk import topk
from dataclasses import dataclass
import triton
import triton.language as tl
from triton_kernels.topk import topk
# @dataclass
# class Bitmatrix:
#     data: "torch.Tensor"  # noqa: F821
#     shape: tuple[int]



# @triton.jit
# def streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
#                    BLOCK_N: tl.constexpr):
#     x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
#     x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
#     x_ultype: tl.constexpr = tl.dtype(f"uint{2*x_nbits}")
#     x_dbtype: tl.constexpr = tl.dtype(f"fp{2*x_nbits}")

#     # subtract 1 from loop iterations because we peel the first (masked) iteration:
#     loop_iterations: tl.constexpr = N_EXPTS_PAD // BLOCK_N - 1

#     offs_x_n = loop_iterations * BLOCK_N + tl.arange(0, BLOCK_N)
#     mask_n = offs_x_n[None, :] < n_expts_tot

#     # first iteration:
#     X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
#     x = tl.load(X_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
#     x = (x.to(x_utype, bitcast=True).to(x_ultype) << x_nbits) | offs_x_n[None, :]
#     x = x.to(x_dbtype, bitcast=True)

#     acc = tl.topk(x, N_EXPTS_ACT, dim=1)

#     # subsequent iterations:
#     for _i in range(loop_iterations):
#         acc = tl.bitonic_merge(acc)  # ensure sorted ascending for the merge
#         X_ptrs -= BLOCK_N
#         offs_x_n -= BLOCK_N
#         x = tl.load(X_ptrs, mask=mask_m, other=float("-inf"))
#         x = (x.to(x_utype, bitcast=True).to(x_ultype) << x_nbits) | offs_x_n[None, :]
#         x = x.to(x_dbtype, bitcast=True)
#         acc = tl.maximum(acc, tl.topk(x, N_EXPTS_ACT, dim=1))

#     return acc


# @triton.jit
# def _topk(X, stride_xm,  # inputs
#           Yv, Yi, stride_ym,  # topk values/indices
#           Bits, stride_rm, n_rows,  # bitmatrix
#           n_expts_tot, BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
#           BLOCK_N: tl.constexpr):

#     tl.static_assert(BLOCK_N % 32 == 0)
#     tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)
#     x_dtype: tl.constexpr = X.dtype.element_ty
#     x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
#     x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
#     x_ultype: tl.constexpr = tl.dtype(f"uint{2*x_nbits}")

#     # load logits
#     offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
#     mask_m = offs_m[:, None] < n_rows
#     y = streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m, N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N)
#     y = y.to(x_ultype, bitcast=True)

#     # sort result in direction of ascending expert index
#     y = (y << x_nbits) | (y >> x_nbits)
#     y = tl.sort(y, dim=1)
#     y_indices = y >> x_nbits
#     y_values = (y & ((1 << x_nbits) - 1)).to(x_utype).to(x_dtype, bitcast=True)
#     y_values = tl.softmax(y_values.to(tl.float32), dim=1, keep_dims=True).to(x_dtype)

#     # write back
#     offs_y_n = tl.arange(0, N_EXPTS_ACT)
#     Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
#     Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
#     tl.store(Yv_ptrs, y_values, mask=mask_m)
#     tl.store(Yi_ptrs, y_indices, mask=mask_m)

#     # pack into bitmatrix
#     y_div = y_indices // 32
#     y_rem = y_indices % 32
#     loop_iterations = N_EXPTS_PAD // BLOCK_N
#     for i in range(loop_iterations):
#         offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
#         y2 = tl.where(y_div[:, :, None] == offs_r_n[None, None, :], (1 << y_rem)[:, :, None], 0)
#         r = tl.reduce_or(y2, axis=1)
#         BitsPtrs = Bits + offs_m[:, None] * stride_rm + offs_r_n[None, :]
#         tl.store(BitsPtrs, r, mask=mask_m)

# def topk(x, k, dim=1, return_bitmatrix=True):
#     cdiv = lambda a, b: (a + b - 1) // b
#     BLOCK_M = 8
#     BLOCK_N = 128
#     assert x.ndim == 2
#     assert x.shape[-1] < 32768
#     assert dim == 1
#     assert return_bitmatrix
#     n_rows, n_cols = x.shape
#     dev = x.device
#     n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
#     n_cols_words = n_cols_pad // 32
#     # scratchpad tensors
#     # NOTE: these are not returned
#     y_vals = torch.empty((n_rows, k), dtype=x.dtype, device=dev)
#     y_indx = torch.empty((n_rows, k), dtype=torch.int16, device=dev)
#     bitmatrix = torch.empty((n_rows, n_cols_words), dtype=torch.uint32, device=dev)
#     _topk[(cdiv(n_rows, BLOCK_M), )](
#         x, x.stride(0),  # inputs
#         y_vals, y_indx, y_vals.stride(0),  # output [topk]
#         bitmatrix, bitmatrix.stride(0),  # output [bitmatrix]
#         n_rows, n_cols,  # shapes
#         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  # tunable parameter
#         N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
#     )
#     return y_vals, y_indx, Bitmatrix(bitmatrix, [n_rows, n_cols])



def simple_benchmark_topk():
    sizes = [128 * i for i in range(2, 6)]
    ks = [4, 8, 16]
    device = "cuda"
    for size in sizes:
        for k in ks:
            x = torch.randn((size, size), device=device, dtype=torch.float16)
            for _ in range(500):
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
        # prof.export_chrome_trace("topk_trace.json")
        with open("topk_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
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