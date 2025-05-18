import torch
from .compaction_details._masked_compaction import _masked_compaction
from triton_kernels import Bitmatrix


def compaction(yv, yi, bitmask, sentinel=-1):
    """
    Return compacted copies of *yv* and *yi* based on a per-row bitmask.

    Only the elements whose index appears among the active bits of *bitmask*
    are kept; the rest are replaced by *sentinel*.  Kept elements preserve
    their original left-to-right order.

    Parameters
    ----------
    yv : torch.Tensor, shape (B, K)
        Values tensor.
    yi : torch.Tensor, shape (B, K), dtype torch.long
        Integer indices (0 ≤ index < 32) associated with *yv*.
    bitmask : torch.Tensor, shape (B,) **or** (B, 32)
        Per-row mask of active indices.  See the in-place version for details.
    sentinel : int, default -1
        Value written into dropped positions of the returned tensors.

    Returns
    -------
    (yv_out, yi_out) : Tuple[torch.Tensor, torch.Tensor], each shape (B, K)
        New tensors with the same dtype/device as the inputs.

    """

    n_rows, n_cols = yi.shape
    ret_yv = torch.empty_like(yv)
    ret_yi = torch.empty_like(yi)
    if isinstance(bitmask, Bitmatrix):
        bitmask = bitmask.data

    _masked_compaction[(n_rows, )](
        yv, yi, bitmask, bitmask.stride(0),  # inputs
        ret_yv, ret_yi,  # outputs
        sentinel,  # sentinel
        K=n_cols  # constants
    )
    return ret_yv, ret_yi


def compaction_torch(yv: torch.Tensor, yi: torch.Tensor, bitmask: torch.Tensor, sentinel=-1):
    """
    reference implementation of `masked_compact`
    """
    B, K = yi.shape
    device = yi.device
    # Expand bitmask to a boolean matrix of active bits  (B, 32)
    w = (1 << torch.arange(32, device=device, dtype=bitmask.dtype))
    bits = (bitmask.unsqueeze(-1) & w) != 0
    mask = bits.flatten(start_dim=-2)  # or bits.reshape(B, -1)
    # For every yi element decide whether it should be kept
    keep = mask.gather(1, yi.long())
    # Build a stable permutation that brings all "keep" items forward
    #    False→0, True→1  ==> invert so kept==0, dropped==1, then argsort
    order = (~keep).to(torch.int).argsort(dim=1, stable=True)
    # Re‑order tensors according to above permutation
    yi_sorted = yi.gather(1, order)
    yv_sorted = yv.gather(1, order)
    # fill relevant positions with sentinel
    keep_sorted = keep.gather(1, order)
    yi_sorted[~keep_sorted] = sentinel
    yv_sorted[~keep_sorted] = sentinel
    return yv_sorted, yi_sorted



def test_compaction(n_tokens, n_cols, k, p, device):
    yi = torch.rand((n_tokens, n_cols), device=device).argsort(dim=-1)
    yi = yi[:, :k].to(torch.int32)
    yv = torch.randn((n_tokens, k), dtype=torch.bfloat16, device=device)
    # "drop" indices from yi with probability `p`
    mask = torch.zeros((n_tokens, n_cols), dtype=torch.int32, device=device)
    keep = (torch.rand(yi.shape, device=device) < p)
    if keep.any():
        rows = torch.arange(yi.size(0), device=device).unsqueeze(1).expand_as(yi)
        mask[rows[keep], yi[keep]] = 1
    chunks = mask.view(*mask.shape[:-1], -1, 32)
    weights = (1 << torch.arange(32, dtype=torch.int32, device=device))
    bitmask = (chunks.int() * weights).sum(dim=-1)
    yv_ref, yi_ref = compaction_torch(yv, yi, bitmask)
    yv_tri, yi_tri = compaction(yv, yi, bitmask)



def profile_compaction(profile_torch=False, device="cuda"):
    import triton.profiler as proton
    import torch.profiler

    compaction_args = [    
        (8192, 64, 4, 0.5),
        (8192, 64, 4, 1.0),
        (4096, 64, 4, 1.0),
        (4096, 64, 4, 0.5),
        (4096, 64, 4, 0.0),
        (1024, 64, 4, 1.0),
        (1024, 64, 4, 0.5),
        (1024, 128, 4, 0.0),
        (131, 128, 16, 0.6),
        (496, 128, 16, 0.),
    ]
    for _ in range(10):
        for n_tokens, n_cols, k, p in compaction_args:
            if profile_torch:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                ) as prof:
                    test_compaction(n_tokens, n_cols, k, p, device)
                prof.export_chrome_trace("compaction_trace.json")
            else:
                with proton.scope("compaction"):
                    test_compaction(n_tokens, n_cols, k, p, device)