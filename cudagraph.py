import argparse
import time
from typing import NamedTuple

import torch
import triton
import triton.language as tl
import triton.profiler as proton


def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
    vec = torch.arange(512, device="cuda")
    return {"name": "dot", "flops": 512 * 512 * 2, "vec": vec}


@triton.jit(launch_metadata=metadata_fn)
def triton_dot(a_ptr, b_ptr, c_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offs[:, None] * BLOCK_SIZE + offs[None, :])
    b = tl.load(b_ptr + offs[:, None] * BLOCK_SIZE + offs[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs[:, None] * BLOCK_SIZE + offs[None, :], c)


def fn(num_scopes: int) -> None:
    device = "cuda"
    for i in range(num_scopes):
        with proton.scope(f"scope_{i}"):
            x = torch.randn(512, 512, device=device)
            y = torch.randn(512, 512, device=device)
            z = torch.relu(x @ y)
            triton_dot[(1,)](x, y, z, 512)


def run(
    *,
    mode: str,
    num_iters: int = 100,
    num_scopes: int = 5000,
    advance_every: int = 10,
) -> None:
    enable_profiling = mode != "none"
    enable_triton_hooks = mode == "profile_triton"

    # warmup (avoid capturing/profiling compilation)
    fn(num_scopes)

    session = None
    if enable_profiling:
        start_mode = "periodic_flushing:format=hatchet_msgpack"
        if enable_triton_hooks:
            session = proton.start("test", mode=start_mode, hook="triton")
        else:
            session = proton.start("test", mode=start_mode)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(num_scopes)

    start_time = time.time()
    for i in range(num_iters):
        g.replay()
        if enable_profiling and i != 0 and i % advance_every == 0:
            proton.data.advance_phase(session)

    torch.cuda.synchronize()
    if enable_profiling:
        proton.finalize()
    end_time = time.time()

    print(f"cpu time: {end_time - start_time:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["none", "profile", "profile_triton"],
        default="none",
        help=(
            "1) none: no profile. "
            "2) profile: periodic_flushing. "
            "3) profile_triton: periodic_flushing + Triton hooks."
        ),
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--scopes", type=int, default=5000)
    parser.add_argument("--advance-every", type=int, default=10)
    args = parser.parse_args()

    run(
        mode=args.mode,
        num_iters=args.iters,
        num_scopes=args.scopes,
        advance_every=args.advance_every,
    )


if __name__ == "__main__":
    main()
