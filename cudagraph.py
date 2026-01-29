import torch 
import triton 
import triton.profiler as proton 
import time
from typing import NamedTuple


def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
    vec = torch.arange(512, device="cuda")
    return {"name": f"dot", "flops": 512 * 512 * 2, "vec": vec}


@triton.jit(launch_metadata=metadata_fn)
def triton_dot(a_ptr, b_ptr, c_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offs[:, None] * BLOCK_SIZE + offs[None, :])
    b = tl.load(b_ptr + offs[:, None] * BLOCK_SIZE + offs[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs[:, None] * BLOCK_SIZE + offs[None, :], c)


def fn(num_scopes=5000): 
    device = "cuda" 
    for i in range(num_scopes):
        with proton.scope(f"scope_{i}"):
            x = torch.randn(512, 512, device=device) 
            y = torch.randn(512, 512, device=device) 
            z = torch.relu(x @ y) 
            triton_dot[(1, )](x, y, z, 512)


def run(num_iters=100):
    # warmup
    fn()

    session = proton.start("test", mode="periodic_flushing:format=hatchet_msgpack")

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    start_time = time.time()

    for i in range(num_iters): 
        g.replay()
        if i != 0 and i % 10 == 0:
            proton.data.advance_phase(session)

    torch.cuda.synchronize()
    proton.finalize()
    end_time = time.time()

    print(f"cpu time: {end_time - start_time:.4f}")

run()