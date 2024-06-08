import torch
import time
import sys
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def add(a_ptr, b_ptr, c_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + pid * BLOCK_SIZE + offs)
    b = tl.load(b_ptr + pid * BLOCK_SIZE + offs)
    c = a + b
    tl.store(c_ptr + pid * BLOCK_SIZE + offs, c)


def run(nelems, iters):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor_a = torch.randn(nelems, dtype=torch.float32, device=device)
    tensor_b = torch.randn(nelems, dtype=torch.float32, device=device)

    result_gpu = torch.empty_like(tensor_a)

    # warmup
    add[(nelems // BLOCK_SIZE, )](tensor_a, tensor_b, result_gpu, BLOCK_SIZE)

    start_time = time.time()
    # measure
    for _ in range(iters):
        add[(nelems // BLOCK_SIZE, )](tensor_a,
                                      tensor_b, result_gpu, BLOCK_SIZE)
    end_time = time.time()

    print("cpu time", end_time - start_time)

    torch.cuda.synchronize()


if __name__ == "__main__":
    workload = sys.argv[1]
    if workload == "cpu_bound":
        run(nelems=BLOCK_SIZE, iters=100000)
    elif workload == "gpu_bound":
        run(nelems=BLOCK_SIZE * 100000, iters=10000)
