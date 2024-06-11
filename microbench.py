import torch
import time
import argparse
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def triton_dot(a_ptr, b_ptr, c_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + pid * BLOCK_SIZE + offs)
    b = tl.load(b_ptr + pid * BLOCK_SIZE + offs)
    a = tl.broadcast(a[:, None], [BLOCK_SIZE, BLOCK_SIZE])
    b = tl.broadcast(b[None, :], [BLOCK_SIZE, BLOCK_SIZE])
    c = tl.sum(tl.dot(a, b), axis=1)
    tl.store(c_ptr + pid * BLOCK_SIZE + offs, c)


@triton.jit
def triton_add(a_ptr, b_ptr, c_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + pid * BLOCK_SIZE + offs)
    b = tl.load(b_ptr + pid * BLOCK_SIZE + offs)
    c = a + b
    tl.store(c_ptr + pid * BLOCK_SIZE + offs, c)


def run(nelems, iters, kernel, workload):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor_a = torch.randn(nelems, dtype=torch.float32, device=device)
    tensor_b = torch.randn(nelems, dtype=torch.float32, device=device)

    def add():
        if kernel == "triton":
            result_gpu = torch.empty_like(tensor_a)
            if workload == "cpu_bound":
                triton_add[(nelems // BLOCK_SIZE, )](tensor_a, tensor_b,
                                                     result_gpu, BLOCK_SIZE)
            else:
                triton_dot[(nelems // BLOCK_SIZE, )](tensor_a, tensor_b,
                                                     result_gpu, BLOCK_SIZE)
        elif kernel == "torch":
            if workload == "cpu_bound":
                result_gpu = tensor_a + tensor_b
            else:
                result_gpu = torch.matmul(
                    tensor_a[:, None], tensor_b[None, :]).sum(dim=1)

    # warmup
    add()

    start_time = time.time()
    # measure
    for _ in range(iters):
        add()
    end_time = time.time()

    torch.cuda.synchronize()

    print("cpu time:", end_time - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str,
                        choices=["torch", "triton"], required=True)
    parser.add_argument("--workload", type=str,
                        choices=["cpu_bound", "gpu_bound"], required=True)
    parser.add_argument("--profiler", type=str,
                        choices=["nsys", "proton", "kineto", "none"], default="none")
    args = parser.parse_args()
    if args.workload == "cpu_bound":
        run(nelems=BLOCK_SIZE, iters=100000,
            kernel=args.kernel, workload=args.workload)
    elif args.workload == "gpu_bound":
        run(nelems=BLOCK_SIZE*10, iters=100,
            kernel=args.kernel, workload=args.workload)
