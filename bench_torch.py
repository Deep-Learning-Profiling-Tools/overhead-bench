import torch
import time
import sys


def run(nelems, iters):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor_a = torch.randn(nelems, dtype=torch.float32, device=device)
    tensor_b = torch.randn(nelems, dtype=torch.float32, device=device)

    result_gpu = torch.empty_like(tensor_a)

    # warmup
    result_gpu.copy_(tensor_a + tensor_b, non_blocking=True)

    start_time = time.time()
    # measure
    for _ in range(iters):
        result_gpu.copy_(tensor_a + tensor_b, non_blocking=True)
    end_time = time.time()

    print("cpu time", end_time - start_time)

    torch.cuda.synchronize()


if __name__ == "__main__":
    workload = sys.argv[1]
    if workload == "cpu_bound":
        run(nelems=1000, iters=100000)
    elif workload == "gpu_bound":
        run(nelems=100000000, iters=10000)
