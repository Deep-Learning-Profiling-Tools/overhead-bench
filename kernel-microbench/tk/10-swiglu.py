from dataclasses import dataclass, field
from triton_kernels.numerics import InFlexData, OutFlexData
import torch
import triton
from triton_kernels.swiglu import swiglu




def init_data(n_tokens, n_expts_tot, dtype=torch.float16, device="cuda"):
    # the reference implementation and the triton implementation do not tie-break experts the same way
    randbits = [torch.randperm(n_expts_tot) for _ in range(n_tokens)]
    x = [(-1)**i * ((16384 + ((i * 512) % 4096) + bits).to(torch.int16).view(dtype)) for i, bits in enumerate(randbits)]
    return torch.stack(x).to(device=device)


def alloc_rand(shape, device, dtype, requires_grad=True):
    if dtype.itemsize == 1:
        tmp = 2**-(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype).requires_grad_(requires_grad)
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)



@dataclass(frozen=True)
class FlexCtx:
    out_data: OutFlexData = OutFlexData()
    inp_data: InFlexData = InFlexData()
    saturate_inf: bool = False


@dataclass(frozen=True)
class PrecisionConfig:
    limit: float
    flex_ctx: FlexCtx = FlexCtx()


def simple_benchmark_swiglu():
    device = "cuda"
    n_expts_tot = 6
    n_expts_act = 2
    alpha = 0.5
    limit = 10
    sizes = [
        (256, 1024, 256),
        (512, 2048, 512),
        (1024, 4096, 1024),
        (1311, 4352, 1311),
    ]
    for M, N, n_tokens in sizes:
        torch.manual_seed(2)
        logits = init_data(M, n_expts_tot).detach()
        x = alloc_rand([n_tokens, N], device=device, dtype=torch.bfloat16)
        precision_config = PrecisionConfig(limit=limit)
        for _ in range(100):
            swiglu(x, alpha, precision_config)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_swiglu"):
                simple_benchmark_swiglu()
        with open("swiglu_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        import triton.profiler as proton
        proton.start(name="proton_swiglu", context="shadow", backend="cupti")
        simple_benchmark_swiglu()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_swiglu()




if __name__ == "__main__":
    main()


