import torch

from triton_kernels.routing import routing

def init_data(n_tokens, n_expts_tot, dtype=torch.float16, device="cuda"):
    # the reference implementation and the triton implementation do not tie-break experts the same way
    randbits = [torch.randperm(n_expts_tot) for _ in range(n_tokens)]
    x = [(-1)**i * ((16384 + ((i * 512) % 4096) + bits).to(torch.int16).view(dtype)) for i, bits in enumerate(randbits)]
    return torch.stack(x).to(device=device)


def simple_benchmark_routing(device="cuda"):
    n_tokens = 8192
    block_m = 128
    n_expts_tot, n_expts_act = 128, 4
    tri_logits = init_data(n_tokens, n_expts_tot, device=device).detach()
    for i in range(500):
        torch.manual_seed(i)
        tri_routing_data, tri_gather, tri_scatter = routing(tri_logits, n_expts_act)
        # tri_metadata = compute_metadata(tri_routing_data, n_tokens * n_expts_act, block_m)  # noqa: F841


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", type=str, default="")
    args = parser.parse_args()
    if args.profiler == "torch":
        print("Profiling with torch")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("benchmark_routing"):
                simple_benchmark_routing()
        with open("routing_trace.json", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10000).__str__())
    elif args.profiler == "proton":
        print("Profiling with proton")
        import triton.profiler as proton
        proton.start(name="proton_routing", context="shadow", backend="cupti")
        simple_benchmark_routing()
        proton.finalize()
    else:
        print("Profiling with nsys (no-op fallback)")
        simple_benchmark_routing()

if __name__ == "__main__":
    main()
