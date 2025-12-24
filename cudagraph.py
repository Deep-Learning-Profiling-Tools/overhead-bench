import torch 
import triton 
import triton.profiler as proton 
import time


def fn(num_scopes=5000): 
    device = "cuda" 
    for i in range(num_scopes):
        with proton.scope(f"scope_{i}"):
            x = torch.randn(512, 512, device=device) 
            y = torch.randn(512, 512, device=device) 
            z = torch.relu(x @ y) 


def run(profiling=False, num_iters=100):
    if profiling:
        session = proton.start(f"profile") 

    # warmup
    fn()

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    start_time = time.time()

    for i in range(num_iters): 
        g.replay()

        if profiling and i % num_iters == num_iters - 1:
            time0 = time.time()
            proton.deactivate(session)
            print(f"deactivate time: {time.time() - time0:.4f}")
            time0 = time.time()
            proton.get_data_msgpack(session)
            print(f"get_data time: {time.time() - time0:.4f}")
            proton.clear_data(session)
            proton.activate(session)

    torch.cuda.synchronize()
    end_time = time.time()

    if profiling:
        print(f"profiling cpu time: {end_time - start_time:.4f}")
        proton.finalize()
    else:
        print(f"pure cpu time: {end_time - start_time:.4f}")

run(profiling=False)
run(profiling=True)
