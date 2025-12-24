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


def run(num_iters=100):
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

        if i == num_iters - 1:
            time0 = time.time()
            proton.deactivate()
            print(f"deactivate time: {time.time() - time0:.4f}")
            time0 = time.time()
            proton.get_data_msgpack(0)
            print(f"get_data time: {time.time() - time0:.4f}")
            proton.clear_data(0)
            proton.activate()

    torch.cuda.synchronize()
    end_time = time.time()

    print(f"cpu time: {end_time - start_time:.4f}")

run()