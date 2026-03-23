import torch
import time
import os
from copy import deepcopy

def profile(model_module, datamodule):
    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")

    results = {}
    model = model_module.model
    if datamodule.batch_size != 1:
        raise ValueError("Batch size is not 1, the results will be unrealistic!")

    total_params = sum(p.numel() for p in model.parameters())
    
    temp_path = "temp_model.pth"
    torch.save(model.state_dict(), temp_path)
    file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)

    print(f"--- Model profile ---")
    print(f"Total parameters: {total_params:,}")
    print(f"Checkpoint size: {file_size_mb:.2f} MB")

    datamodule.setup(stage="test")
    batch = next(iter(datamodule.test_dataloader()))
    inputs, _ = batch
    batch_size = inputs.size(0)

    def measure_inference(target_device):
        m = deepcopy(model).to(target_device)
        m.eval()
        x = inputs.to(target_device)
        
        # WARMUP for hot cache!!
        with torch.no_grad():
            for _ in range(10): _ = m(x)
        
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(50): _ = m(x)
        end = time.perf_counter()
        
        avg_time = (end - start) / 50
        return avg_time

    inf_cpu = measure_inference(device_cpu)
    print(f"\n--- Inference time (CPU): {inf_cpu:.4f} s")
    inf_gpu = measure_inference(device_gpu)
    print(f"Inference time (GPU): {inf_gpu:.4f} s")
    
    torch.cuda.reset_peak_memory_stats()
    model_gpu = deepcopy(model).to(device_gpu)
    model_gpu.train()
    
    optimizer = torch.optim.Adam(model_gpu.parameters())
    x, y = inputs.to(device_gpu), torch.zeros(batch_size, dtype=torch.long).to(device_gpu)
    
    outputs = model_gpu(x)
    loss = torch.nn.functional.cross_entropy(outputs, y)
    loss.backward()
    optimizer.step()
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"Peak VRAM use: {peak_mem:.2f} MB")
