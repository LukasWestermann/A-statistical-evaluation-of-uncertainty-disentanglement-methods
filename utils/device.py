import torch

def get_device():
    """Automatically select the best available device (fastest GPU or CPU)."""
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")
   
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        # Find GPU with most free memory (often correlates with fastest/least busy)
        best_gpu = 0
        max_free_mem = 0
        for i in range(num_gpus):
            free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            if free_mem > max_free_mem:
                max_free_mem = free_mem
                best_gpu = i
        device = torch.device(f"cuda:{best_gpu}")
        print(f"Found {num_gpus} GPUs. Using GPU {best_gpu}: {torch.cuda.get_device_name(best_gpu)}")
   
    return device