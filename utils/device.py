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

def get_device_for_worker(worker_id):
    """
    Get device for a specific worker in parallel execution.
    Uses round-robin assignment across available GPUs.
    
    Args:
        worker_id: Integer ID of the worker (0-indexed)
    
    Returns:
        torch.device: Device assigned to this worker
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    num_gpus = torch.cuda.device_count()
    gpu_id = worker_id % num_gpus
    return torch.device(f"cuda:{gpu_id}")

def get_num_gpus():
    """Get number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0