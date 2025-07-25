import torch

if torch.cuda.is_available():
    print("CUDA is available!\n")
    print(f"Number of GPUs: {torch.cuda.device_count()}\n")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} Name: {torch.cuda.get_device_name(i)}\n")
        print(f"GPU {i} Capability: {torch.cuda.get_device_capability(i)}\n")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB\n")
else:
    print("CUDA is not available. PyTorch will use CPU.\n")