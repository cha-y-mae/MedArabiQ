import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
x = torch.randn(100, 100).cuda()
print(f"GPU memory after tensor: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")