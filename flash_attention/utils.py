import torch
from config import ExperimentConfig
from torch.utils.data import DataLoader

class MemoryTracker:
    """Track GPU memory usage throughout training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    def get_memory_stats(self):
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
            'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
        }
    
    def print_memory_summary(self):
        stats = self.get_memory_stats()
        if stats:
            print("\n=== GPU Memory Summary ===")
            for key, value in stats.items():
                print(f"{key}: {value:.2f} MB")
            print("=" * 26)

def find_max_batch_size(config: ExperimentConfig, model_fn, dataset, 
                        start_bs=1, max_bs=128):
    """Binary search to find maximum batch size that fits in memory"""

    print(f"\nFinding maximum batch size for {config}...")
    def try_batch_size(bs):
        try:
            test_config = ExperimentConfig(**vars(config))
            test_config.batch_size = bs
            test_config.max_iters = 5 
            
            model = model_fn(test_config)
            loader = DataLoader(dataset, batch_size=bs, shuffle=True)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=test_config.learning_rate)
            scaler = torch.GradScaler(enabled=test_config.use_amp)
            
            model.train()
            for i, (x, y) in enumerate(loader):
                if i >= 3:  
                    break
                    
                x, y = x.to(test_config.device), y.to(test_config.device)
                
                optimizer.zero_grad()
                
                if test_config.use_amp:
                    with torch.autocast(device_type=test_config.device, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    _, loss = model(x, y)
                    loss.backward()
                    optimizer.step()
            
            del model, optimizer, scaler
            torch.cuda.empty_cache()
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return False
            raise e
    
    # Binary search
    low, high = start_bs, max_bs
    best_bs = start_bs
    
    while low <= high:
        mid = (low + high) // 2
        print(f"  Testing batch size: {mid}...", end=" ")
        
        if try_batch_size(mid):
            print("Success")
            best_bs = mid
            low = mid + 1
        else:
            print("OOM")
            high = mid - 1
    
    print(f"Maximum batch size: {best_bs}\n")
    return best_bs