import torch
import torch.nn.functional as F

def gem(x, p=3, eps=1e-6):
    if x.ndim == 3:
        x = x.permute(0, 2, 1)
    
    x_abs = x.abs().clamp(min=eps)
    x_pow = x_abs.pow(p) * x.sign()
    
    pooled = F.avg_pool1d(x_pow, x.size(-1)).squeeze(-1)
    
    pooled_abs = pooled.abs().clamp(min=eps)
    return pooled_abs.pow(1./p) * pooled.sign()
