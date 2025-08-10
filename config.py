import torch
from dataclasses import dataclass

@dataclass
class Config:
    block_size = 256 # context-length
    batch_size = 64 # mini-batch size
    vocab_size = None  # Will be set dynamically
    n_embed = 256
    n_heads = 8
    head_size = n_embed // n_heads # computes to 384/6=64 or 128/4=32 or 256/8
    
    n_layers = 3
    
    train_iters = 10_000
    val_iters = 1000
    lr = 3e-4
    
    attn_dropout = 0.1
    block_dropout = 0.1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'