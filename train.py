import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from config import Config
from tokenizer import CharacterLevelTokenizer
from dataset import ShakespeareDataset
from model import ShakespeareGPT

# Set random seed
torch.manual_seed(1357)

# Load data
with open('/data/shakespeare.txt','r',encoding='utf-8') as f:
    data = f.read()

# Initialize tokenizer and update config
tokenizer = CharacterLevelTokenizer(data)
Config.vocab_size = tokenizer.VOCAB_SIZE

# Create datasets
train_ds = ShakespeareDataset(Config, data)
val_ds = ShakespeareDataset(Config, data, is_test=True)

# Initialize model
lm = ShakespeareGPT(Config)
lm = lm.to(device=Config.device)

# Initialize optimizer
optim = torch.optim.AdamW(lm.parameters(), lr=Config.lr)

def loss_fn(logits, targets):
    B,T,C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits,targets)
    return loss

def train_N_iters():
    lm.train()
    train_step_losses = []
    for batch in tqdm(range(Config.train_iters)):
        optim.zero_grad()
        inputs, targets = train_ds.get()
        inputs, targets = inputs.to(device=Config.device), targets.to(device=Config.device)
        logits = lm(inputs)
        loss = loss_fn(logits,targets)
        loss.backward()
        optim.step()
        train_step_losses.append(loss.item())
        
        if batch%(Config.train_iters//10)==0 or batch==Config.train_iters-1:
            print(f"batch {batch} train step loss: {loss.item()}")
        
        del inputs, targets, loss, logits
        
    return train_step_losses
    
@torch.no_grad()
def valid_N_iters():
    lm.eval()
    val_step_losses = []
    for batch in tqdm(range(Config.val_iters)):
        inputs, targets = val_ds.get()
        inputs, targets = inputs.to(device=Config.device), targets.to(device=Config.device)
        logits = lm(inputs)
        loss = loss_fn(logits,targets)
        val_step_losses.append(loss.item())
        
        if batch%(Config.val_iters//10)==0 or batch==Config.val_iters-1:
            print(f"batch {batch} valid step loss: {loss.item()}")
        
        del inputs, targets, loss, logits
    
    return val_step_losses

def save_lm():
    state_dict = lm.state_dict()
    save_path = Path('./').resolve() / 'shakespeareGPT'
    save_path.mkdir(exist_ok=True)
    model_path = save_path / f'shakespeareGPT.pth'
    torch.save(state_dict, model_path)

def train_lm():
    train_losses = train_N_iters()
    valid_losses = valid_N_iters()
    save_lm()
    return train_losses, valid_losses

if __name__ == "__main__":
    # Train the model
    print("Starting training...")
    tl, vl = train_lm()
    
    # Plot losses
    plt.plot(tl, label='train loss', color='orange')
    plt.plot(vl, label='valid loss', color='blue')
    plt.title('Shakespeare GPT Losses')
    plt.legend()
    plt.savefig('losses.png')
    plt.show()
    
    print("Training complete! Model saved to shakespeareGPT.pth")