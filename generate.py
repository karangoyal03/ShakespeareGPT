import torch
from pathlib import Path

from config import Config
from tokenizer import CharacterLevelTokenizer
from model import ShakespeareGPT

# Set random seed
torch.manual_seed(1357)

# Load data for tokenizer
with open('./data/shakespeare.txt','r',encoding='utf-8') as f:
    data = f.read()

# Initialize tokenizer and update config
tokenizer = CharacterLevelTokenizer(data)
Config.vocab_size = tokenizer.VOCAB_SIZE

# Initialize model
lm = ShakespeareGPT(Config)
lm = lm.to(device=Config.device)

# Load trained weights
model_path = Path('./shakespeareGPT.pth')
if model_path.exists():
    lm.load_state_dict(torch.load(model_path, map_location=Config.device))
    lm.eval()
    print("Model loaded successfully!")
else:
    print("Warning: No trained model found. Using untrained model.")

# Generate text
generated_texts = []
for length in [100, 300, 500, 700, 1000]:
    generated = lm.generate(
        torch.zeros((1,1), dtype=torch.long, device=Config.device), # initial context 0
        total=length
    )
    generated = tokenizer.decode(generated[0])
    text = f'generated ({length} tokens)\n{"="*50}\n{generated}\n{"="*50}\n\n'
    generated_texts.append(text)
    print(text)

# Save generated text
with open('generated.txt', 'w') as f:
    for text in generated_texts:
        f.write(text)

print("Generated text saved to generated.txt")