import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os

# # For checkpointing with Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# --- Model and Diffusion Definitions (Same as before) ---

# Timestep embedding
class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

# Simple U-Net for denoiser (lightweight for MNIST)
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, time_dim=128, dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimestepEmbedder(time_dim),
            nn.Linear(time_dim, dim),
            nn.SiLU()
        )
        self.conv0 = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(dim, dim*2, 3, padding=1)
        self.conv3 = nn.Conv2d(dim*2, dim*2, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(dim*2 + dim, dim, 3, padding=1)
        self.conv5 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv_out = nn.Conv2d(dim, in_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(16, dim*2)
        self.norm3 = nn.GroupNorm(8, dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = self.norm1(F.silu(self.conv0(x)))
        h = h + t_emb
        h1 = self.norm1(F.silu(self.conv1(h)))
        h2 = self.pool(h1)
        h2 = self.norm2(F.silu(self.conv2(h2)))
        h3 = self.norm2(F.silu(self.conv3(h2)))
        h4 = self.up(h3)
        h4 = torch.cat([h4, h1], dim=1)
        h4 = self.norm3(F.silu(self.conv4(h4)))
        h5 = self.norm3(F.silu(self.conv5(h4)))
        return self.conv_out(h5)

# --- Main Experiment Runner ---

def run_experiment(config):
    print(f"--- Starting Experiment: {config['name']} ---")
    
    T = config['T']
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    output_dir = config['output_dir']

    # NEW: Create the output directory in Google Drive if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # NEW: Define all output paths relative to the new directory
    model_path = os.path.join(output_dir, f"model_{config['name']}.pth")
    checkpoint_path = os.path.join(output_dir, f"ckpt_{config['name']}.pth")
    loss_curve_path = os.path.join(output_dir, f"loss_curve_{config['name']}.png")
    samples_path = os.path.join(output_dir, f"samples_{config['name']}.png")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("Starting training from scratch.")

    losses = []
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_0, _ = batch
            x_0 = x_0.to(device)
            t = torch.randint(0, T, (x_0.shape[0],)).to(device)
            
            noise = torch.randn_like(x_0)
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
            
            predicted_noise = model(x_t, t)
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    
    plt.figure()
    plt.plot(losses)
    plt.title(f'Training Loss ({config["name"]})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Loss curve saved to {loss_curve_path}")

    model.eval()
    with torch.no_grad():
        x_t = torch.randn(16, 1, 28, 28).to(device)
        for i in reversed(range(T)):
            t = torch.full((16,), i, device=device, dtype=torch.long)
            predicted_noise = model(x_t, t)
            alpha_t = alphas[t].view(-1, 1, 1, 1)
            alphas_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alphas_cumprod_t)) * predicted_noise)
            if i > 0:
                z = torch.randn_like(x_t)
                beta_t = betas[t].view(-1, 1, 1, 1)
                sigma = torch.sqrt(beta_t)
                x_t += sigma * z
    
    samples = torch.clamp(x_t, -1, 1)
    grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
    plt.figure(figsize=(4,4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(samples_path)
    plt.close()
    print(f"Sample grid saved to {samples_path}")
    print(f"--- Finished Experiment: {config['name']} ---")

# --- Define and Run Experiments ---

# NEW: Define the main output directory in your Google Drive
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/Diffusion models/part_a'

# Experiment 1: Baseline
baseline_config = {
    'name': 'Baseline_T1000_lr2e-4',
    'T': 1000,
    'lr': 2e-4,
    'epochs': 10,
    'batch_size': 128,
    'output_dir': DRIVE_OUTPUT_DIR,
}
run_experiment(baseline_config)

# Experiment 2: Different Hyperparameters
experiment_config = {
    'name': 'Experiment_T500_lr1e-4',
    'T': 500,
    'lr': 1e-4,
    'epochs': 15,
    'batch_size': 128,
    'output_dir': DRIVE_OUTPUT_DIR,
}
run_experiment(experiment_config)