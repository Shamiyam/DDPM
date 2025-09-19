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

# --- Hyperparameters and Setup ---
T = 1000
beta_start = 1e-4
beta_end = 0.02
image_size = 28
channels = 1
batch_size = 128
lr = 2e-4
epochs = 15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Diffusion Schedule ---
betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# --- Helper Class and Functions ---

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

def forward_noising(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# --- Conditional U-Net Model ---

class ConditionalUNet(nn.Module):
    def __init__(self, in_ch=1, time_dim=128, class_dim=10, dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(TimestepEmbedder(time_dim), nn.Linear(time_dim, dim), nn.SiLU())
        self.class_embed = nn.Embedding(class_dim, dim)
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

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        y_emb = self.class_embed(y).unsqueeze(-1).unsqueeze(-1)
        h = self.norm1(F.silu(self.conv0(x)))
        h = h + t_emb + y_emb
        h1 = self.norm1(F.silu(self.conv1(h)))
        h2 = self.pool(h1)
        h2 = self.norm2(F.silu(self.conv2(h2)))
        h3 = self.norm2(F.silu(self.conv3(h2)))
        h4 = self.up(h3)
        h4 = torch.cat([h4, h1], dim=1)
        h4 = self.norm3(F.silu(self.conv4(h4)))
        h5 = self.norm3(F.silu(self.conv5(h4)))
        return self.conv_out(h5)

# --- Training and Sampling Functions ---

def train_conditional(model, dataloader, optimizer, checkpoint_path):
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
        for x_0, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_0 = x_0.to(device)
            y = y.to(device)
            t = torch.randint(0, T, (x_0.shape[0],)).to(device)
            x_t, noise = forward_noising(x_0, t)
            predicted_noise = model(x_t, t, y)
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
    return losses

@torch.no_grad()
def conditional_sample(model, n_samples, class_id):
    model.eval()
    y = torch.full((n_samples,), class_id, device=device, dtype=torch.long)
    x_t = torch.randn(n_samples, channels, image_size, image_size).to(device)
    for i in reversed(range(T)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        predicted_noise = model(x_t, t, y)
        alpha_t = alphas[t].view(-1, 1, 1, 1)
        alphas_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alphas_cumprod_t)) * predicted_noise)
        if i > 0:
            z = torch.randn_like(x_t)
            beta_t = betas[t].view(-1, 1, 1, 1)
            sigma = torch.sqrt(beta_t)
            x_t += sigma * z
    return torch.clamp(x_t, -1, 1)

# --- Main Execution ---

if __name__ == '__main__':
    # NEW: Define the main output directory in your Google Drive
    DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/Diffusion models/part_b'
    os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
    
    MODEL_PATH = os.path.join(DRIVE_OUTPUT_DIR, 'conditional_model.pth')
    CHECKPOINT_PATH = os.path.join(DRIVE_OUTPUT_DIR, 'ckpt.pth')
    LOSS_CURVE_PATH = os.path.join(DRIVE_OUTPUT_DIR, 'loss_curve.png')
    SAMPLES_PATH = os.path.join(DRIVE_OUTPUT_DIR, 'conditional_samples.png')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_cond = ConditionalUNet().to(device)
    optimizer_cond = torch.optim.Adam(model_cond.parameters(), lr=lr)

    losses_cond = train_conditional(model_cond, dataloader, optimizer_cond, CHECKPOINT_PATH)
    
    torch.save(model_cond.state_dict(), MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")

    plt.figure()
    plt.plot(losses_cond)
    plt.title('Conditional Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()
    print(f"Loss curve saved to {LOSS_CURVE_PATH}")

    classes_to_sample = [0, 3, 7]
    fig, axs = plt.subplots(1, len(classes_to_sample), figsize=(len(classes_to_sample) * 3, 3))
    for idx, c in enumerate(classes_to_sample):
        samples = conditional_sample(model_cond, n_samples=8, class_id=c)
        grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True)
        axs[idx].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axs[idx].set_title(f'Generated Class {c}')
        axs[idx].axis('off')
    plt.savefig(SAMPLES_PATH)
    plt.close()
    print(f"Conditional samples saved to {SAMPLES_PATH}")