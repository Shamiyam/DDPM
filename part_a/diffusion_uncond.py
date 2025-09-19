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

# Hyperparameters
T = 1000  # Timesteps
beta_start = 1e-4
beta_end = 0.02
image_size = 28
channels = 1  # Grayscale MNIST
batch_size = 128
lr = 2e-4
epochs = 10  # Lightweight for ~30min training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Linear beta schedule
betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

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
        self.time_dim = time_dim
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
        self.conv4 = nn.Conv2d(dim*2, dim, 3, padding=1)
        self.conv5 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv_out = nn.Conv2d(dim, in_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(16, dim*2)
        self.norm3 = nn.GroupNorm(8, dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        h = self.norm1(F.silu(self.conv0(x)))
        h = h + t_emb[:, :h.shape[1]]  # Add time to first layer
        h1 = self.norm1(F.silu(self.conv1(h)))
        h2 = self.pool(h1)
        h2 = self.norm2(F.silu(self.conv2(h2)))
        h3 = self.norm2(F.silu(self.conv3(h2)))
        h4 = self.up(h3)
        h4 = torch.cat([h4, h1], dim=1)  # Skip connection
        h4 = self.norm3(F.silu(self.conv4(h4)))
        h5 = self.norm3(F.silu(self.conv5(h4)))
        return self.conv_out(h5)

# Forward noising
def forward_noising(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# Training
def train(model, dataloader, optimizer):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader):
            x_0 = batch.to(device)
            t = torch.randint(0, T, (x_0.shape[0],)).to(device)
            x_t, noise = forward_noising(x_0, t)
            predicted_noise = model(x_t, t)
            loss = F.mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    return losses

# Sampling
@torch.no_grad()
def sample(model, n_samples=16):
    model.eval()
    x_t = torch.randn(n_samples, channels, image_size, image_size).to(device)
    for i in reversed(range(T)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        predicted_noise = model(x_t, t)
        alpha_t = alphas[t].view(-1, 1, 1, 1)
        one_minus_alpha_t = (1 - alpha_t).view(-1, 1, 1, 1)
        x_t_minus_one = (1 / torch.sqrt(alpha_t)) * (x_t - (one_minus_alpha_t / torch.sqrt(1 - alphas_cumprod[t].view(-1, 1, 1, 1))) * predicted_noise)
        if i > 0:
            sigma_t = torch.sqrt(betas[i].view(-1, 1, 1, 1))
            z = torch.randn_like(x_t)
            x_t_minus_one += sigma_t * z
        x_t = x_t_minus_one
    return torch.clamp(x_t, -1, 1)

# Data loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # To [-1,1]
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and optimizer
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Run training
losses = train(model, dataloader, optimizer)

# Plot loss
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.savefig('part_a/loss_curve.png')
plt.close()

# Generate samples at different steps (for simplicity, full sampling; adapt for intermediate)
samples = sample(model)
grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
plt.axis('off')
plt.savefig('part_a/samples_grid.png')
plt.close()

# Experiment: Try different timesteps T=500, lr=1e-4
# Repeat above with T=500, betas adjusted, lr=1e-4; compare loss curves and FID (manual viz)