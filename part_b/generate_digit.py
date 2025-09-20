import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import math
import argparse
import os

# --- Model and Helper Classes (Must match the training script) ---

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

# --- Main Generation Logic ---

def generate(args):
    """
    Loads a trained model and generates images of a specific digit.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Recreate Diffusion Schedule ---
    T = args.timesteps
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # --- Load Model ---
    model = ConditionalUNet().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        print("Please ensure the path is correct.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    model.eval()

    # --- Generation (Reverse Diffusion) ---
    print(f"Generating {args.num_samples} samples of digit '{args.digit}'...")
    with torch.no_grad():
        y = torch.full((args.num_samples,), args.digit, device=device, dtype=torch.long)
        x_t = torch.randn(args.num_samples, 1, 28, 28).to(device)
        for i in reversed(range(T)):
            t = torch.full((args.num_samples,), i, device=device, dtype=torch.long)
            predicted_noise = model(x_t, t, y)
            alpha_t = alphas[t].view(-1, 1, 1, 1)
            alphas_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alphas_cumprod_t)) * predicted_noise)
            if i > 0:
                z = torch.randn_like(x_t)
                beta_t = betas[t].view(-1, 1, 1, 1)
                sigma = torch.sqrt(beta_t)
                x_t += sigma * z
    
    samples = torch.clamp(x_t, -1, 1)
    
    # --- Save Output ---
    grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(args.num_samples)), normalize=True)
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.title(f'Generated Digit: {args.digit}')
    plt.axis('off')
    
    output_filename = os.path.join(args.output_dir, f'generated_digit_{args.digit}.png')
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved generated image to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate digits using a trained conditional diffusion model.")
    parser.add_argument('--digit', type=int, required=True, choices=range(10), help='The digit to generate (0-9).')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate (should be a perfect square).')
    parser.add_argument('--model_path', type=str, default='/content/drive/MyDrive/Diffusion models/part_b/conditional_model.pth', help='Path to the trained model .pth file.')
    parser.add_argument('--output_dir', type=str, default='/content/drive/MyDrive/Diffusion models/part_b/generated_samples', help='Directory to save the output image.')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of timesteps used during training.')
    
    args = parser.parse_args()
    
    generate(args)
