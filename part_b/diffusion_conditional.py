# Reuse imports and schedules from part_a

# Modified U-Net with class conditioning
class ConditionalUNet(nn.Module):
    def __init__(self, in_ch=1, time_dim=128, class_dim=10, dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(TimestepEmbedder(time_dim), nn.Linear(time_dim, dim), nn.SiLU())
        self.class_embed = nn.Embedding(class_dim, dim)
        self.conv0 = nn.Conv2d(in_ch, dim, 3, padding=1)
        # ... (same as SimpleUNet conv layers)
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

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        y_emb = self.class_embed(y).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        h = self.norm1(F.silu(self.conv0(x)))
        h = h + t_emb[:, :h.shape[1]] + y_emb[:, :h.shape[1]]  # Add time + class
        # ... (rest same as SimpleUNet)
        h1 = self.norm1(F.silu(self.conv1(h)))
        h2 = self.pool(h1)
        h2 = self.norm2(F.silu(self.conv2(h2)))
        h3 = self.norm2(F.silu(self.conv3(h2)))
        h4 = self.up(h3)
        h4 = torch.cat([h4, h1], dim=1)
        h4 = self.norm3(F.silu(self.conv4(h4)))
        h5 = self.norm3(F.silu(self.conv5(h4)))
        return self.conv_out(h5)

# Updated forward_noising (same)
# Updated training
def train_conditional(model, dataloader, optimizer):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch, labels in tqdm(dataloader):
            x_0 = batch.to(device)
            y = labels.to(device)
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
    return losses

# Updated data loader with labels
dataloader_cond = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Already has labels

# Model
model_cond = ConditionalUNet().to(device)
optimizer_cond = torch.optim.Adam(model_cond.parameters(), lr=lr)

# Train
losses_cond = train_conditional(model_cond, dataloader_cond, optimizer_cond)

# Plot loss
plt.plot(losses_cond)
plt.title('Conditional Training Loss')
plt.savefig('part_b/loss_curve.png')
plt.close()

# Conditional sampling
@torch.no_grad()
def conditional_sample(model, n_samples=16, class_id=0):
    model.eval()
    y = torch.full((n_samples,), class_id, device=device, dtype=torch.long)
    x_t = torch.randn(n_samples, channels, image_size, image_size).to(device)
    for i in reversed(range(T)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        predicted_noise = model(x_t, t, y)
        # Same reverse step as unconditional
        alpha_t = alphas[t].view(-1, 1, 1, 1)
        one_minus_alpha_t = (1 - alpha_t).view(-1, 1, 1, 1)
        x_t_minus_one = (1 / torch.sqrt(alpha_t)) * (x_t - (one_minus_alpha_t / torch.sqrt(1 - alphas_cumprod[t].view(-1, 1, 1, 1))) * predicted_noise)
        if i > 0:
            sigma_t = torch.sqrt(betas[i].view(-1, 1, 1, 1))
            z = torch.randn_like(x_t)
            x_t_minus_one += sigma_t * z
        x_t = x_t_minus_one
    return torch.clamp(x_t, -1, 1)

# Generate for classes 0,3,7
classes = [0, 3, 7]
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for idx, c in enumerate(classes):
    samples = conditional_sample(model_cond, 8, c)
    grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True)
    axs[idx].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    axs[idx].set_title(f'Class {c}')
    axs[idx].axis('off')
plt.savefig('part_b/conditional_samples.png')
plt.close()

# Compare with unconditional (reuse sample from part_a)