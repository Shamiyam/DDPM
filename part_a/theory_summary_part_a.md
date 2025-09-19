# Part A — Basic Diffusion Model (Unconditional)

## Forward & Reverse Diffusion Processes

Diffusion models are generative models that learn to reverse a gradual noising process. The **forward process** (diffusion) adds Gaussian noise to data over several timesteps, transforming a clean image into pure noise. Mathematically, at each timestep $t$, the data $x_0$ is corrupted to $x_t$ using a schedule of noise levels (betas):

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

where $\bar{\alpha}_t$ is the cumulative product of $1-\beta_t$ up to $t$.

The **reverse process** aims to recover $x_0$ from $x_T$ (pure noise) by iteratively denoising using a neural network. At each step, the model predicts the noise added, and the sample is updated accordingly.

## Simplified Training Objective

The training objective is to teach the denoiser network to predict the noise $\epsilon$ added at each timestep. The loss is:

$$
\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

where $\epsilon_\theta$ is the model's prediction of the noise.

## Role of Denoiser Neural Network (CNN/UNet)

The denoiser network (often a UNet or CNN) takes the noisy image $x_t$ and timestep $t$ as input and predicts the noise component. This prediction is used in the reverse process to iteratively denoise and reconstruct the original image. The UNet architecture is popular due to its ability to capture multi-scale features, but lightweight CNNs can also be used for small datasets like MNIST.

---

**References:**
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021)
