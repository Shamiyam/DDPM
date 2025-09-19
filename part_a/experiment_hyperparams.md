# Part A — Hyperparameter Experiment

## Experiment: Varying Timesteps

To study the effect of the number of diffusion steps, we train the unconditional DDPM with different values for `timesteps` (e.g., 50, 100, 200).

**Setup:**
- Dataset: MNIST
- Model: Simple UNet
- Timesteps: [50, 100, 200]
- Learning rate: 2e-4
- Epochs: 5

**Observations:**
- Fewer timesteps (e.g., 50) lead to faster training and sampling but may result in lower sample quality (images less sharp).
- More timesteps (e.g., 200) improve sample quality but increase training and sampling time.
- The loss curve stabilizes faster for lower timesteps.

**Recommendation:**
- For quick experiments, 100 timesteps offer a good balance between quality and speed on MNIST.

---

Repeat similar experiments for learning rate or batch size as needed.
