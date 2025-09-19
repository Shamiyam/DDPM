# Part B — Conditioning Implementation & Observations

## Conditioning Implementation

The conditional DDPM modifies the denoiser network to accept both the noisy image and the class label. Labels are embedded using an `nn.Embedding` layer and concatenated with timestep embeddings. This combined embedding is injected into the network's feature maps, allowing the model to generate images conditioned on the specified class.

During training, the label for each image is provided to the model. During sampling, the desired class label is set for each generated image.

## Quality Observations

- Conditional samples are more coherent and match the specified class (e.g., digits 0, 1, 2 in MNIST).
- Compared to the unconditional model, conditional samples show improved class-specific features and less ambiguity.
- The loss curve for the conditional model is similar to the unconditional case, but sample quality is higher for class-specific generations.

---

**References:**
- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (NeurIPS 2021)
