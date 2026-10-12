# Simple StyleGAN

A minimal, single-file PyTorch implementation of StyleGAN designed for legibility and quick experimentation. The entire architecture, training loop, and sampling logic are self-contained in [`stylegan.py`](stylegan.py) (~500 lines including inline comments).

Includes an integrated toy dataset of colored geometric shapes—allowing complete training from scratch on a laptop CPU in ~20 minutes with zero external assets required. Custom image datasets are also natively supported.

---

## Core Features & Mechanics

| Architectural Feature | Function | `stylegan.py` Reference |
|---|---|---|
| **Mapping Network** | Converts standard Gaussian noise $z$ into disentangled latent style $w$ via an 8-layer MLP | `MappingNetwork` |
| **Learned Input Tensor** | Replaces noise input with a learned $4 \times 4$ constant base input | `Generator.const` |
| **AdaIN Normalization** | Modulates feature map scale and bias per layer using vector $w$ | `AdaIN`, `StyleLayer` |
| **Noise Injection** | Introduces spatial per-pixel Gaussian noise for stochastic details | `NoiseInjection` |
| **Style Mixing** | Swaps two distinct $w$ vectors mid-network during training to enforce feature localization | `Generator.forward(mix_prob=...)` |
| **Truncation Trick** | Scales $w$ toward the empirical mean to trade sample diversity for fidelity | `Generator.z_to_w(truncation=...)` |
| **Equalized LR** | Dynamically scales weights at runtime for stable learning across all layers | `EqualLinear`, `EqualConv2d` |
| **Minibatch StdDev** | Passes cross-sample variance to the Discriminator to prevent mode collapse | `minibatch_stddev` |
| **Lazy R1 Penalty** | Evaluates gradient penalty on real samples every $N$ steps to maintain stability efficiently | `train()` |
| **Exponential Moving Average** | Maintains an EMA copy of generator weights for evaluation and sampling | `train()` $\rightarrow$ `G_ema` |

---

## Network Architecture & Style Hierarchy

```
            z (128-dim)                           Generator Network (32×32 output)
                 │
     ┌───────────▼───────────┐         const 4×4 ─► Noise ─► AdaIN ◄── w[0]
     │    Mapping Network    │                        Conv  ─► Noise ─► AdaIN ◄── w[1]
     │ PixelNorm + 8×Linear  │                                   │
     └───────────┬───────────┘          Upsample ─► Conv ─► Noise ─► AdaIN ◄── w[2]   (8×8)
                 │                                  Conv ─► Noise ─► AdaIN ◄── w[3]
                 w (128-dim)                                     │
                 │                      Upsample ─► Conv ─► Noise ─► AdaIN ◄── w[4]   (16×16)
   Broadcast across layers ───────────►             Conv ─► Noise ─► AdaIN ◄── w[5]
   (or apply style mixing)                                       │
                                        Upsample ─► Conv ─► Noise ─► AdaIN ◄── w[6]   (32×32)
                                                    Conv ─► Noise ─► AdaIN ◄── w[7]
                                                                 │
                                                            1×1 Conv ─► Output RGB
```

### Feature Control Progression
* **Coarse Layers ($4\times4, 8\times8$):** Control high-level geometry, pose, shape, and spatial distribution.
* **Fine Layers ($16\times16, 32\times32$):** Control micro-features, color palettes, edge sharpening, and surface textures.

---

## Setup & Execution

### Installation

Requires **Python 3.9+**. Dependencies are limited to `torch`, `numpy`, and `Pillow` (no `torchvision` required). Hardware acceleration (`cuda` or `mps`) is auto-detected.

```bash
pip install -r requirements.txt
```

### Quickstart Workflow

```bash
# 1. Train on built-in synthetic shapes dataset
python stylegan.py train

# 2. Generate 16 synthetic samples
python stylegan.py generate --ckpt runs/stylegan/checkpoint.pt

# 3. Perform a style-mixing matrix export
python stylegan.py mix --ckpt runs/stylegan/checkpoint.pt
```

*Training logs automatically export fixed-latent sample grids to `runs/stylegan/samples/step_XXXXXX.png` every 500 steps.*

---

## Working with Custom Datasets

Point `--data` to any directory containing images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`). Images are automatically square-cropped, resized, and randomly flipped horizontally.

```bash
python stylegan.py train \
  --data path/to/dataset \
  --img-size 64 \
  --channels 128 \
  --steps 20000 \
  --out runs/custom_run
```

### Configuration Guidelines

* **Dataset Volume:** Recommended minimum of 2,000+ images. Very small datasets ($< 500$ samples) will result in discriminator overfitting.
* **Resolution Scaling:** Start at $32\times32$ or $64\times64$. Higher resolutions ($128\times128+$) require dedicated GPUs.
* **Model Capacity:** Increase `--channels` ($128 \rightarrow 256$) when modeling complex real-world photographic datasets.
* **Resuming Progress:** Append `--resume runs/custom_run/checkpoint.pt --steps <total_target_steps>`. Hyperparameters automatically restore from saved metadata.

---

## Advanced Options & CLI Reference

### Inference & Sampling

```bash
# Generate deterministic samples with truncation controls
python stylegan.py generate \
  --ckpt runs/stylegan/checkpoint.pt \
  --n 64 \
  --truncation 0.7 \
  --seed 42 \
  --out samples.png

# Generate style mixing grid matrix
python stylegan.py mix \
  --ckpt runs/stylegan/checkpoint.pt \
  --n 5 \
  --cut 4 \
  --out mixing.png
```

* **Truncation ($\psi$):** Standard range is `0.5`–`0.7`. Values near `1.0` preserve full distribution variation (higher risk of artifacts); values near `0.0` collapse toward the dataset mean image.
* **Crossover Point (`--cut`):** Layer threshold at which fine styles switch source inputs.

### CLI Command Flags

#### `train`
| Flag | Default | Description |
|---|---|---|
| `--data` | `shapes` | Path to image folder or `shapes` for internal dataset |
| `--img-size` | `32` | Target image resolution (must be power of 2, $\ge 8$) |
| `--z-dim` / `--w-dim` | `128` | Dimensionality of latent $z$ and style $w$ |
| `--channels` | `64` | Base feature map count across convolutional layers |
| `--mapping-layers` | `8` | Layer depth of the MLP mapping network |
| `--batch` | `32` | Training batch size |
| `--steps` | `3000` | Target training iteration count |
| `--lr` | `0.002` | Adam optimizer learning rate ($\beta_1=0, \beta_2=0.99$) |
| `--r1` | `0.1` | Regularization coefficient $\gamma$ |
| `--r1-every` | `4` | Interval frequency for lazy R1 loss computation |
| `--mix-prob` | `0.9` | Probability of style mixing per iteration |
| `--ema` | `0.999` | Exponential moving average decay rate |
| `--out` | `runs/stylegan` | Output directory for checkpoints and logs |
| `--log-every` / `--sample-every` | `50` / `500` | Log frequency / Sample rendering interval |
| `--resume` | None | Checkpoint path to load state |
| `--device` | `auto` | Force execution backend (`auto`, `cpu`, `cuda`, `mps`) |

---

## Diagnostics & Loss Monitoring

Sample stdout output during execution:
```
step   1000 | D 0.912 | G 1.734 | R1 0.052 | 450s
```

* **Loss Equilibrium:** $D$ and $G$ values should fluctuate dynamically. A monotonically declining $D$ loss approaching $0.0$ combined with exploding $G$ loss indicates total discriminator dominance. Remediation: Increase `--r1`, reduce `--lr`, or expand training dataset volume.
* **R1 regularizer tracking:** Unbounded growth in R1 values signals that the discriminator is developing unstable, overly sharp decision boundaries.

---

## Deviations from Official Paper Implementations

This implementation prioritizes architectural readability over state-of-the-art benchmark matching:

1. **No Progressive Growing:** Operates directly at target output resolution (adopting StyleGAN2 mechanics) rather than growing layers dynamically.
2. **Standard AdaIN Operations:** Uses direct AdaIN normalization (StyleGAN1) instead of weight demodulation (StyleGAN2).
3. **Flat Capacity Scaling:** Feature channels remain uniform across spatial steps; bilinear interpolation is used for scaling without antialiasing low-pass filters.
4. **Omitted Optimization Features:** Excludes path-length regularization, adaptive discriminator augmentation (ADA), mixed precision operations, and FID tracking loops.

---

## Key References

* **StyleGAN:** Karras et al., *A Style-Based Generator Architecture for Generative Adversarial Networks* (CVPR 2019). [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
* **StyleGAN2:** Karras et al., *Analyzing and Improving the Image Quality of StyleGAN* (CVPR 2020). [arXiv:1912.04958](https://arxiv.org/abs/1912.04958)
* **Progressive GANs:** Karras et al., *Progressive Growing of GANs for Improved Quality, Stability, and Variation* (ICLR 2018). [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
* **R1 Regularization:** Mescheder et al., *Which Training Methods for GANs do actually Converge?* (ICML 2018). [arXiv:1801.04406](https://arxiv.org/abs/1801.04406)
