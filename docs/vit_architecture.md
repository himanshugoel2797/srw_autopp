# ViT-Based Propagation Parameter Predictor — Architecture Design

## 1. Why Vision Transformer

The core challenge: SRW wavefront grids range from 64×64 to 4096×4096+,
with physical scales from nanometres to metres. The propagation parameters
must satisfy the worst-case region of the wavefront, but each region's
requirements depend on its context within the whole beam.

A Vision Transformer solves this naturally:

| Problem | ViT solution |
|---------|-------------|
| Variable input size | More/fewer patches, same model |
| Need global context per local region | Self-attention connects all patches |
| Physical scale varies by orders of magnitude | Physics-normalised channels + physical position encoding |
| Worst-case region drives parameters | Attention-weighted aggregation over patches |
| Analytical prior needs to inform spatial analysis | Prior as a special token attending to all patches |

## 2. Architecture Overview

```
INPUT WAVEFRONT (nz × nx, arbitrary size)
                │
                ▼
┌─────────────────────────────────────────────────────┐
│  Patchify: split into P×P patches (P = 128)         │
│  Each patch: 5 channels (same as before)            │
│  N_patches = (nx/P) × (nz/P)                        │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Patch Embedding: Conv or Linear projection          │
│  (5, P, P) → (D,) per patch                         │
│  D = 256 (embedding dimension)                       │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
    [PRIOR token]  [patch_1]  ...  [patch_N]
     (D=256)        + pos_enc      + pos_enc
          │             │              │
          └─────────────┼──────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Transformer Encoder (L layers, H heads)             │
│                                                      │
│  Layer: MultiHeadAttention → LayerNorm               │
│         → FeedForward(D→4D→D) → LayerNorm            │
│                                                      │
│  L = 4–6 layers, H = 4–8 heads                      │
│  All patches + PRIOR token attend to each other      │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
    [PRIOR token]  [patch_1]  ...  [patch_N]
    (global repr)  (local repr)    (local repr)
          │             │              │
          │             └──────┬───────┘
          │                    ▼
          │            Max-pool / Attention-pool
          │            over patch tokens → (D,)
          │                    │
          └────────┬───────────┘
                   ▼
             Concat (2D = 512)
                   │
                   ▼
            Dense 512→128, ReLU
            Dense 128→64, ReLU
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       AT head  Resize Δ  Confidence
       (3)      (4)       (1)
```

## 3. Components in Detail

### 3.1 Patchification

```python
def patchify(spatial_maps, patch_size=128):
    """
    Split (5, H, W) spatial maps into patches of (5, P, P).
    
    Returns:
        patches: (N_patches, 5, P, P)
        positions: (N_patches, 4)  — physical coords of each patch
                   [x_center, z_center, x_frac, z_frac]
    """
    C, H, W = spatial_maps.shape
    P = patch_size
    
    # Pad to multiple of P if needed
    pad_h = (P - H % P) % P
    pad_w = (P - W % P) % P
    if pad_h > 0 or pad_w > 0:
        spatial_maps = np.pad(spatial_maps, 
            ((0,0), (0,pad_h), (0,pad_w)), mode='constant')
    
    _, H_pad, W_pad = spatial_maps.shape
    nh, nw = H_pad // P, W_pad // P
    
    # Reshape into patches
    patches = spatial_maps.reshape(C, nh, P, nw, P)
    patches = patches.transpose(1, 3, 0, 2, 4)  # (nh, nw, C, P, P)
    patches = patches.reshape(nh * nw, C, P, P)
    
    # Physical positions (as fraction of grid, 0=left/bottom, 1=right/top)
    positions = np.zeros((nh * nw, 4))
    idx = 0
    for i in range(nh):
        for j in range(nw):
            positions[idx] = [
                (j + 0.5) / nw,    # x_frac: 0→1 across grid
                (i + 0.5) / nh,    # z_frac: 0→1 across grid
                j / nw,            # x_start_frac
                i / nh,            # z_start_frac
            ]
            idx += 1
    
    return patches, positions
```

**Patch size choice:** 128×128 gives each patch a large field of view — enough
to capture multiple fringe cycles, ring structure from zone plates, and speckle
correlation lengths from mirror figure errors. The larger patch means fewer
patches per grid, keeping the transformer sequence length short and attention
cheap even for large grids.

**Minimum grid size:** 128×128 (1 patch + prior token = 2 tokens). Grids smaller
than 128 in either dimension are padded to 128. This is rare in practice — SRW
grids are almost always ≥256.

| Grid size | Patches (P=128) | Sequence length | Attention cost |
|-----------|-----------------|-----------------|----------------|
| 128×128 | 1×1 = 1 | 2 (with prior) | Trivial (prior-only refinement) |
| 256×256 | 2×2 = 4 | 5 (with prior) | Trivial |
| 512×512 | 4×4 = 16 | 17 | Trivial |
| 1024×1024 | 8×8 = 64 | 65 | Light |
| 2048×2048 | 16×16 = 256 | 257 | Light |
| 4096×4096 | 32×32 = 1024 | 1025 | Moderate |

For grids larger than 4096², use stride-2 patches (skip every other patch)
or a window attention variant to keep sequence length manageable.

### 3.2 Patch Embedding

Each (5, 128, 128) patch is projected to a D-dimensional vector.

**Option A: Linear projection** (original ViT)
```python
# Flatten and project
patch_flat = patch.reshape(5 * 128 * 128)  # 81920-dim
embedding = Linear(81920, D)(patch_flat)    # D=256
```

**Option B: Small CNN** (hybrid, recommended)
```python
# 4-layer CNN to extract local features before sequence processing
h = Conv(5, 32, 3, stride=2, pad=1) → BN → ReLU   # (32, 64, 64)
h = Conv(32, 64, 3, stride=2, pad=1) → BN → ReLU   # (64, 32, 32)
h = Conv(64, 128, 3, stride=2, pad=1) → BN → ReLU  # (128, 16, 16)
h = Conv(128, D, 3, stride=2, pad=1) → ReLU         # (D, 8, 8)
embedding = h.mean(dim=(2,3))                         # (D,) global avg pool
```

**Recommendation: Option B (hybrid).** With 128×128 patches, the linear
projection would be 81920→256, which is wasteful and hard to train. The
4-layer CNN progressively reduces spatial dimensions (128→64→32→16→8)
while building up channel depth, extracting local spatial features
(fringe spacing, speckle density, edge intensity, ring structure) that
the transformer then reasons about globally. This is the architecture
used in most successful vision transformer variants (DeiT, Swin, etc.).

### 3.3 Position Encoding

Standard ViT uses learned position embeddings, but we have physically
meaningful coordinates. Use sinusoidal encoding of the physical position:

```python
def physical_position_encoding(positions, D):
    """
    Encode patch physical positions using sinusoidal encoding.
    
    positions: (N, 4) — [x_frac, z_frac, x_start, z_start]
    D: embedding dimension (must be divisible by 8)
    
    Returns: (N, D)
    """
    N = positions.shape[0]
    d = D // 8  # frequencies per coordinate
    
    # Frequency bands (log-spaced)
    freqs = 2 ** torch.linspace(0, 6, d)  # 1 to 64 cycles across grid
    
    enc = []
    for coord_idx in range(4):
        coord = positions[:, coord_idx:coord_idx+1]  # (N, 1)
        angles = coord * freqs.unsqueeze(0) * 2 * pi  # (N, d)
        enc.append(torch.sin(angles))
        enc.append(torch.cos(angles))
    
    return torch.cat(enc, dim=1)  # (N, 8d = D)
```

This encoding is:
- **Scale-invariant:** position is in grid fractions (0–1), not physical units
- **Resolution-invariant:** a patch at 75% of the grid range encodes the same
  regardless of whether that's 0.5 mm or 50 mm from center
- **Informative about edge proximity:** patches near 0 or 1 are at grid edges
  (where clipping risk is highest)

### 3.4 Prior Token

The analytical prior (12 scalars from Stage 1) is projected to a D-dimensional
token and prepended to the patch sequence, similar to BERT's [CLS] token:

```python
class PriorToken(nn.Module):
    def __init__(self, n_prior_features=12, D=256):
        self.proj = nn.Sequential(
            nn.Linear(n_prior_features, 64),
            nn.ReLU(),
            nn.Linear(64, D),
        )
    
    def forward(self, prior_scalars):
        # prior_scalars: (B, 12)
        return self.proj(prior_scalars).unsqueeze(1)  # (B, 1, D)
```

This token participates in self-attention with all patches. The attention
mechanism naturally implements: "given the analytical prediction of R=2m,
does each patch's sampling quality map match what I'd expect?"

### 3.5 Transformer Encoder

Standard pre-norm transformer with modifications for this domain:

```python
class TransformerEncoder(nn.Module):
    def __init__(self, D=256, n_layers=4, n_heads=8, ff_mult=4, dropout=0.1):
        self.layers = nn.ModuleList([
            TransformerBlock(D, n_heads, ff_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(D)
    
    def forward(self, x):
        # x: (B, N+1, D) — prior token + patch tokens
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, D, n_heads, ff_mult, dropout):
        self.attn = nn.MultiheadAttention(D, n_heads, dropout=dropout, 
                                           batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(D, D * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * ff_mult, D),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
    
    def forward(self, x):
        # Pre-norm attention
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        # Pre-norm feedforward
        h = self.norm2(x)
        x = x + self.ff(h)
        return x
```

**Why 4 layers is enough:** The reasoning required is not deep — it's mostly
"find the worst patch, compare it to the prior, decide if correction is needed."
This doesn't require the 12–24 layers typical in NLP. Vision transformers for
classification often use 6–12 layers, and our task is simpler than ImageNet
classification.

### 3.6 Output Aggregation

Two representations are extracted from the transformer output:

```python
# After transformer: (B, N+1, D)
prior_out = transformer_out[:, 0, :]     # (B, D) — prior token
patch_out = transformer_out[:, 1:, :]    # (B, N, D) — patch tokens

# Global representation via attention-weighted pooling
# Each patch gets an importance weight based on its relevance
attn_weights = torch.softmax(
    nn.Linear(D, 1)(patch_out).squeeze(-1),  # (B, N)
    dim=1
)
patch_global = (patch_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

# Also keep max-pooled representation (worst-case signal)
patch_max = patch_out.max(dim=1).values  # (B, D)

# Combine
combined = torch.cat([prior_out, patch_global, patch_max], dim=1)  # (B, 3D)
```

The three components capture:
- **Prior token:** global context informed by analytical prediction
- **Attention-pooled patches:** weighted summary emphasising important regions
- **Max-pooled patches:** worst-case features that drive conservative parameter choices

### 3.7 Prediction Heads

```python
class PredictionHeads(nn.Module):
    def __init__(self, D=256):
        self.shared = nn.Sequential(
            nn.Linear(3 * D, 128),  # 768→128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.at_head = nn.Linear(64, 3)     # agree, force_0, force_1
        self.resize_head = nn.Linear(64, 4)  # delta log resize
        self.conf_head = nn.Linear(64, 1)    # confidence
    
    def forward(self, combined):
        h = self.shared(combined)
        return {
            'at_logits': self.at_head(h),
            'resize_delta': self.resize_head(h),
            'confidence': torch.sigmoid(self.conf_head(h)),
        }
```

## 4. Complete Model

```python
class WavefrontViT(nn.Module):
    """
    Vision Transformer for SRW propagation parameter prediction.
    
    Handles arbitrary input resolutions via patchification.
    Integrates analytical prior as a special token.
    Predicts corrections to analytical parameter estimates.
    """
    
    def __init__(self, 
                 patch_size=128,
                 n_channels=5,
                 n_prior_features=12,
                 D=256,
                 n_layers=4,
                 n_heads=8,
                 ff_mult=4,
                 dropout=0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.D = D
        
        # Patch embedding: 4-layer CNN per 128×128 patch → D-dim vector
        self.patch_embed = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, stride=2, padding=1),   # →64×64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),          # →32×32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),         # →16×16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, D, 3, stride=2, padding=1),          # →8×8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                             # →(D, 1, 1)
        )
        
        # Prior token projection
        self.prior_proj = nn.Sequential(
            nn.Linear(n_prior_features, 64),
            nn.ReLU(),
            nn.Linear(64, D),
        )
        
        # Position encoding (sinusoidal, not learned)
        # Registered as buffer (not parameter)
        self.pos_dim = D
        
        # Transformer
        self.transformer = TransformerEncoder(D, n_layers, n_heads, ff_mult, dropout)
        
        # Output
        self.attn_pool = nn.Linear(D, 1)  # attention pooling weights
        self.heads = PredictionHeads(D)
    
    def forward(self, spatial_maps, prior_scalars, grid_info):
        """
        Parameters
        ----------
        spatial_maps : (B, 5, H, W) — physics-normalised channels
        prior_scalars : (B, 12) — analytical prior features
        grid_info : dict with 'dx', 'dz', 'nx', 'nz' for position encoding
        
        Returns
        -------
        dict with 'at_logits', 'resize_delta', 'confidence'
        """
        B, C, H, W = spatial_maps.shape
        P = self.patch_size
        
        # Pad to multiple of P
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            spatial_maps = F.pad(spatial_maps, (0, pad_w, 0, pad_h))
        
        _, _, H_pad, W_pad = spatial_maps.shape
        nh, nw = H_pad // P, W_pad // P
        N = nh * nw
        
        # Extract and embed patches
        patches = spatial_maps.unfold(2, P, P).unfold(3, P, P)  # (B, C, nh, nw, P, P)
        patches = patches.contiguous().view(B * N, C, P, P)
        
        patch_embeddings = self.patch_embed(patches)  # (B*N, D, 1, 1)
        patch_embeddings = patch_embeddings.view(B, N, self.D)  # (B, N, D)
        
        # Position encoding
        positions = self._make_positions(nh, nw, spatial_maps.device)  # (N, 4)
        pos_enc = self._sinusoidal_encoding(positions)  # (N, D)
        patch_embeddings = patch_embeddings + pos_enc.unsqueeze(0)
        
        # Prior token
        prior_token = self.prior_proj(prior_scalars).unsqueeze(1)  # (B, 1, D)
        
        # Assemble sequence: [prior, patch_1, ..., patch_N]
        sequence = torch.cat([prior_token, patch_embeddings], dim=1)  # (B, N+1, D)
        
        # Transformer
        encoded = self.transformer(sequence)  # (B, N+1, D)
        
        # Extract representations
        prior_out = encoded[:, 0, :]       # (B, D)
        patches_out = encoded[:, 1:, :]    # (B, N, D)
        
        # Attention-weighted pool
        attn_w = torch.softmax(self.attn_pool(patches_out).squeeze(-1), dim=1)
        patch_attn = (patches_out * attn_w.unsqueeze(-1)).sum(dim=1)  # (B, D)
        
        # Max pool
        patch_max = patches_out.max(dim=1).values  # (B, D)
        
        # Predict
        combined = torch.cat([prior_out, patch_attn, patch_max], dim=1)  # (B, 3D)
        return self.heads(combined)
    
    def _make_positions(self, nh, nw, device):
        """Grid-fraction positions for each patch."""
        positions = torch.zeros(nh * nw, 4, device=device)
        idx = 0
        for i in range(nh):
            for j in range(nw):
                positions[idx] = torch.tensor([
                    (j + 0.5) / nw,  # x_frac
                    (i + 0.5) / nh,  # z_frac
                    j / max(nw - 1, 1),  # x_edge_proximity
                    i / max(nh - 1, 1),  # z_edge_proximity
                ])
                idx += 1
        return positions
    
    def _sinusoidal_encoding(self, positions):
        """Sinusoidal encoding of physical positions."""
        N, n_coords = positions.shape
        d = self.D // (2 * n_coords)  # frequencies per coordinate
        
        freqs = 2.0 ** torch.linspace(0, 6, d, device=positions.device)
        
        enc_parts = []
        for c in range(n_coords):
            coord = positions[:, c:c+1]  # (N, 1)
            angles = coord * freqs.unsqueeze(0) * 2 * 3.14159265  # (N, d)
            enc_parts.append(torch.sin(angles))
            enc_parts.append(torch.cos(angles))
        
        enc = torch.cat(enc_parts, dim=1)  # (N, 2*n_coords*d)
        # Pad or truncate to D
        if enc.shape[1] < self.D:
            enc = F.pad(enc, (0, self.D - enc.shape[1]))
        elif enc.shape[1] > self.D:
            enc = enc[:, :self.D]
        
        return enc


## 5. Model Configuration

### Recommended hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Patch size (P) | 128×128 | Large FOV per patch captures fringe structure, ring patterns, and speckle correlation lengths. Keeps patch count low for efficient attention. |
| Embedding dim (D) | 256 | Sufficient for this problem, keeps model small |
| Transformer layers | 4 | Shallow — task requires pattern matching, not deep reasoning |
| Attention heads | 8 | Standard for D=256 (32 dim per head) |
| FF multiplier | 4 | Standard (D→4D→D in feedforward) |
| Dropout | 0.1 | Light regularisation |
| Patch embedding | 4-layer CNN | 128→64→32→16→8 spatial reduction, better than linear for spatial features |

### Model size

| Component | Parameters |
|-----------|-----------|
| Patch embedding CNN (4 layers) | ~120K |
| Prior projection | ~17K |
| Transformer (4 layers) | ~1.1M |
| Pooling + heads | ~120K |
| **Total** | **~1.4M** |

Small enough to train on a single GPU in hours, fast enough for <10ms inference.

### Scaling for large grids

| Grid | Patches (P=128) | Sequence length | Strategy |
|------|-----------------|-----------------|----------|
| ≤512² | ≤16 | ≤17 | Standard attention |
| 512²–2048² | 16–256 | 17–257 | Standard attention |
| 2048²–4096² | 256–1024 | 257–1025 | Standard (still feasible) |
| >4096² | >1024 | >1025 | Stride-2 patches OR windowed attention |

For stride-2 patching on very large grids (>4096²):
```python
# Skip every other patch position
patches = spatial_maps.unfold(2, P, 2*P).unfold(3, P, 2*P)
# 8192×8192 → 32×32 = 1024 patches instead of 64×64 = 4096
# Still captures global structure, just at 2× coarser sampling
```

## 6. Training

### Data generation

Same as previous design:
1. Generate diverse wavefronts (Gaussian, zone plate, CRL, KB, grating)
2. Run Stage 1 analytical estimator → prior
3. Try ~30 parameter candidates with SRW propagation + validator
4. Best candidate → compute delta from analytical → label
5. Patchify wavefront → CNN input

### Loss function

```python
def loss(output, target, analytical_prior):
    # AT override: 3-class CE
    L_at = F.cross_entropy(output['at_logits'], target['at_class'],
                           label_smoothing=0.1)
    
    # Resize delta: Smooth L1 in log space
    L_resize = F.smooth_l1_loss(output['resize_delta'], target['resize_delta'])
    
    # Confidence calibration
    L_conf = F.binary_cross_entropy(output['confidence'].squeeze(), 
                                     target['quality'])
    
    # Regularisation: penalise large corrections
    # (prefer staying close to analytical suggestion)
    L_reg = 0.01 * (output['resize_delta'] ** 2).mean()
    
    return L_at + L_resize + 0.5 * L_conf + L_reg
```

The regularisation term `L_reg` encodes the prior that the analytical
estimate is usually right — corrections should be small unless the
spatial evidence is strong.

### Training procedure

- Optimizer: AdamW, lr=1e-4, weight decay=0.05
- Schedule: cosine annealing over 100 epochs
- Batch size: 16–32 (variable sequence length requires padding within batch)
- Augmentation: horizontal/vertical flips (swap pxm↔pzm labels accordingly)
- Validation: 15% held out, stratified by wavefront type

### Handling variable sequence lengths in batches

```python
def collate_fn(batch):
    """Pad patch sequences to the longest in the batch."""
    max_patches = max(item['patches'].shape[0] for item in batch)
    
    padded_patches = []
    attention_masks = []
    
    for item in batch:
        n = item['patches'].shape[0]
        pad_n = max_patches - n
        
        padded = F.pad(item['patches'], (0,0, 0,0, 0,0, 0,pad_n))
        mask = torch.cat([torch.ones(n+1), torch.zeros(pad_n)])  # +1 for prior
        
        padded_patches.append(padded)
        attention_masks.append(mask)
    
    return {
        'patches': torch.stack(padded_patches),
        'masks': torch.stack(attention_masks),
        'priors': torch.stack([item['prior'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
    }
```

## 7. Inference Pipeline

```python
class DriftParameterPredictor:
    def __init__(self, model_path, analytical_estimator):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.estimator = analytical_estimator
    
    def predict(self, wfr, drift_length_m, 
                R_x=None, R_z=None, sigma_x=None, sigma_z=None):
        """
        Full prediction pipeline.
        
        1. Run analytical estimator → prior params
        2. Prepare spatial maps from wavefront
        3. Run ViT → get corrections
        4. Apply corrections to analytical params
        5. Return final parameters + diagnostics
        """
        # Stage 1: Analytical
        prior = self.estimator.estimate(
            R_x=R_x or wfr.Robs_x,
            R_z=R_z or wfr.Robs_z,
            sigma_x=sigma_x, sigma_z=sigma_z,
            drift_length=drift_length_m,
            photon_energy_eV=wfr.photon_energy_eV,
            dx=wfr.x_step, dz=wfr.z_step,
            nx=wfr.nx, nz=wfr.nz,
        )
        
        # Stage 2: CNN refinement
        spatial = prepare_spatial_maps(wfr)
        prior_tensor = prior.to_tensor()
        
        with torch.no_grad():
            output = self.model(
                spatial.unsqueeze(0),
                prior_tensor.unsqueeze(0),
                {}
            )
        
        # Parse AT decision
        at_probs = F.softmax(output['at_logits'][0], dim=0)
        at_decision = at_probs.argmax().item()
        if at_decision == 0:
            final_AT = prior.AT  # agree with analytical
        elif at_decision == 1:
            final_AT = 0
        else:
            final_AT = 1
        
        # Apply resize corrections
        delta = output['resize_delta'][0].numpy()
        final_pxm = prior.pxm * np.exp(delta[0])
        final_pxd = prior.pxd * np.exp(delta[1])
        final_pzm = prior.pzm * np.exp(delta[2])
        final_pzd = prior.pzd * np.exp(delta[3])
        
        confidence = output['confidence'][0].item()
        
        return PredictionResult(
            # Final parameters
            analyt_treat=final_AT,
            pxm=round(final_pxm, 2),
            pxd=round(final_pxd, 2),
            pzm=round(final_pzm, 2),
            pzd=round(final_pzd, 2),
            confidence=confidence,
            
            # Diagnostics
            analytical_suggestion=prior,
            corrections_applied={
                'AT_override': at_decision != 0,
                'delta_log_pxm': delta[0],
                'delta_log_pxd': delta[1],
                'delta_log_pzm': delta[2],
                'delta_log_pzd': delta[3],
            },
            
            # Attention map (which patches drove the correction)
            # Useful for debugging — shows which part of the beam
            # caused the CNN to deviate from the analytical estimate
            # patch_attention_weights=attn_weights,
        )
```

## 8. What the Model Learns — Concrete Examples

### Clean Gaussian beam
```
Prior: AT=1, pxm=1.5, pzm=1.5, pxd=1.0, pzd=1.0
Spatial maps: smooth intensity, clean gradients, high sampling quality
CNN output: agree with prior, delta ≈ [0, 0, 0, 0], confidence=0.95
Final: same as prior
```

### Zone plate with 3rd order
```
Prior: AT=1, pxm=1.5, pzm=1.5 (based on 1st order)
Spatial maps: ring structure in intensity, complex gradient pattern
CNN output: agree AT, delta ≈ [+0.4, 0, +0.4, 0], confidence=0.80
Final: AT=1, pxm=2.2, pzm=2.2 (expanded for 3rd order)
Attention: highest weight on outer ring patches
```

### KB mirror with figure errors
```
Prior: AT=4, pxm=1.0, pzm=1.0 (propagation to waist)
Spatial maps: speckle in intensity, noisy gradients locally
CNN output: agree AT, delta ≈ [0, +0.3, 0, +0.3], confidence=0.75
Final: AT=4, pxd=1.35, pzd=1.35 (more resolution for speckle)
Attention: highest weight on patches with worst speckle
```

### Wavefront where analytical R is wrong
```
Prior: AT=1, pxm=2.0 (based on stated R=5m)
Spatial maps: sampling quality map inconsistent with R=5m prediction
CNN output: override to AT=0, delta ≈ [-0.3, +0.2, -0.3, +0.2], confidence=0.65
Final: AT=0, pxm=1.5, pxd=1.2 (safer standard propagation)
Low confidence → flag for user review
```

## 9. Implementation Roadmap

### Phase 1: PyTorch model + training loop (1 week)
- WavefrontViT model definition
- prepare_spatial_maps with physics normalisation
- Training loop with synthetic data + toy propagator
- Validate architecture trains and converges

### Phase 2: SRW integration + real training data (2–3 weeks)
- Wire wavefront generators to actual SRW
- Generate 10K+ labeled examples with SRW propagation + validator
- Include all 5 wavefront types
- Train and evaluate on held-out SRW wavefronts

### Phase 3: Analytical estimator + two-stage pipeline (1 week)
- Implement Stage 1 analytical estimator (from ABCD matrix trace)
- Connect Stage 1 output to ViT prior token
- Retrain with prior-aware labels (delta targets)
- Evaluate improvement over analytical-only baseline

### Phase 4: Deployment + feedback (1 week)
- Export model (TorchScript / ONNX)
- Package with SRW Python interface integration
- Attention visualisation for debugging
- Feedback logging for continuous improvement
