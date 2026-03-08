"""
RL-Based SRW Propagation Parameter Optimizer
=============================================

Contextual bandit agent that learns to select propagation mode (AnalTreatment
0-5) and resize parameters (pxm, pxd, pzm, pzd) jointly, optimizing for
parameter stability and physical validity.

Architecture:
  - ViT-style encoder: 128×128 patches with statistical embedding + transformer
  - Mode-conditional policy: 6-class categorical over modes, each mode has
    its own Gaussian resize distribution
  - Value baseline for variance reduction
  - REINFORCE with entropy bonus

Reward: stability-based (no reference wavefront needed).  Quality and
sufficiency are treated as separate signals: validator quality is an
independent term, while stability margin (headroom above a threshold)
provides gradient across valid modes.  Cost is ratio-based (excess over
analytical baseline).  A supervised mode loss (imitation on the oracle)
supplements REINFORCE for faster convergence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import json
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from srw_param_advisor.wavefront import WavefrontSnapshot
from srw_param_advisor.validator import (
    generate_test_wavefront,
    _estimate_beam_sigma,
    PropagationValidator,
)
from srw_param_advisor.preprocessing import (
    prepare_spatial_maps,
    extract_patches,
    sinusoidal_position_encoding,
    PATCH_SIZE,
    N_CHANNELS,
)


# ============================================================================
# Constants
# ============================================================================

N_AUX = 12       # analytical prior scalars
N_MODES = 5      # AnalTreatment {0, 1, 2, 3, 4}
N_RESIZE = 4     # (pxm, pxd, pzm, pzd)
EMBED_DIM = 256  # embedding dimension (256 with PyTorch autograd)

MAX_SIZE = 8192  # maximum grid size in SRW (for stability penalty)


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    """Return the best available device (CUDA > CPU) unless explicitly given."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

MODE_NAMES = {
    0: "Standard angular",
    1: "Quad-phase (moment)",
    2: "Quad-phase (special)",
    3: "From waist",
    4: "To waist",
}


# ============================================================================
# Analytical prior preparation
# ============================================================================

def prepare_analytical_prior(
    wfr: WavefrontSnapshot, drift_length: float,
    R_x: Optional[float] = None, R_z: Optional[float] = None,
    sigma_x: Optional[float] = None, sigma_z: Optional[float] = None,
) -> np.ndarray:
    """Build the 12-element analytical prior vector for the agent."""
    Rx = R_x if R_x is not None else (wfr.Robs_x if wfr.Robs_x else 1e23)
    Rz = R_z if R_z is not None else (wfr.Robs_z if wfr.Robs_z else 1e23)
    L = drift_length
    lambda_m = wfr.wavelength

    I = wfr.intensity
    if sigma_x is None:
        sigma_x = _estimate_beam_sigma(I, axis=1) * wfr.x_step
    if sigma_z is None:
        sigma_z = _estimate_beam_sigma(I, axis=0) * wfr.z_step

    new_Rx, new_Rz = Rx + L, Rz + L
    grid_half_x = wfr.nx * wfr.x_step / 2
    grid_half_z = wfr.nz * wfr.z_step / 2
    phase_cycles_x = abs(grid_half_x**2 / (lambda_m * Rx)) if Rx != 0 else 0
    phase_cycles_z = abs(grid_half_z**2 / (lambda_m * Rz)) if Rz != 0 else 0
    waist_x = abs(new_Rx) < 0.3 * abs(Rx) if Rx != 0 else False
    waist_z = abs(new_Rz) < 0.3 * abs(Rz) if Rz != 0 else False

    if waist_x and waist_z:
        AT_suggest = 4
    elif phase_cycles_x > 2 or phase_cycles_z > 2:
        AT_suggest = 1
    else:
        AT_suggest = 0

    growth_x = abs(new_Rx / Rx) if Rx != 0 else 1.0
    growth_z = abs(new_Rz / Rz) if Rz != 0 else 1.0
    pxm_0 = max(1.15 * growth_x, 1.0) if growth_x > 1.15 else 1.0
    pzm_0 = max(1.15 * growth_z, 1.0) if growth_z > 1.15 else 1.0
    phase_pp_x = abs(L) * lambda_m / (2 * wfr.x_step**2 * wfr.nx)
    phase_pp_z = abs(L) * lambda_m / (2 * wfr.z_step**2 * wfr.nz)
    target = 0.5 * np.pi
    pxd_0 = max(phase_pp_x / target, 1.0) if AT_suggest == 0 else max(phase_pp_x / target * 0.3, 1.0)
    pzd_0 = max(phase_pp_z / target, 1.0) if AT_suggest == 0 else max(phase_pp_z / target * 0.3, 1.0)

    def signed_log(x):
        return np.sign(x) * np.log10(max(abs(x), 1e-20))

    return np.array([
        signed_log(Rx), signed_log(Rz),
        np.log10(max(sigma_x, 1e-20)), np.log10(max(sigma_z, 1e-20)),
        signed_log(-Rx), signed_log(-Rz),
        float(AT_suggest),
        np.log(max(pxm_0, 0.1)), np.log(max(pxd_0, 0.1)),
        np.log(max(pzm_0, 0.1)), np.log(max(pzd_0, 0.1)),
        signed_log(L),
    ], dtype=np.float32)


def get_analytical_params(prior: np.ndarray) -> dict:
    """Extract analytical parameter suggestions from prior vector."""
    return {
        'AT': int(prior[6]),
        'pxm': float(np.exp(prior[7])),
        'pxd': float(np.exp(prior[8])),
        'pzm': float(np.exp(prior[9])),
        'pzd': float(np.exp(prior[10])),
    }



# ============================================================================
# Patch feature extraction (for CNN pretraining)
# ============================================================================

# Feature names and count — kept in sync with compute_patch_features()
PATCH_FEATURE_NAMES = [
    'mean_intensity',       # mean of ch0 (normalised intensity)
    'max_intensity',        # max of ch0
    'std_intensity',        # std of ch0
    'fill_fraction',        # mean of ch4 (validity mask)
    'mean_sampling_quality',  # mean of ch3
    'min_sampling_quality',   # min of ch3 over valid pixels (or 1 if none)
    'gradient_energy',      # mean(ch1² + ch2²)
    'mean_abs_theta_x',     # mean |ch1| over valid pixels
    'mean_abs_theta_z',     # mean |ch2| over valid pixels
    'edge_energy_fraction', # intensity at 2-pixel border / total intensity
    'horizontal_asymmetry', # |mean(left) - mean(right)| / (mean + eps)
    'vertical_asymmetry',   # |mean(top) - mean(bottom)| / (mean + eps)
]
N_PATCH_FEATURES = len(PATCH_FEATURE_NAMES)


def compute_patch_features(patches: np.ndarray, device=None) -> np.ndarray:
    """
    Compute ground-truth feature targets from raw 5-channel patches.

    Fully vectorized PyTorch implementation — no Python loop over patches.
    Uses GPU if available for significant speedup on large batches.

    Parameters
    ----------
    patches : np.ndarray, shape (N, 5, P, P)
        Batch of patches with channels:
          0: normalised intensity [0, 1]
          1: θ_x / θ_nyquist [-1, 1]
          2: θ_z / θ_nyquist [-1, 1]
          3: sampling quality [0, 1]
          4: validity mask {0, 1}
    device : torch.device, optional
        Computation device. Auto-detects GPU if None.

    Returns
    -------
    features : np.ndarray, shape (N, N_PATCH_FEATURES), float32
        All features are normalised to roughly [0, 1] or [-1, 1].
    """
    dev = _get_device(device)
    t = torch.from_numpy(patches).to(dev, dtype=torch.float32)  # (N, 5, P, P)

    ch0 = t[:, 0]  # (N, P, P) intensity
    ch1 = t[:, 1]  # theta_x
    ch2 = t[:, 2]  # theta_z
    ch3 = t[:, 3]  # sampling quality
    ch4 = t[:, 4]  # validity mask

    N, P, _ = ch0.shape
    valid = ch4 > 0.5                          # (N, P, P)
    n_valid = valid.sum(dim=(1, 2))             # (N,)
    has_valid = n_valid > 0                     # (N,)

    feats = torch.zeros(N, N_PATCH_FEATURES, device=dev)

    # [0] mean intensity
    feats[:, 0] = ch0.mean(dim=(1, 2))
    # [1] max intensity
    feats[:, 1] = ch0.amax(dim=(1, 2))
    # [2] std intensity
    feats[:, 2] = ch0.std(dim=(1, 2))
    # [3] fill fraction
    feats[:, 3] = ch4.mean(dim=(1, 2))
    # [4] mean sampling quality
    feats[:, 4] = ch3.mean(dim=(1, 2))

    # [5] min sampling quality over valid pixels (1.0 if no valid pixels)
    ch3_masked = torch.where(valid, ch3, torch.tensor(float('inf'), device=dev))
    min_sq = ch3_masked.amin(dim=(1, 2))
    feats[:, 5] = torch.where(has_valid, min_sq, torch.ones(N, device=dev))

    # [6] gradient energy
    feats[:, 6] = (ch1 ** 2 + ch2 ** 2).mean(dim=(1, 2))

    # [7, 8] mean |theta_x|, |theta_z| over valid pixels
    ch1_abs_valid = (ch1.abs() * valid).sum(dim=(1, 2))
    ch2_abs_valid = (ch2.abs() * valid).sum(dim=(1, 2))
    n_valid_safe = n_valid.clamp(min=1).float()
    feats[:, 7] = torch.where(has_valid, ch1_abs_valid / n_valid_safe, torch.zeros(N, device=dev))
    feats[:, 8] = torch.where(has_valid, ch2_abs_valid / n_valid_safe, torch.zeros(N, device=dev))

    # [9] edge energy fraction: intensity in 2-pixel border / total
    border = torch.zeros(P, P, device=dev, dtype=torch.bool)
    border[:2, :] = True
    border[-2:, :] = True
    border[:, :2] = True
    border[:, -2:] = True
    border_sum = (ch0 * border.unsqueeze(0)).sum(dim=(1, 2))  # (N,)
    total_int = ch0.sum(dim=(1, 2))                            # (N,)
    feats[:, 9] = torch.where(total_int > 0, border_sum / total_int,
                              torch.zeros(N, device=dev))

    # [10] horizontal asymmetry
    mid_w = P // 2
    left_mean = ch0[:, :, :mid_w].mean(dim=(1, 2))
    right_mean = ch0[:, :, mid_w:].mean(dim=(1, 2))
    avg_int = ch0.mean(dim=(1, 2)) + 1e-8
    feats[:, 10] = (left_mean - right_mean).abs() / avg_int

    # [11] vertical asymmetry
    mid_h = P // 2
    top_mean = ch0[:, :mid_h, :].mean(dim=(1, 2))
    bot_mean = ch0[:, mid_h:, :].mean(dim=(1, 2))
    feats[:, 11] = (top_mean - bot_mean).abs() / avg_int

    return feats.cpu().numpy()


class PatchFeatureHead(nn.Module):
    """
    MLP head that predicts patch-level features from CNN embeddings.

    Used for self-supervised pretraining of the patch CNN encoder.
    """

    def __init__(self, D: int, n_features: int = N_PATCH_FEATURES):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_features),
        )

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_embeddings : (N, D) — output of patch_cnn

        Returns
        -------
        predicted_features : (N, n_features)
        """
        return self.head(patch_embeddings)


class CNNPretrainer:
    """
    Self-supervised pretraining of the patch CNN encoder.

    The CNN learns to predict physics-meaningful features of each patch
    (mean intensity, fill fraction, sampling quality, gradient energy, etc.)
    from its raw 5-channel input.  This gives the encoder a useful
    initialisation before RL fine-tuning.
    """

    def __init__(self, agent: 'BanditAgent', lr: float = 1e-3,
                 log_dir: Optional[str] = None,
                 device: Optional[torch.device] = None):
        self.device = _get_device(device)
        self.agent = agent
        self.feature_head = PatchFeatureHead(agent.D).to(self.device)

        # Move CNN to device
        self.agent.patch_cnn.to(self.device)

        # Only train the CNN encoder and the feature head
        self.params = list(agent.patch_cnn.parameters()) + \
                      list(self.feature_head.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.history: List[dict] = []

    def train(self, n_epochs: int = 50, samples_per_epoch: int = 64,
              verbose: bool = True,
              dataset: 'PrecomputedDataset | None' = None):
        """
        Pretrain the patch CNN on feature prediction.

        Parameters
        ----------
        n_epochs : int
            Number of pretraining epochs.
        samples_per_epoch : int
            Wavefronts to process per epoch (each yields multiple patches).
        verbose : bool
        dataset : PrecomputedDataset, optional
            If provided, wavefronts are drawn from the dataset.
        """
        dev = self.device
        rng = np.random.RandomState(123)
        dataset_iter = None
        if dataset is not None:
            dataset_iter = iter(dataset.iter_epoch(rng))

        self.agent.patch_cnn.train()
        self.feature_head.train()

        if verbose:
            n_cnn_params = sum(p.numel() for p in self.agent.patch_cnn.parameters())
            n_head_params = sum(p.numel() for p in self.feature_head.parameters())
            source = f"precomputed ({len(dataset)} samples)" if dataset else "on-the-fly"
            print(f"CNN pretraining: {n_epochs} epochs, "
                  f"{samples_per_epoch} samples/epoch, data={source}")
            print(f"  CNN params: {n_cnn_params:,}, head params: {n_head_params:,}")
            print(f"  Device: {dev}")
            print(f"  Features: {N_PATCH_FEATURES} targets: "
                  f"{', '.join(PATCH_FEATURE_NAMES)}")

        t0 = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_patches = 0
            per_feature_mse = np.zeros(N_PATCH_FEATURES)

            for _ in range(samples_per_epoch):
                # Get a wavefront
                if dataset_iter is not None:
                    try:
                        wfr, _L = next(dataset_iter)
                    except StopIteration:
                        dataset_iter = iter(dataset.iter_epoch(rng))
                        wfr, _L = next(dataset_iter)
                else:
                    grid_n = rng.choice([128, 256])
                    wfr, _L = generate_universal_wavefront(rng, nx=grid_n, nz=grid_n,
                                                           device=dev)

                spatial = prepare_spatial_maps(wfr)
                patches, _positions = extract_patches(spatial)

                # Compute ground-truth features (vectorized, on device)
                targets = compute_patch_features(patches, device=dev)  # (N, F)
                targets_t = torch.from_numpy(targets).to(dev)

                # Forward through CNN
                patches_t = torch.from_numpy(patches).to(dev)      # (N, C, P, P)
                embeddings = self.agent.patch_cnn(patches_t)  # (N, D)
                predictions = self.feature_head(embeddings)   # (N, F)

                # MSE loss
                loss = nn.functional.mse_loss(predictions, targets_t)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
                self.optimizer.step()

                n_patches = patches.shape[0]
                epoch_loss += loss.item() * n_patches
                epoch_patches += n_patches

                with torch.no_grad():
                    per_feat = (predictions - targets_t).pow(2).mean(dim=0).cpu().numpy()
                    per_feature_mse += per_feat * n_patches

            mean_loss = epoch_loss / max(epoch_patches, 1)
            per_feature_mse /= max(epoch_patches, 1)

            self.history.append({
                'epoch': epoch,
                'mean_loss': mean_loss,
                'n_patches': epoch_patches,
                'per_feature_mse': per_feature_mse.tolist(),
            })

            self.writer.add_scalar('pretrain/loss', mean_loss, epoch)
            for fi, fname in enumerate(PATCH_FEATURE_NAMES):
                self.writer.add_scalar(f'pretrain/mse_{fname}',
                                       per_feature_mse[fi], epoch)

            if verbose and (epoch + 1) % max(1, n_epochs // 20) == 0:
                elapsed = time.time() - t0
                worst_feat = PATCH_FEATURE_NAMES[np.argmax(per_feature_mse)]
                print(f"  epoch {epoch+1:4d}/{n_epochs} | loss={mean_loss:.5f} | "
                      f"patches={epoch_patches} | worst={worst_feat} "
                      f"({per_feature_mse.max():.5f}) | {elapsed:.0f}s")

        self.writer.flush()

        if verbose:
            print(f"\nCNN pretraining complete in {time.time()-t0:.1f}s")
            print(f"  Final loss: {self.history[-1]['mean_loss']:.5f}")
            print("  Per-feature MSE:")
            final_mse = self.history[-1]['per_feature_mse']
            for fi, fname in enumerate(PATCH_FEATURE_NAMES):
                print(f"    {fname:25s}: {final_mse[fi]:.5f}")


# ============================================================================
# Agent (PyTorch nn.Module)
# ============================================================================

class BanditAgent(nn.Module):
    """
    Contextual bandit agent with ViT encoder and mode-conditional policy.
    PyTorch implementation with autograd.
    """

    def __init__(self, D: int = EMBED_DIM, n_transformer_blocks: int = 2):
        super().__init__()
        self.D = D
        self.n_blocks = n_transformer_blocks

        # Patch embedding: CNN on raw (C, P, P) patches → D
        self.patch_cnn = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, 7, stride=4, padding=3),   # (32, 32, 32)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # (64, 16, 16)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),          # (128, 8, 8)
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                              # (128, 1, 1)
            nn.Flatten(),                                         # (128,)
            nn.Linear(128, D),
        )

        # Prior encoder: N_AUX → D
        self.prior_enc = nn.Sequential(
            nn.Linear(N_AUX, 32), nn.ReLU(),
            nn.Linear(32, D),
        )

        # Transformer encoder (pre-norm via norm_first=True)
        n_heads = max(1, D // 64)  # 64 dims per head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=4 * D,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_blocks)

        # Attention pool
        self.attn_pool = nn.Linear(D, 1)

        combined_dim = 3 * D

        # Policy shared trunk
        self.policy_shared = nn.Sequential(
            nn.Linear(combined_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )

        # Mode logits
        self.mode_head = nn.Linear(64, N_MODES)

        # Per-mode resize heads (mean + log_std for each resize param)
        self.resize_heads = nn.ModuleList([
            nn.Linear(64, N_RESIZE * 2) for _ in range(N_MODES)
        ])

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialise resize head log_std biases near 0 (unit std)
        for head in self.resize_heads:
            nn.init.zeros_(head.bias)

    @property
    def device(self) -> torch.device:
        """Return the device of the model parameters."""
        return next(self.parameters()).device

    def forward(self, spatial_maps: np.ndarray, prior_scalars: np.ndarray):
        """
        Full forward pass.

        Parameters
        ----------
        spatial_maps : np.ndarray, shape (C, H, W)
        prior_scalars : np.ndarray, shape (N_AUX,)

        Returns
        -------
        mode_logits : Tensor (N_MODES,)
        resize_params : list of N_MODES dicts, each with 'mean' and 'log_std' Tensors (N_RESIZE,)
        value : Tensor scalar
        combined : Tensor (3D,)
        """
        dev = self.device

        # Extract patches and embed via CNN
        patches, positions = extract_patches(spatial_maps)

        patches_t = torch.from_numpy(patches).to(dev)  # (N, C, P, P)
        patch_emb = self.patch_cnn(patches_t)           # (N, D)

        pos_enc = sinusoidal_position_encoding(positions, self.D)
        patch_emb = patch_emb + torch.from_numpy(pos_enc.astype(np.float32)).to(dev)

        # Prior token
        prior_t = torch.from_numpy(prior_scalars.astype(np.float32)).to(dev)
        prior_token = self.prior_enc(prior_t).unsqueeze(0)  # (1, D)

        # Sequence: [prior, patches…] → (1, N+1, D) for batch_first transformer
        sequence = torch.cat([prior_token, patch_emb], dim=0).unsqueeze(0)  # (1, N+1, D)
        h = self.transformer(sequence).squeeze(0)  # (N+1, D)

        prior_out = h[0]       # (D,)
        patches_out = h[1:]    # (N, D)

        # Attention-weighted pool
        attn_scores = self.attn_pool(patches_out).squeeze(-1)  # (N,)
        attn_weights = torch.softmax(attn_scores, dim=0)        # (N,)
        patch_attn = (patches_out * attn_weights.unsqueeze(-1)).sum(0)  # (D,)

        # Max pool
        patch_max = patches_out.max(0).values  # (D,)

        combined = torch.cat([prior_out, patch_attn, patch_max])  # (3D,)

        # Policy trunk
        h_policy = self.policy_shared(combined)  # (64,)
        mode_logits = self.mode_head(h_policy)   # (N_MODES,)

        # Per-mode resize distributions
        resize_params = []
        for m in range(N_MODES):
            out = self.resize_heads[m](h_policy)  # (N_RESIZE*2,)
            means = out[:N_RESIZE]
            log_stds = out[N_RESIZE:].clamp(-3, 1)
            resize_params.append({'mean': means, 'log_std': log_stds})

        value = self.value_head(combined).squeeze()  # scalar

        return mode_logits, resize_params, value

    def sample_action(self, spatial_maps: np.ndarray, prior_scalars: np.ndarray):
        """
        Sample mode and resize from the policy.

        Returns
        -------
        mode : int
        resize_deltas : np.ndarray (N_RESIZE,)
        log_prob : Tensor (for gradient computation)
        entropy : Tensor (for gradient computation)
        value : Tensor
        mode_probs : np.ndarray (N_MODES,)
        """
        mode_logits, resize_params, value = self.forward(spatial_maps, prior_scalars)

        mode_dist = Categorical(logits=mode_logits)
        mode_t = mode_dist.sample()
        mode = mode_t.item()

        rp = resize_params[mode]
        stds = rp['log_std'].exp()
        resize_dist = Normal(rp['mean'], stds)
        resize_deltas_t = resize_dist.rsample()

        log_prob = mode_dist.log_prob(mode_t) + resize_dist.log_prob(resize_deltas_t).sum()
        entropy = mode_dist.entropy() + resize_dist.entropy().sum()

        mode_probs = torch.softmax(mode_logits, dim=0).detach().cpu().numpy()
        resize_deltas = resize_deltas_t.detach().cpu().numpy()

        return mode, resize_deltas, log_prob, entropy, value, mode_probs

    @torch.no_grad()
    def deterministic_action(self, spatial_maps: np.ndarray, prior_scalars: np.ndarray):
        """For inference: highest-probability mode, mean resize."""
        mode_logits, resize_params, value = self.forward(spatial_maps, prior_scalars)

        mode_probs = torch.softmax(mode_logits, dim=0).cpu().numpy()
        mode = int(np.argmax(mode_probs))

        resize_deltas = resize_params[mode]['mean'].cpu().numpy()

        return mode, resize_deltas, float(value.item()), mode_probs, resize_params


# ============================================================================
# Environment: propagation + reward
# ============================================================================

def action_to_params(mode: int, resize_deltas: np.ndarray, prior: np.ndarray) -> dict:
    """Convert agent action to SRW propagation parameters."""
    ap = get_analytical_params(prior)
    return {
        'analyt_treat': mode,
        'pxm': float(np.clip(ap['pxm'] * np.exp(resize_deltas[0]), 0.3, 20.0)),
        'pxd': float(np.clip(ap['pxd'] * np.exp(resize_deltas[1]), 0.3, 10.0)),
        'pzm': float(np.clip(ap['pzm'] * np.exp(resize_deltas[2]), 0.3, 20.0)),
        'pzd': float(np.clip(ap['pzd'] * np.exp(resize_deltas[3]), 0.3, 10.0)),
    }


def apply_resize(wfr, pxm, pzm, pxd, pzd):
    """Simplified resize: pad/crop for range, adjust step for resolution."""
    Ex, Ez = wfr.Ex.copy(), wfr.Ez.copy()
    nx, nz = wfr.nx, wfr.nz
    dx, dz = wfr.x_step, wfr.z_step
    xs, zs = wfr.x_start, wfr.z_start

    # Range: pad or crop
    for axis in ['x', 'z']:
        pm = pxm if axis == 'x' else pzm
        n = nx if axis == 'x' else nz
        d = dx if axis == 'x' else dz

        new_n = max(int(np.round(n * pm)), 8)
        new_n = min(new_n, MAX_SIZE)
        if new_n % 2 != 0:
            new_n += 1

        if new_n != n:
            if axis == 'x':
                if new_n > nx:
                    p = new_n - nx
                    pl, pr = p // 2, p - p // 2
                    Ex = np.pad(Ex, ((0, 0), (pl, pr)))
                    Ez = np.pad(Ez, ((0, 0), (pl, pr)))
                    xs -= pl * dx
                else:
                    t = (nx - new_n) // 2
                    Ex, Ez = Ex[:, t:t + new_n], Ez[:, t:t + new_n]
                    xs += t * dx
                nx = new_n
            else:
                if new_n > nz:
                    p = new_n - nz
                    pt, pb = p // 2, p - p // 2
                    Ex = np.pad(Ex, ((pt, pb), (0, 0)))
                    Ez = np.pad(Ez, ((pt, pb), (0, 0)))
                    zs -= pt * dz
                else:
                    t = (nz - new_n) // 2
                    Ex, Ez = Ex[t:t + new_n, :], Ez[t:t + new_n, :]
                    zs += t * dz
                nz = new_n

    # Resolution: adjust step (actual interpolation in production)
    if pxd != 1.0:
        dx = dx / pxd
    if pzd != 1.0:
        dz = dz / pzd

    return WavefrontSnapshot(
        Ex=Ex, Ez=Ez, x_start=xs, x_step=dx, z_start=zs, z_step=dz,
        nx=nx, nz=nz, photon_energy_eV=wfr.photon_energy_eV,
        Robs_x=wfr.Robs_x, Robs_z=wfr.Robs_z)


def srw_propagate(wfr: WavefrontSnapshot, drift_length: float,
                   params: dict) -> WavefrontSnapshot:
    """
    Propagate a wavefront through a drift using SRW with the given parameters.

    Parameters
    ----------
    wfr : WavefrontSnapshot
        Input wavefront.
    drift_length : float
        Drift length in metres.
    params : dict
        Must contain 'analyt_treat', 'pxm', 'pxd', 'pzm', 'pzd'.
        analyt_treat: 0 = no semi-analytical treatment, 1 = allow quadratic
                      phase removal (standard for most cases).
        pxm/pzm: horizontal/vertical range magnification factors.
        pxd/pzd: horizontal/vertical resolution change factors.

    Returns
    -------
    WavefrontSnapshot
        Propagated wavefront.
    """
    from srwpy.srwlib import srwl, SRWLOptD, SRWLOptC

    srw_wfr = wfr.to_srw()

    o_nx = int(np.round(wfr.nx * params['pxm'] * params['pxd']))
    o_nz = int(np.round(wfr.nz * params['pzm'] * params['pzd']))
    if o_nx > MAX_SIZE or o_nz > MAX_SIZE:
        print (f"Warning: requested grid size ({o_nx} x {o_nz}) exceeds SRW limits, failing this propagation.")
        raise ValueError("Requested grid size exceeds SRW limits.")

    # SRW propagation parameter list (12 elements):
    #  [0]: Auto-Resize before (1=yes, 0=no)
    #  [1]: Auto-Resize after (1=yes, 0=no)
    #  [2]: Relative precision for auto-resizing (1.0 = nominal)
    #  [3]: Allow semi-analytical quadratic phase treatment (1=yes, 0=no)
    #  [4]: Resize on Fourier side using FFT (1=yes, 0=no)
    #  [5]: Horizontal range modification factor (multiplier)
    #  [6]: Horizontal resolution modification factor (multiplier)
    #  [7]: Vertical range modification factor (multiplier)
    #  [8]: Vertical resolution modification factor (multiplier)
    #  [9]: Wavefront shift type before resizing
    # [10]: New horizontal center after shift
    # [11]: New vertical center after shift
    pp = [0] * 12
    pp[0] = 0                         # no auto-resize before
    pp[1] = 0                         # no auto-resize after
    pp[2] = 1.0                       # precision (unused without auto-resize)
    pp[3] = params['analyt_treat']    # semi-analytical quadratic phase
    pp[4] = 0                         # resize in real space, not Fourier
    pp[5] = params['pxm']             # horizontal range factor
    pp[6] = params['pxd']             # horizontal resolution factor
    pp[7] = params['pzm']             # vertical range factor
    pp[8] = params['pzd']             # vertical resolution factor

    drift = SRWLOptD(drift_length)
    optBL = SRWLOptC([drift], [pp])
    srwl.PropagElecField(srw_wfr, optBL)

    return WavefrontSnapshot.from_srw(srw_wfr)


def complex_correlation(E1, E2):
    """
    |⟨E1|E2⟩|² / (⟨E1|E1⟩·⟨E2|E2⟩)
    Invariant to global phase. Returns ∈ [0, 1].
    """
    n1 = np.sum(np.abs(E1)**2)
    n2 = np.sum(np.abs(E2)**2)
    if n1 == 0 or n2 == 0:
        return 0.0
    inner = np.abs(np.sum(E1 * np.conj(E2)))**2
    return float(inner / (n1 * n2))


def compute_accuracy(result, reference):
    """
    Compare result against reference using complex field correlation.
    Both are interpolated onto a common grid.
    """
    x_min = max(result.x_start, reference.x_start)
    x_max_r = result.x_start + (result.nx - 1) * result.x_step
    x_max_ref = reference.x_start + (reference.nx - 1) * reference.x_step
    x_max = min(x_max_r, x_max_ref)

    z_min = max(result.z_start, reference.z_start)
    z_max_r = result.z_start + (result.nz - 1) * result.z_step
    z_max_ref = reference.z_start + (reference.nz - 1) * reference.z_step
    z_max = min(z_max_r, z_max_ref)

    if x_max <= x_min or z_max <= z_min:
        return 0.0

    dx = max(result.x_step, reference.x_step)
    dz = max(result.z_step, reference.z_step)
    nx = max(int((x_max - x_min) / dx), 4)
    nz = max(int((z_max - z_min) / dz), 4)

    x_common = np.linspace(x_min, x_max, nx)
    z_common = np.linspace(z_min, z_max, nz)

    E_res = _interpolate_field(result.Ex, result, x_common, z_common)
    E_ref = _interpolate_field(reference.Ex, reference, x_common, z_common)

    return complex_correlation(E_res, E_ref)


def _interpolate_field(E, wfr, x_target, z_target):
    """Interpolate complex field onto target grid."""
    from scipy.interpolate import RegularGridInterpolator

    x_src = wfr.x_start + np.arange(wfr.nx) * wfr.x_step
    z_src = wfr.z_start + np.arange(wfr.nz) * wfr.z_step

    X_t, Z_t = np.meshgrid(x_target, z_target)
    points = np.column_stack([Z_t.ravel(), X_t.ravel()])

    interp_re = RegularGridInterpolator((z_src, x_src), E.real,
                                         bounds_error=False, fill_value=0)
    interp_im = RegularGridInterpolator((z_src, x_src), E.imag,
                                         bounds_error=False, fill_value=0)

    E_out = interp_re(points) + 1j * interp_im(points)
    return E_out.reshape(len(z_target), len(x_target))


def compute_cost(params, analytical_params=None):
    """Computational cost — excess over analytical baseline if available.

    When *analytical_params* is provided the cost measures how much larger the
    chosen resize factors are compared to the minimum-necessary estimates from
    the analytical prior.  This avoids penalising modes that legitimately need
    larger grids (e.g. AT=0 on a high-curvature beam) — only *waste* is
    penalised.

    Falls back to the original ``log(product)`` cost when no baseline is given.
    """
    if analytical_params is not None:
        excess_pxm = max(params['pxm'] / max(analytical_params['pxm'], 0.1) - 1.0, 0.0)
        excess_pxd = max(params['pxd'] / max(analytical_params['pxd'], 0.1) - 1.0, 0.0)
        excess_pzm = max(params['pzm'] / max(analytical_params['pzm'], 0.1) - 1.0, 0.0)
        excess_pzd = max(params['pzd'] / max(analytical_params['pzd'], 0.1) - 1.0, 0.0)
        return float(excess_pxm + excess_pxd + excess_pzm + excess_pzd)
    factor = params['pxm'] * params['pxd'] * params['pzm'] * params['pzd']
    return float(np.log(max(factor, 0.1)))


# ============================================================================
# Stability-based reward (no reference wavefront needed)
# ============================================================================

_validator = PropagationValidator()


def _double_resize_params(params):
    """
    Return a copy of *params* with all resize factors doubled.

    The AnalTreatment mode is kept unchanged — only pxm, pxd, pzm, pzd are
    scaled so that the output grid has 2× the range and 2× the resolution
    on each axis.
    """
    return {
        'analyt_treat': params['analyt_treat'],
        'pxm': params['pxm'] * 2.0,
        'pxd': params['pxd'] * 2.0,
        'pzm': params['pzm'] * 2.0,
        'pzd': params['pzd'] * 2.0,
    }


def compute_stability_reward(
    wfr_before,
    result,
    result_doubled,
    params,
    lambda_cost=0.1,
    analytical_params=None,
    doubled_params=None,
    stability_threshold=0.95,
    alpha_margin=1.0,
):
    """
    Stability-based reward that requires no reference wavefront.

    The reward separates quality from sufficiency so that the policy sees a
    real gradient across valid modes:

    1. **Validator quality** (0–1): independent quality signal from the
       ``PropagationValidator``.
    2. **Stability margin**: ``stability - threshold`` measures headroom
       above the sufficiency bar.  Modes that barely pass get a tiny margin;
       modes with comfortable headroom get more.
    3. **Quality headroom** (optional): if *doubled_params* is provided the
       validator is also run on the doubled result and the quality
       improvement is rewarded as a small bonus.
    4. **Cost penalty**: ratio-based excess over the analytical baseline when
       available, otherwise ``log(product)``.

    Final reward::

        reward = quality + α * margin - λ * cost - grid_penalty + headroom_bonus

    Parameters
    ----------
    wfr_before : WavefrontSnapshot
        Input wavefront (before propagation).
    result : WavefrontSnapshot
        Propagation result with the original parameters.
    result_doubled : WavefrontSnapshot
        Propagation result with doubled resize parameters.
    params : dict
        The original propagation parameters (for cost computation).
    lambda_cost : float
        Weight of the computational-cost penalty (default 0.1).
    analytical_params : dict, optional
        Baseline parameters from the analytical prior.  When provided the
        cost term is ratio-based (excess only).
    doubled_params : dict, optional
        The actual doubled parameter dict.  When provided the validator is
        run on *result_doubled* too and a headroom bonus is computed.
    stability_threshold : float
        Threshold for the stability margin term (default 0.95).
    alpha_margin : float
        Weight of the stability margin term (default 1.0).

    Returns
    -------
    reward : float
    info : dict
        Breakdown with keys ``validator_quality``, ``validator_passed``,
        ``stability``, ``cost``, ``grid_penalty``, ``margin``,
        ``headroom_bonus``, ``reward``.
    """
    # --- 1. Validator quality (independent signal) ---
    report = _validator.validate(wfr_before, result, params)
    validator_quality = report.overall_quality

    # --- 2. Stability margin ---
    stability = compute_accuracy(result, result_doubled)
    margin = stability - stability_threshold  # can be negative

    # --- 3. Cost (ratio-based when analytical baseline available) ---
    cost = compute_cost(params, analytical_params)

    # --- 4. Quality headroom from doubled params ---
    headroom_bonus = 0.0
    if doubled_params is not None:
        try:
            report_doubled = _validator.validate(
                wfr_before, result_doubled, doubled_params)
            quality_margin = report_doubled.overall_quality - validator_quality
            headroom_bonus = max(quality_margin, 0.0) * 0.1
        except Exception:
            pass

    # --- 5. Grid-size penalty (check doubled size) ---
    # The stability test doubles all resize factors, so the doubled grid
    # is what must fit within MAX_SIZE.
    dbl_nx = int(np.round(wfr_before.nx * params['pxm'] * params['pxd'] * 2))
    dbl_nz = int(np.round(wfr_before.nz * params['pzm'] * params['pzd'] * 2))
    max_n = max(dbl_nx, dbl_nz)
    if max_n > MAX_SIZE:
        grid_penalty = 2.0 * (max_n / MAX_SIZE)
    else:
        grid_penalty = 0.0

    reward = (validator_quality
              + alpha_margin * margin
              - lambda_cost * cost
              - grid_penalty
              + headroom_bonus)

    info = {
        'validator_quality': validator_quality,
        'validator_passed': report.passed,
        'stability': stability,
        'cost': cost,
        'grid_penalty': grid_penalty,
        'margin': margin,
        'headroom_bonus': headroom_bonus,
        'reward': reward,
    }
    return reward, info


# ============================================================================
# Universal parametric source
# ============================================================================

def generate_universal_wavefront(rng, nx=256, nz=128, device=None):
    """
    Generate a wavefront from the universal parametric distribution.

    Uses PyTorch for heavy array computation and GPU if available.

    Guarantees:
      - Beam contained in grid (3*w < grid_half on each axis)
      - Phase properly sampled (max phase step < pi/2 per pixel)

    Parameters
    ----------
    rng : np.random.RandomState
        Random state for reproducible sampling.
    nx, nz : int
        Grid dimensions.
    device : torch.device, optional
        Device for tensor computation. Auto-detects GPU if None.
    """
    dev = _get_device(device)

    # --- Scalar parameter sampling (stays on CPU / Python) ---
    energy = 10 ** rng.uniform(2, 5)
    lambda_m = 1.239842e-06 / energy
    dx = 10 ** rng.uniform(-7, -4)
    dz = dx * 10 ** rng.uniform(-0.3, 0.3)

    k = 2 * np.pi / lambda_m
    grid_half_x = nx * dx / 2
    grid_half_z = nz * dz / 2

    p_x = float(np.clip(2.0 + abs(rng.standard_cauchy()) * 1.0, 1.0, 10.0))
    p_z = float(np.clip(2.0 + abs(rng.standard_cauchy()) * 1.0, 1.0, 10.0))

    contain_x = 9.21 ** (1.0 / p_x)
    contain_z = 9.21 ** (1.0 / p_z)
    fill = 10 ** rng.uniform(-0.5, 0.0)
    w_x = float(np.clip(fill * grid_half_x * 0.25, dx * 2, grid_half_x / contain_x))
    w_z = float(np.clip(w_x * 10 ** rng.uniform(-0.3, 0.3), dz * 2, grid_half_z / contain_z))

    R_min_x = 4 * contain_x * w_x * dx / lambda_m
    R_min_z = 4 * contain_z * w_z * dz / lambda_m

    if rng.random() < 0.1:
        R_x = rng.choice([-1, 1]) * 1e23
        R_z = rng.choice([-1, 1]) * 1e23
    else:
        log_R_min_x = max(np.log10(R_min_x), -0.5)
        log_R_min_z = max(np.log10(R_min_z), -0.5)
        R_x = rng.choice([-1, 1]) * 10 ** rng.uniform(log_R_min_x, 4)
        R_z = rng.choice([-1, 1]) * 10 ** rng.uniform(log_R_min_z, 4)

    # --- Grid computation on device (GPU if available) ---
    x_t = (torch.arange(nx, device=dev, dtype=torch.float64) - nx // 2) * dx
    z_t = (torch.arange(nz, device=dev, dtype=torch.float64) - nz // 2) * dz
    Z_t, X_t = torch.meshgrid(z_t, x_t, indexing='ij')

    xn = X_t / w_x
    zn = Z_t / w_z

    # Super-Gaussian amplitude
    amp = torch.exp(-0.5 * (xn.abs().pow(p_x) + zn.abs().pow(p_z)))

    # Ring modulation
    eta = 0.0 if rng.random() < 0.6 else rng.uniform(0.02, 0.4)
    if eta > 0:
        r_norm = torch.sqrt(xn ** 2 + zn ** 2)
        ring_period = 10 ** rng.uniform(-0.3, 0.5)
        amp = amp * (1.0 + eta * torch.cos(2 * np.pi * r_norm / ring_period))
        amp = torch.clamp(amp, min=0)

    # Phase: quadratic curvature
    phase = torch.zeros_like(X_t)
    if abs(R_x) < 1e20:
        phase = phase + (k / (2 * R_x)) * X_t ** 2
    if abs(R_z) < 1e20:
        phase = phase + (k / (2 * R_z)) * Z_t ** 2

    # Zernike aberrations
    if rng.random() < 0.3:
        r_pupil = torch.sqrt(xn ** 2 + zn ** 2)
        r_max = float(r_pupil.max().item()) or 1e-10
        rho = r_pupil / r_max
        theta = torch.atan2(zn, xn)
        if rng.random() < 0.5:
            phase = phase + rng.exponential(0.5) * (3 * rho ** 3 - 2 * rho) * torch.cos(theta)
        if rng.random() < 0.5:
            phase = phase + rng.exponential(0.5) * rho ** 2 * torch.cos(2 * theta)

    # Linear phase tilt + constant offset
    phase = phase + k * rng.normal(0, 1e-5) * X_t
    phase = phase + k * rng.normal(0, 1e-5) * Z_t
    phase = phase + rng.uniform(0, 2 * np.pi)

    # Complex field E = amp * exp(i * phase)
    E_real = amp * torch.cos(phase)
    E_imag = amp * torch.sin(phase)

    # Noise
    if rng.random() < 0.1:
        amp_max = float(amp.max().item())
        noise_scale = 10 ** rng.uniform(-3, -1) * amp_max
        noise_re = torch.tensor(rng.randn(nz, nx), device=dev, dtype=torch.float64)
        noise_im = torch.tensor(rng.randn(nz, nx), device=dev, dtype=torch.float64)
        E_real = E_real + noise_scale * noise_re
        E_imag = E_imag + noise_scale * noise_im

    # Apertures (~25%)
    if rng.random() < 0.25:
        if rng.random() < 0.5:
            frac_x = rng.uniform(0.3, 0.9)
            frac_z = rng.uniform(0.3, 0.9)
            half_ax = frac_x * grid_half_x
            half_az = frac_z * grid_half_z
            aperture = ((X_t.abs() <= half_ax) & (Z_t.abs() <= half_az)).to(E_real.dtype)
        else:
            frac_r = rng.uniform(0.3, 0.9)
            radius = frac_r * min(grid_half_x, grid_half_z)
            R_dist = torch.sqrt(X_t ** 2 + Z_t ** 2)
            aperture = (R_dist <= radius).to(E_real.dtype)
        E_real = E_real * aperture
        E_imag = E_imag * aperture

    L = rng.choice([-1, 1]) * 10 ** rng.uniform(-1, 2)

    # Transfer back to CPU numpy for WavefrontSnapshot
    E_np = (E_real.cpu().numpy() + 1j * E_imag.cpu().numpy())
    x_np = x_t.cpu().numpy()

    wfr = WavefrontSnapshot(
        Ex=E_np, Ez=np.zeros_like(E_np),
        x_start=float(x_np[0]), x_step=dx, z_start=float(z_t[0].cpu().item()), z_step=dz,
        nx=nx, nz=nz, photon_energy_eV=energy,
        Robs_x=R_x, Robs_z=R_z)

    return wfr, L


# ============================================================================
# Precomputed dataset
# ============================================================================

def _save_wavefront(path: Path, wfr: WavefrontSnapshot):
    """Save a WavefrontSnapshot to a .npz file."""
    np.savez(
        path,
        Ex_real=wfr.Ex.real, Ex_imag=wfr.Ex.imag,
        Ez_real=wfr.Ez.real, Ez_imag=wfr.Ez.imag,
        x_start=wfr.x_start, x_step=wfr.x_step,
        z_start=wfr.z_start, z_step=wfr.z_step,
        nx=wfr.nx, nz=wfr.nz,
        photon_energy_eV=wfr.photon_energy_eV,
        Robs_x=wfr.Robs_x if wfr.Robs_x is not None else np.nan,
        Robs_z=wfr.Robs_z if wfr.Robs_z is not None else np.nan,
    )


def _load_wavefront(path: Path) -> WavefrontSnapshot:
    """Load a WavefrontSnapshot from a .npz file."""
    d = np.load(path)
    Robs_x = float(d['Robs_x'])
    Robs_z = float(d['Robs_z'])
    return WavefrontSnapshot(
        Ex=d['Ex_real'] + 1j * d['Ex_imag'],
        Ez=d['Ez_real'] + 1j * d['Ez_imag'],
        x_start=float(d['x_start']), x_step=float(d['x_step']),
        z_start=float(d['z_start']), z_step=float(d['z_step']),
        nx=int(d['nx']), nz=int(d['nz']),
        photon_energy_eV=float(d['photon_energy_eV']),
        Robs_x=None if np.isnan(Robs_x) else Robs_x,
        Robs_z=None if np.isnan(Robs_z) else Robs_z,
    )


def _validate_input_wavefront(wfr):
    """Check that beam is contained in grid (edge intensity < 1% of peak)."""
    I_in = np.abs(wfr.Ex) ** 2
    peak_in = I_in.max()
    if peak_in == 0:
        return False, "zero intensity"
    edge_max = max(I_in[0, :].max(), I_in[-1, :].max(),
                   I_in[:, 0].max(), I_in[:, -1].max())
    if edge_max > 0.01 * peak_in:
        return False, f"beam clipped ({edge_max/peak_in:.1%} of peak)"
    return True, ""



def precompute_dataset(
    output_dir: str,
    n_samples: int = 500,
    grid_sizes: List[int] = None,
    seed: int = 42,
    verbose: bool = True,
    batch_size: int = 32,
) -> str:
    """
    Precompute a dataset of (wavefront, drift_length) tuples.

    No reference wavefronts are computed — the stability-based reward
    does not need them.  Only the input wavefront and drift length are
    saved, making dataset generation much faster.

    Each sample is saved as:
      output_dir/NNNN_wfr.npz   — input wavefront
    Plus a manifest: output_dir/manifest.json with drift lengths and metadata.

    Parameters
    ----------
    output_dir : str
        Directory to save the dataset.
    n_samples : int
        Number of samples to generate.
    grid_sizes : list of int
        Grid sizes to randomly choose from. Default [128, 256].
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress.
    batch_size : int
        Number of samples to process per batch. Default 32.

    Returns
    -------
    str : Path to the output directory.
    """
    from concurrent.futures import ThreadPoolExecutor

    if grid_sizes is None:
        grid_sizes = [128, 256]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    manifest = {'seed': seed, 'samples': []}
    n_saved = 0
    n_failed = 0
    t0 = time.time()

    # Thread pool for background I/O (saving .npz files)
    save_pool = ThreadPoolExecutor(max_workers=4)
    save_futures = []

    def _drain_completed_saves():
        """Remove completed futures to release memory held by their arguments."""
        nonlocal save_futures
        still_pending = []
        for f in save_futures:
            if f.done():
                f.result()  # raise if save failed
            else:
                still_pending.append(f)
        save_futures = still_pending

    for i in range(n_samples):
        grid_n = rng.choice(grid_sizes)
        wfr, L = generate_universal_wavefront(rng, nx=grid_n, nz=grid_n)

        ok, reason = _validate_input_wavefront(wfr)
        if not ok:
            n_failed += 1
            if verbose and n_failed <= 10:
                print(f"  Sample {i}: input {reason}, skipping")
            continue

        idx = f"{n_saved:05d}"
        wfr_path = out / f"{idx}_wfr.npz"
        save_futures.append(save_pool.submit(_save_wavefront, wfr_path, wfr))

        manifest['samples'].append({
            'index': n_saved,
            'drift_length': float(L),
            'grid_nx': int(grid_n),
            'grid_nz': int(grid_n),
        })
        n_saved += 1

        # Periodically drain completed saves to free memory
        if n_saved % batch_size == 0:
            _drain_completed_saves()

        if verbose and n_saved > 0 and n_saved % 50 == 0:
            elapsed = time.time() - t0
            rate = n_saved / elapsed if elapsed > 0 else 0
            print(f"  Generated {n_saved}/{n_samples} samples "
                  f"({n_failed} failed, {rate:.1f} samples/s)")

    # Wait for remaining saves to complete
    for f in save_futures:
        f.result()
    save_pool.shutdown(wait=True)

    manifest['n_samples'] = n_saved
    manifest['n_failed'] = n_failed
    with open(out / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    if verbose:
        rate = n_saved / elapsed if elapsed > 0 else 0
        print(f"Dataset saved to {out}: {n_saved} samples "
              f"({n_failed} failed) in {elapsed:.1f}s ({rate:.1f} samples/s)")

    return str(out)


class PrecomputedDataset:
    """
    Loads a precomputed dataset of wavefronts from disk for training.

    The stability-based reward does not require reference wavefronts, so
    the dataset only stores input wavefronts and drift lengths.

    Usage
    -----
        dataset = PrecomputedDataset("path/to/dataset")
        for wfr, L in dataset.iter_epoch(rng):
            ...
    """

    def __init__(self, dataset_dir: str):
        self.dir = Path(dataset_dir)
        with open(self.dir / 'manifest.json') as f:
            self.manifest = json.load(f)
        self.n_samples = self.manifest['n_samples']

    def __len__(self):
        return self.n_samples

    def load_sample(self, idx: int):
        """Load a single (wavefront, drift_length) tuple."""
        entry = self.manifest['samples'][idx]
        prefix = f"{entry['index']:05d}"
        wfr = _load_wavefront(self.dir / f"{prefix}_wfr.npz")
        return wfr, entry['drift_length']

    def iter_epoch(self, rng: np.random.RandomState = None):
        """Yield all samples in shuffled order."""
        indices = np.arange(self.n_samples)
        if rng is not None:
            rng.shuffle(indices)
        for idx in indices:
            yield self.load_sample(idx)


# ============================================================================
# Training loop
# ============================================================================

class BanditTrainer:
    """
    Trains the contextual bandit using REINFORCE with value baseline.

    Reward is stability-based: propagate with predicted parameters, validate
    with the PropagationValidator, then double all resize factors and check
    that the complex-field correlation stays high.  No reference wavefront
    is needed.
    """

    def __init__(self, agent: BanditAgent, lambda_cost=0.1, lr=3e-4,
                 entropy_coeff=0.01, log_dir=None, propagate_fn=None,
                 device: Optional[torch.device] = None):
        self.device = _get_device(device)
        self.agent = agent.to(self.device)
        self.lambda_cost = lambda_cost
        self.entropy_coeff = entropy_coeff
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.history = []
        self.writer = SummaryWriter(log_dir=log_dir)
        self.propagate_fn = propagate_fn or srw_propagate

    def train(self, n_episodes=200, batch_size=8, verbose=True,
              dataset: 'PrecomputedDataset | None' = None):
        """
        Main training loop.

        Parameters
        ----------
        n_episodes : int
            Number of training episodes.
        batch_size : int
            Episodes per gradient update.
        verbose : bool
        dataset : PrecomputedDataset, optional
            If provided, wavefronts and drift lengths are drawn from the
            precomputed dataset instead of being generated on the fly.
            References stored in the dataset are ignored — the stability-
            based reward is computed from scratch each time.
        """
        dev = self.device
        rng = np.random.RandomState(42)

        # Set up sample iterator from precomputed dataset if provided
        if dataset is not None:
            dataset_iter = iter(dataset.iter_epoch(rng))

        if verbose:
            n_params = sum(p.numel() for p in self.agent.parameters())
            source = f"precomputed ({len(dataset)} samples)" if dataset else "on-the-fly"
            print(f"Training contextual bandit: {n_episodes} episodes, "
                  f"batch={batch_size}, λ_cost={self.lambda_cost}, data={source}")
            print(f"Agent: D={self.agent.D}, {self.agent.n_blocks} transformer blocks, "
                  f"{n_params:,} parameters")
            print(f"Device: {dev}")
            print(f"Reward: quality + α·margin - λ·cost + headroom (separated quality/sufficiency)")

        t0 = time.time()

        for ep in range(0, n_episodes, batch_size):
            actual_batch = min(batch_size, n_episodes - ep)

            batch_rewards = []
            batch_stabilities = []
            batch_qualities = []
            batch_costs = []
            batch_modes = []
            batch_loss = torch.tensor(0.0, device=dev)
            n_valid = 0

            self.agent.train()
            self.optimizer.zero_grad()

            for b in range(actual_batch):
                if dataset is not None:
                    try:
                        wfr, L = next(dataset_iter)
                    except StopIteration:
                        dataset_iter = iter(dataset.iter_epoch(rng))
                        wfr, L = next(dataset_iter)
                else:
                    grid_n = rng.choice([128, 256])
                    wfr, L = generate_universal_wavefront(rng, nx=grid_n, nz=grid_n,
                                                          device=dev)

                spatial = prepare_spatial_maps(wfr)
                prior = prepare_analytical_prior(wfr, L)

                # Forward pass once to get all mode distributions
                mode_logits, resize_params_all, value = \
                    self.agent.forward(spatial, prior)
                mode_dist = Categorical(logits=mode_logits)

                # Analytical baseline for ratio-based cost
                ap = get_analytical_params(prior)

                # Explore ALL modes for this wavefront
                mode_rewards = []
                mode_infos = []
                for m in range(N_MODES):
                    rp = resize_params_all[m]
                    rd_m = rp['mean'].detach().cpu().numpy()
                    params_m = action_to_params(m, rd_m, prior)

                    # Check against doubled grid size before expensive
                    # propagation — the stability test doubles all resize
                    # factors, so the *doubled* grid is what must fit.
                    dbl_nx = int(np.round(wfr.nx * params_m['pxm'] * params_m['pxd'] * 2))
                    dbl_nz = int(np.round(wfr.nz * params_m['pzm'] * params_m['pzd'] * 2))
                    if dbl_nx > MAX_SIZE or dbl_nz > MAX_SIZE:
                        pen = 2.0 * (max(dbl_nx, dbl_nz) / MAX_SIZE)
                        mode_rewards.append(-pen)
                        mode_infos.append({'stability': 0.0,
                                           'validator_quality': 0.0,
                                           'cost': 0.0})
                        continue

                    try:
                        res_m = self.propagate_fn(wfr, L, params_m)
                        dbl_params_m = _double_resize_params(params_m)
                        res_m_dbl = self.propagate_fn(wfr, L, dbl_params_m)
                        rw_m, info_m = compute_stability_reward(
                            wfr, res_m, res_m_dbl, params_m,
                            self.lambda_cost,
                            analytical_params=ap,
                            doubled_params=dbl_params_m)
                        mode_rewards.append(rw_m)
                        mode_infos.append(info_m)
                    except Exception:
                        mode_rewards.append(0.0)
                        mode_infos.append({'stability': 0.0,
                                           'validator_quality': 0.0,
                                           'cost': 0.0})

                # Best-mode reward used as value target and supervised signal
                best_reward = max(mode_rewards)
                best_mode = int(np.argmax(mode_rewards))

                # Sample from the policy for gradient
                mode_t = mode_dist.sample()
                mode = mode_t.item()
                rp = resize_params_all[mode]
                stds = rp['log_std'].exp()
                resize_dist = Normal(rp['mean'], stds)
                resize_deltas_t = resize_dist.rsample()

                log_prob = (mode_dist.log_prob(mode_t)
                            + resize_dist.log_prob(resize_deltas_t).sum())
                entropy = mode_dist.entropy() + resize_dist.entropy().sum()

                # Use the sampled mode's actual reward for REINFORCE
                reward = mode_rewards[mode]

                reward_t = torch.tensor(reward, dtype=torch.float32, device=dev)

                # --- Fix 2: proper actor-critic ---
                # Value targets the sampled reward (standard actor-critic)
                advantage = reward_t - value.detach()
                policy_loss = -(advantage * log_prob)
                value_loss = 0.5 * (value - reward_t.detach()).pow(2)
                entropy_loss = -self.entropy_coeff * entropy

                # Supervised mode loss: push policy toward best mode
                # (imitation learning on the oracle — converges faster
                # than pure REINFORCE since we evaluate all modes)
                best_mode_t = torch.tensor(best_mode, dtype=torch.long,
                                           device=dev)
                supervised_loss = nn.CrossEntropyLoss()(
                    mode_logits.unsqueeze(0), best_mode_t.unsqueeze(0))
                # Scale by how much better the best mode is
                avg_mode_reward = float(np.mean(mode_rewards))
                best_advantage = max(best_reward - avg_mode_reward, 0.0)
                supervised_loss = supervised_loss * best_advantage

                info = mode_infos[mode] if mode_infos[mode] else {
                    'stability': 0.0,
                    'validator_quality': 0.0,
                    'cost': compute_cost(
                        action_to_params(mode,
                                         resize_params_all[mode]['mean']
                                         .detach().cpu().numpy(), prior), ap),
                    'reward': reward,
                }

                batch_loss = (batch_loss + policy_loss + value_loss
                              + entropy_loss + supervised_loss)
                n_valid += 1

                batch_rewards.append(reward)
                batch_stabilities.append(info['stability'])
                batch_qualities.append(info['validator_quality'])
                batch_costs.append(info['cost'])
                batch_modes.append(mode)

            if n_valid == 0:
                continue

            (batch_loss / n_valid).backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.optimizer.step()

            mean_reward = np.mean(batch_rewards)
            mean_stability = np.mean(batch_stabilities)
            mean_quality = np.mean(batch_qualities)
            mean_cost = np.mean(batch_costs)

            self.history.append({
                'episode': ep,
                'mean_reward': mean_reward,
                'mean_stability': mean_stability,
                'mean_validator_quality': mean_quality,
                'mean_cost': mean_cost,
                'mode_dist': np.bincount(batch_modes, minlength=N_MODES).tolist(),
            })

            step = ep // batch_size
            self.writer.add_scalar('train/reward', mean_reward, step)
            self.writer.add_scalar('train/stability', mean_stability, step)
            self.writer.add_scalar('train/validator_quality', mean_quality, step)
            self.writer.add_scalar('train/cost', mean_cost, step)
            self.writer.add_scalar('train/loss', (batch_loss / n_valid).item(), step)
            mode_counts = np.bincount(batch_modes, minlength=N_MODES)
            for i in range(N_MODES):
                self.writer.add_scalar(f'train/mode_{i}_count', mode_counts[i], step)

            if verbose and (ep + batch_size) % max(batch_size, 20) == 0:
                elapsed = time.time() - t0
                mode_counts = np.bincount(batch_modes, minlength=N_MODES)
                mode_str = " ".join(f"AT{i}:{c}" for i, c in enumerate(mode_counts) if c > 0)
                print(f"  ep {ep+actual_batch:4d} | R={mean_reward:+.3f} | "
                      f"stab={mean_stability:.3f} | qual={mean_quality:.3f} | "
                      f"cost={mean_cost:.2f} | modes: {mode_str} | {elapsed:.0f}s")

        self.writer.flush()

        if verbose:
            print(f"\nTraining complete in {time.time()-t0:.1f}s")


# ============================================================================
# Evaluation and prediction interface
# ============================================================================

@dataclass
class PredictionResult:
    analyt_treat: int
    mode_name: str
    pxm: float
    pxd: float
    pzm: float
    pzd: float
    confidence: float
    expected_reward: float
    mode_probabilities: Dict[str, float]
    analytical_suggestion: dict
    corrections: dict

    def to_srw_prop_params(self) -> list:
        """Build SRW propagation parameter list (12 elements)."""
        p = [0] * 12
        p[0] = 0              # no auto-resize before
        p[1] = 0              # no auto-resize after
        p[2] = 1.0            # precision
        p[3] = self.analyt_treat  # semi-analytical quadratic phase
        p[5] = self.pxm       # horizontal range factor
        p[6] = self.pxd       # horizontal resolution factor
        p[7] = self.pzm       # vertical range factor
        p[8] = self.pzd       # vertical resolution factor
        return p

    def __str__(self):
        lines = [
            f"Predicted: AnalTreatment={self.analyt_treat} ({self.mode_name})",
            f"  pxm={self.pxm:.3f}, pxd={self.pxd:.3f}, "
            f"pzm={self.pzm:.3f}, pzd={self.pzd:.3f}",
            f"  Confidence: {self.confidence:.0%}",
            f"  Expected reward: {self.expected_reward:.3f}",
            f"  Mode probabilities:",
        ]
        for name, prob in sorted(self.mode_probabilities.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            lines.append(f"    {name:30s} {prob:5.1%} {bar}")
        lines.append(f"  Analytical suggestion: AT={self.analytical_suggestion['AT']}, "
                     f"pxm={self.analytical_suggestion['pxm']:.2f}, "
                     f"pzm={self.analytical_suggestion['pzm']:.2f}")
        lines.append(f"  Corrections: Δlog_pxm={self.corrections['d_pxm']:+.3f}, "
                     f"Δlog_pzm={self.corrections['d_pzm']:+.3f}")
        return "\n".join(lines)


def predict(agent: BanditAgent, wfr: WavefrontSnapshot, drift_length: float,
            R_x=None, R_z=None) -> PredictionResult:
    """Full prediction pipeline."""
    agent.eval()
    spatial = prepare_spatial_maps(wfr)
    prior = prepare_analytical_prior(wfr, drift_length, R_x=R_x, R_z=R_z)
    ap = get_analytical_params(prior)

    mode, resize_deltas, value, mode_probs, _ = agent.deterministic_action(spatial, prior)

    params = action_to_params(mode, resize_deltas, prior)
    confidence = float(mode_probs[mode])

    return PredictionResult(
        analyt_treat=mode,
        mode_name=MODE_NAMES[mode],
        pxm=round(params['pxm'], 3),
        pxd=round(params['pxd'], 3),
        pzm=round(params['pzm'], 3),
        pzd=round(params['pzd'], 3),
        confidence=confidence,
        expected_reward=value,
        mode_probabilities={MODE_NAMES[i]: float(p) for i, p in enumerate(mode_probs)},
        analytical_suggestion=ap,
        corrections={
            'd_pxm': float(resize_deltas[0]),
            'd_pxd': float(resize_deltas[1]),
            'd_pzm': float(resize_deltas[2]),
            'd_pzd': float(resize_deltas[3]),
        },
    )


def predict_all_modes(agent: BanditAgent, wfr: WavefrontSnapshot,
                      drift_length: float, R_x=None, R_z=None,
                      propagate_fn=None, lambda_cost=0.1,
                      verbose=False) -> list:
    """
    Explore all modes and return a ranked list of PredictionResults.

    For each mode the agent's learned resize distribution is used (mean),
    the candidate is propagated and scored with the stability reward so
    that the caller can compare numerically-correct but differently-
    efficient solutions side by side.

    Parameters
    ----------
    agent : BanditAgent
    wfr : WavefrontSnapshot
    drift_length : float
    R_x, R_z : float, optional
        Radius-of-curvature overrides.
    propagate_fn : callable, optional
        Propagation function (defaults to ``srw_propagate``).
    lambda_cost : float
        Cost weight passed to ``compute_stability_reward``.
    verbose : bool

    Returns
    -------
    list of dict
        One entry per mode, sorted best-first by reward.  Each dict
        contains ``'prediction'`` (PredictionResult), ``'reward'``,
        ``'stability'``, ``'validator_quality'``, ``'cost'``,
        ``'grid_penalty'``, and ``'output_grid'`` (nx, nz tuple).
    """
    _propagate = propagate_fn or srw_propagate
    agent.eval()
    spatial = prepare_spatial_maps(wfr)
    prior = prepare_analytical_prior(wfr, drift_length, R_x=R_x, R_z=R_z)
    ap = get_analytical_params(prior)

    # Run a single forward pass to get per-mode resize distributions
    mode_logits, resize_params, value = agent.forward(spatial, prior)
    mode_probs = torch.softmax(mode_logits, dim=0).detach().cpu().numpy()

    results = []
    for mode in range(N_MODES):
        resize_deltas = resize_params[mode]['mean'].detach().cpu().numpy()
        params = action_to_params(mode, resize_deltas, prior)

        # Predicted output grid size
        o_nx = int(np.round(wfr.nx * params['pxm'] * params['pxd']))
        o_nz = int(np.round(wfr.nz * params['pzm'] * params['pzd']))

        pred = PredictionResult(
            analyt_treat=mode,
            mode_name=MODE_NAMES[mode],
            pxm=round(params['pxm'], 3),
            pxd=round(params['pxd'], 3),
            pzm=round(params['pzm'], 3),
            pzd=round(params['pzd'], 3),
            confidence=float(mode_probs[mode]),
            expected_reward=float(value.item()),
            mode_probabilities={MODE_NAMES[i]: float(p)
                                for i, p in enumerate(mode_probs)},
            analytical_suggestion=ap,
            corrections={
                'd_pxm': float(resize_deltas[0]),
                'd_pxd': float(resize_deltas[1]),
                'd_pzm': float(resize_deltas[2]),
                'd_pzd': float(resize_deltas[3]),
            },
        )

        entry = {
            'prediction': pred,
            'mode': mode,
            'mode_name': MODE_NAMES[mode],
            'output_grid': (o_nx, o_nz),
            'reward': None,
            'stability': None,
            'validator_quality': None,
            'cost': compute_cost(params, ap),
            'grid_penalty': 0.0,
        }

        # Check against doubled grid size — the stability test doubles
        # all resize factors, so skip early if even the original × 2
        # would exceed the cap.
        dbl_nx, dbl_nz = o_nx * 2, o_nz * 2
        if dbl_nx > MAX_SIZE or dbl_nz > MAX_SIZE:
            entry['reward'] = -2.0 * (max(dbl_nx, dbl_nz) / MAX_SIZE)
            entry['grid_penalty'] = -entry['reward']
            entry['stability'] = 0.0
            entry['validator_quality'] = 0.0
        else:
            try:
                res = _propagate(wfr, drift_length, params)
                dbl_params = _double_resize_params(params)
                res_dbl = _propagate(wfr, drift_length, dbl_params)
                reward, info = compute_stability_reward(
                    wfr, res, res_dbl, params, lambda_cost,
                    analytical_params=ap, doubled_params=dbl_params)
                entry['reward'] = reward
                entry['stability'] = info['stability']
                entry['validator_quality'] = info['validator_quality']
                entry['grid_penalty'] = info.get('grid_penalty', 0.0)
            except Exception as exc:
                entry['reward'] = 0.0
                entry['stability'] = 0.0
                entry['validator_quality'] = 0.0
                if verbose:
                    print(f"  Mode {mode} ({MODE_NAMES[mode]}): "
                          f"propagation failed — {exc}")

        results.append(entry)

    # Sort best-first by reward
    results.sort(key=lambda r: -(r['reward'] if r['reward'] is not None else -999))

    if verbose:
        print(f"All-mode exploration ({len(results)} modes):")
        print(f"  {'Mode':30s} {'Grid':>12s} {'Stab':>6s} "
              f"{'Qual':>6s} {'Cost':>7s} {'Pen':>6s} {'Reward':>8s}")
        for r in results:
            nx, nz = r['output_grid']
            print(f"  {r['mode_name']:30s} {nx:5d}x{nz:<5d} "
                  f"{r['stability'] or 0:6.3f} "
                  f"{r['validator_quality'] or 0:6.3f} "
                  f"{r['cost']:7.2f} "
                  f"{r['grid_penalty']:6.2f} "
                  f"{r['reward'] or 0:+8.3f}")

    return results


def evaluate(agent, n_test=30, verbose=True, writer=None, global_step=None,
             propagate_fn=None):
    """Evaluate agent vs analytical baseline using stability-based reward."""
    agent.eval()
    rng = np.random.RandomState(999)
    _propagate = propagate_fn or srw_propagate

    agent_rewards, baseline_rewards = [], []
    agent_stabs, baseline_stabs = [], []
    agent_quals, baseline_quals = [], []
    agent_costs, baseline_costs = [], []

    for i in range(n_test):
        wfr, L = generate_universal_wavefront(rng, nx=rng.choice([128, 256]),
                                                nz=rng.choice([128, 256]))

        # --- Analytical baseline ---
        prior = prepare_analytical_prior(wfr, L)
        ap = get_analytical_params(prior)

        # --- Agent prediction ---
        pred = predict(agent, wfr, L)
        pred_params = {'analyt_treat': pred.analyt_treat,
                       'pxm': pred.pxm, 'pxd': pred.pxd,
                       'pzm': pred.pzm, 'pzd': pred.pzd}
        try:
            res_a = _propagate(wfr, L, pred_params)
            dbl_pred = _double_resize_params(pred_params)
            res_a_dbl = _propagate(wfr, L, dbl_pred)
            reward_a, info_a = compute_stability_reward(
                wfr, res_a, res_a_dbl, pred_params,
                analytical_params=ap, doubled_params=dbl_pred)
        except Exception:
            reward_a = 0.0
            info_a = {'stability': 0.0, 'validator_quality': 0.0,
                      'cost': compute_cost(pred_params, ap)}

        # --- Baseline: analytical params, no correction ---
        base_params = {'analyt_treat': ap['AT'],
                       'pxm': ap['pxm'], 'pxd': ap['pxd'],
                       'pzm': ap['pzm'], 'pzd': ap['pzd']}
        try:
            res_b = _propagate(wfr, L, base_params)
            dbl_base = _double_resize_params(base_params)
            res_b_dbl = _propagate(wfr, L, dbl_base)
            reward_b, info_b = compute_stability_reward(
                wfr, res_b, res_b_dbl, base_params,
                analytical_params=ap, doubled_params=dbl_base)
        except Exception:
            reward_b = 0.0
            info_b = {'stability': 0.0, 'validator_quality': 0.0,
                      'cost': compute_cost(base_params, ap)}

        agent_rewards.append(reward_a)
        baseline_rewards.append(reward_b)
        agent_stabs.append(info_a['stability'])
        baseline_stabs.append(info_b['stability'])
        agent_quals.append(info_a['validator_quality'])
        baseline_quals.append(info_b['validator_quality'])
        agent_costs.append(info_a['cost'])
        baseline_costs.append(info_b['cost'])

    if verbose and agent_rewards:
        print(f"\nEvaluation ({len(agent_rewards)} test cases):")
        print(f"  {'':20s} {'Stability':>10s} {'Quality':>10s} {'Cost':>8s} {'Reward':>10s}")
        print(f"  {'Analytical baseline':20s} {np.mean(baseline_stabs):10.4f} "
              f"{np.mean(baseline_quals):10.4f} "
              f"{np.mean(baseline_costs):8.2f} "
              f"{np.mean(baseline_rewards):10.4f}")
        print(f"  {'RL agent':20s} {np.mean(agent_stabs):10.4f} "
              f"{np.mean(agent_quals):10.4f} "
              f"{np.mean(agent_costs):8.2f} "
              f"{np.mean(agent_rewards):10.4f}")
        imp = np.sum(np.array(agent_rewards) > np.array(baseline_rewards))
        print(f"  Agent beats baseline: {imp}/{len(agent_rewards)} cases")

    if writer is not None and agent_rewards:
        s = global_step if global_step is not None else 0
        writer.add_scalar('eval/agent_reward', np.mean(agent_rewards), s)
        writer.add_scalar('eval/baseline_reward', np.mean(baseline_rewards), s)
        writer.add_scalar('eval/agent_stability', np.mean(agent_stabs), s)
        writer.add_scalar('eval/agent_quality', np.mean(agent_quals), s)
        writer.add_scalar('eval/agent_cost', np.mean(agent_costs), s)
        win_rate = np.sum(np.array(agent_rewards) > np.array(baseline_rewards)) / len(agent_rewards)
        writer.add_scalar('eval/win_rate', win_rate, s)

    return agent_rewards, baseline_rewards


# ============================================================================
# Demo
# ============================================================================

def main():
    print("=" * 72)
    print("RL Contextual Bandit — SRW Propagation Parameter Optimizer (PyTorch)")
    print("=" * 72)
    print(f"Modes: {N_MODES}, Resize params: {N_RESIZE}, Embed dim: {EMBED_DIM}")
    print(f"Patch size: {PATCH_SIZE}×{PATCH_SIZE}")
    print()

    agent = BanditAgent(D=EMBED_DIM, n_transformer_blocks=2)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent parameters: {n_params:,}")

    # Demo forward pass
    print("\n— Forward pass test —")
    rng = np.random.RandomState(0)
    wfr, L = generate_universal_wavefront(rng, nx=256, nz=128)
    spatial = prepare_spatial_maps(wfr)
    prior = prepare_analytical_prior(wfr, L)
    print(f"Spatial maps: {spatial.shape}")
    print(f"Prior: {prior.shape}")

    patches, positions = extract_patches(spatial)
    print(f"Patches: {patches.shape} ({patches.shape[0]} patches of {PATCH_SIZE}×{PATCH_SIZE})")

    agent.eval()
    mode, resize_deltas, log_prob, entropy, value, mode_probs = agent.sample_action(spatial, prior)
    print(f"Sampled mode: AT={mode} ({MODE_NAMES[mode]})")
    print(f"Resize deltas: {resize_deltas}")
    print(f"Value: {value.item():.3f}")
    print(f"Mode probs: {dict(zip(range(N_MODES), [f'{p:.2f}' for p in mode_probs]))}")

    # Demo prediction
    print("\n— Prediction test —")
    pred = predict(agent, wfr, L)
    print(pred)

    # Training
    print("\n— Training (100 episodes) —")
    trainer = BanditTrainer(agent, lambda_cost=0.05, lr=1e-3, entropy_coeff=0.01)
    trainer.train(n_episodes=100, batch_size=4, verbose=True)

    # Evaluation
    print("\n— Evaluation —")
    evaluate(agent, n_test=20, verbose=True, writer=trainer.writer, global_step=100)

    # Final prediction demo
    print("\n— Post-training prediction —")
    test_cases = [
        ("Gaussian, moderate drift", dict(
            nx=256, nz=128, dx=1e-6, dz=1e-6,
            photon_energy_eV=12000, R_x=20, R_z=20,
            beam_sigma_x=80e-6, beam_sigma_z=40e-6), 5.0),
        ("Converging to waist", dict(
            nx=256, nz=256, dx=0.5e-6, dz=0.5e-6,
            photon_energy_eV=8000, R_x=-2.0, R_z=-2.0,
            beam_sigma_x=30e-6, beam_sigma_z=30e-6), 2.5),
        ("Large beam, long drift", dict(
            nx=256, nz=256, dx=5e-6, dz=5e-6,
            photon_energy_eV=500, R_x=50, R_z=50,
            beam_sigma_x=400e-6, beam_sigma_z=400e-6), 30.0),
    ]

    for name, params, L in test_cases:
        print(f"\n  {name}:")
        wfr = generate_test_wavefront(**params)
        pred = predict(agent, wfr, L)
        print(f"  {pred}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
