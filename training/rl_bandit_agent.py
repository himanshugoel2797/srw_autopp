"""
RL-Based SRW Propagation Parameter Optimizer
=============================================

Contextual bandit agent that learns to select propagation mode (AnalTreatment
0-5) and resize parameters (pxm, pxd, pzm, pzd) jointly, optimizing for
accuracy against a high-resolution reference while minimizing computational cost.

Architecture:
  - ViT-style encoder: 128×128 patches with statistical embedding + transformer
  - Mode-conditional policy: 6-class categorical over modes, each mode has
    its own Gaussian resize distribution
  - Value baseline for variance reduction
  - REINFORCE with entropy bonus

Reference wavefront: adaptive BPM (guaranteed correct, no parameters to tune).
Implemented in PyTorch with autograd.
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
)
from srw_param_advisor.preprocessing import (
    prepare_spatial_maps,
    extract_patches,
    sinusoidal_position_encoding,
    PATCH_SIZE,
    N_CHANNELS,
)
from training.adaptive_bpm import compute_reference_adaptive, batch_compute_references


# ============================================================================
# Constants
# ============================================================================

N_AUX = 12       # analytical prior scalars
N_MODES = 5      # AnalTreatment {0, 1, 2, 3, 4}
N_RESIZE = 4     # (pxm, pxd, pzm, pzd)
EMBED_DIM = 256  # embedding dimension (256 with PyTorch autograd)

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
        # Extract patches and embed via CNN
        patches, positions = extract_patches(spatial_maps)

        patches_t = torch.from_numpy(patches)  # (N, C, P, P)
        patch_emb = self.patch_cnn(patches_t)   # (N, D)

        pos_enc = sinusoidal_position_encoding(positions, self.D)
        patch_emb = patch_emb + torch.from_numpy(pos_enc.astype(np.float32))

        # Prior token
        prior_t = torch.from_numpy(prior_scalars.astype(np.float32))
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
# Environment: propagation + reference + reward
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


def compute_cost(params):
    """Computational cost relative to base grid."""
    factor = params['pxm'] * params['pxd'] * params['pzm'] * params['pzd']
    return float(np.log(max(factor, 0.1)))


def compute_reward(accuracy, cost, lambda_cost=0.1):
    """Reward = accuracy - λ·cost."""
    return accuracy - lambda_cost * cost


# ============================================================================
# Universal parametric source
# ============================================================================

def generate_universal_wavefront(rng, nx=256, nz=128):
    """
    Generate a wavefront from the universal parametric distribution.

    Guarantees:
      - Beam contained in grid (3*w < grid_half on each axis)
      - Phase properly sampled (max phase step < pi/2 per pixel)
    """
    energy = 10 ** rng.uniform(2, 5)
    lambda_m = 1.239842e-06 / energy
    dx = 10 ** rng.uniform(-7, -4)
    dz = dx * 10 ** rng.uniform(-0.3, 0.3)

    k = 2 * np.pi / lambda_m
    x = (np.arange(nx) - nx // 2) * dx
    z = (np.arange(nz) - nz // 2) * dz
    X, Z = np.meshgrid(x, z)

    grid_half_x = nx * dx / 2
    grid_half_z = nz * dz / 2

    p_x = np.clip(2.0 + abs(rng.standard_cauchy()) * 1.0, 1.0, 10.0)
    p_z = np.clip(2.0 + abs(rng.standard_cauchy()) * 1.0, 1.0, 10.0)

    # Beam width: cap so 1% containment fits inside the grid.
    # For super-Gaussian exp(-0.5*|x/w|^p), the 1% radius is
    # r_1pct = (2*ln(100))^(1/p) * w ≈ 9.21^(1/p) * w
    contain_x = 9.21 ** (1.0 / p_x)  # containment multiplier for x
    contain_z = 9.21 ** (1.0 / p_z)
    fill = 10 ** rng.uniform(-0.5, 0.0)
    w_x = np.clip(fill * grid_half_x * 0.25, dx * 2, grid_half_x / contain_x)
    w_z = np.clip(w_x * 10 ** rng.uniform(-0.3, 0.3), dz * 2, grid_half_z / contain_z)

    # Curvature: ensure phase step at beam edge is < pi/2 per pixel.
    # Phase gradient = k*x/R, step/pixel = k*x_edge*dx/R < pi/2
    # x_edge = contain * w, so |R| > 2*k*contain*w*dx / pi = 4*contain*w*dx / lambda
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

    xn, zn = X / w_x, Z / w_z
    amp = np.exp(-0.5 * (np.abs(xn)**p_x + np.abs(zn)**p_z))

    eta = 0.0 if rng.random() < 0.6 else rng.uniform(0.02, 0.4)
    if eta > 0:
        r_norm = np.sqrt(xn**2 + zn**2)
        ring_period = 10 ** rng.uniform(-0.3, 0.5)
        amp *= (1.0 + eta * np.cos(2 * np.pi * r_norm / ring_period))
        amp = np.maximum(amp, 0)

    phase = np.zeros_like(X)
    if abs(R_x) < 1e20:
        phase += (k / (2 * R_x)) * X**2
    if abs(R_z) < 1e20:
        phase += (k / (2 * R_z)) * Z**2

    if rng.random() < 0.3:
        r_pupil = np.sqrt(xn**2 + zn**2)
        r_max = max(r_pupil.max(), 1e-10)
        rho = r_pupil / r_max
        theta = np.arctan2(zn, xn)
        if rng.random() < 0.5:
            phase += rng.exponential(0.5) * (3 * rho**3 - 2 * rho) * np.cos(theta)
        if rng.random() < 0.5:
            phase += rng.exponential(0.5) * rho**2 * np.cos(2 * theta)

    phase += k * rng.normal(0, 1e-5) * X
    phase += k * rng.normal(0, 1e-5) * Z
    phase += rng.uniform(0, 2 * np.pi)

    E = amp * np.exp(1j * phase)

    if rng.random() < 0.1:
        E += 10 ** rng.uniform(-3, -1) * amp.max() * (rng.randn(nz, nx) + 1j * rng.randn(nz, nx))

    # Randomly apply apertures (~25% of samples)
    # Use smooth (tapered) edges to avoid hard discontinuities that would
    # introduce high-frequency ringing and aliasing artifacts.
    if rng.random() < 0.25:
        # Edge taper width: 3-10 pixels, expressed in physical units
        taper_px = rng.uniform(3, 10)

        if rng.random() < 0.5:
            # --- Rectangular aperture ---
            # Half-widths: between 30% and 90% of grid half-extent
            frac_x = rng.uniform(0.3, 0.9)
            frac_z = rng.uniform(0.3, 0.9)
            half_ax = frac_x * grid_half_x
            half_az = frac_z * grid_half_z
            taper_x = taper_px * dx
            taper_z = taper_px * dz

            # Smooth rectangular mask via product of 1D tanh edges
            rect_x = 0.5 * (np.tanh((half_ax - np.abs(X)) / taper_x) + 1.0)
            rect_z = 0.5 * (np.tanh((half_az - np.abs(Z)) / taper_z) + 1.0)
            aperture = rect_x * rect_z
        else:
            # --- Circular aperture ---
            # Radius: between 30% and 90% of the smaller grid half-extent
            frac_r = rng.uniform(0.3, 0.9)
            radius = frac_r * min(grid_half_x, grid_half_z)
            taper_r = taper_px * min(dx, dz)

            R_dist = np.sqrt(X**2 + Z**2)
            aperture = 0.5 * (np.tanh((radius - R_dist) / taper_r) + 1.0)

        E *= aperture

    L = rng.choice([-1, 1]) * 10 ** rng.uniform(-1, 2)

    wfr = WavefrontSnapshot(
        Ex=E, Ez=np.zeros_like(E),
        x_start=x[0], x_step=dx, z_start=z[0], z_step=dz,
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


def _validate_output(wfr, ref):
    """Check energy conservation and finite values of reference."""
    if not np.all(np.isfinite(ref.Ex)):
        return False, "non-finite values"
    I_in = np.abs(wfr.Ex) ** 2
    E_in_total = np.sum(I_in) * wfr.x_step * wfr.z_step
    I_out = np.abs(ref.Ex) ** 2
    E_out_total = np.sum(I_out) * ref.x_step * ref.z_step
    if E_in_total > 0:
        energy_err = abs(E_out_total - E_in_total) / E_in_total
        if energy_err > 0.05 or not np.isfinite(energy_err):
            return False, f"energy error {energy_err:.1%}"
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
    Precompute a dataset of (wavefront, reference, drift_length) tuples.

    Uses batched GPU propagation and threaded I/O for speed:
    - Wavefronts are generated in batches on CPU
    - References are computed in GPU batches (keeps GPU warm)
    - Saving is done in a background thread pool (overlaps with next batch)

    Each sample is saved as:
      output_dir/NNNN_wfr.npz   — input wavefront
      output_dir/NNNN_ref.npz   — reference (adaptive BPM) wavefront
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
        Number of samples to process per GPU batch. Default 32.

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

    # Process in batches for efficient GPU utilisation
    i = 0
    while i < n_samples:
        # --- Phase 1: Generate a batch of wavefronts on CPU ---
        batch_wfrs = []
        batch_Ls = []
        batch_grids = []

        for _ in range(min(batch_size, n_samples - i)):
            grid_n = rng.choice(grid_sizes)
            wfr, L = generate_universal_wavefront(rng, nx=grid_n, nz=grid_n)

            ok, reason = _validate_input_wavefront(wfr)
            if not ok:
                n_failed += 1
                if verbose and n_failed <= 10:
                    print(f"  Sample {i}: input {reason}, skipping")
                i += 1
                continue

            batch_wfrs.append(wfr)
            batch_Ls.append(L)
            batch_grids.append(grid_n)
            i += 1

        if not batch_wfrs:
            continue

        # --- Phase 2: Batch GPU propagation ---
        refs = batch_compute_references(batch_wfrs, batch_Ls, verbose=False)

        # --- Phase 3: Validate and save (I/O in background threads) ---
        for wfr, ref, L, grid_n in zip(batch_wfrs, refs, batch_Ls, batch_grids):
            if ref is None:
                n_failed += 1
                continue

            ok, reason = _validate_output(wfr, ref)
            if not ok:
                n_failed += 1
                if verbose and n_failed <= 10:
                    print(f"  Output {reason}, skipping")
                continue

            idx = f"{n_saved:05d}"
            wfr_path = out / f"{idx}_wfr.npz"
            ref_path = out / f"{idx}_ref.npz"

            # Save in background thread
            save_futures.append(save_pool.submit(_save_wavefront, wfr_path, wfr))
            save_futures.append(save_pool.submit(_save_wavefront, ref_path, ref))

            manifest['samples'].append({
                'index': n_saved,
                'drift_length': float(L),
                'grid_nx': int(grid_n),
                'grid_nz': int(grid_n),
            })
            n_saved += 1

        # Release memory: drain completed saves so their wavefront references
        # can be garbage-collected, and free GPU cache between batches
        _drain_completed_saves()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose and n_saved > 0 and (n_saved % 50 < batch_size or n_saved < batch_size + 1):
            elapsed = time.time() - t0
            rate = n_saved / elapsed if elapsed > 0 else 0
            print(f"  Precomputed {n_saved}/{n_samples} samples "
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
    Loads a precomputed dataset from disk for use in training.

    Usage
    -----
        dataset = PrecomputedDataset("path/to/dataset")
        for wfr, ref, L in dataset.iter_epoch(rng):
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
        """Load a single (wavefront, reference, drift_length) tuple."""
        entry = self.manifest['samples'][idx]
        prefix = f"{entry['index']:05d}"
        wfr = _load_wavefront(self.dir / f"{prefix}_wfr.npz")
        ref = _load_wavefront(self.dir / f"{prefix}_ref.npz")
        return wfr, ref, entry['drift_length']

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
    Reference wavefronts are computed via adaptive BPM (guaranteed correct).
    """

    def __init__(self, agent: BanditAgent, lambda_cost=0.1, lr=3e-4,
                 entropy_coeff=0.01, log_dir=None, propagate_fn=None):
        self.agent = agent
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
            If provided, samples are drawn from the precomputed dataset
            instead of being generated on the fly.
        """
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

        t0 = time.time()

        for ep in range(0, n_episodes, batch_size):
            actual_batch = min(batch_size, n_episodes - ep)

            batch_rewards = []
            batch_accuracies = []
            batch_costs = []
            batch_modes = []
            batch_loss = torch.tensor(0.0)
            n_valid = 0

            self.agent.train()
            self.optimizer.zero_grad()

            for b in range(actual_batch):
                if dataset is not None:
                    try:
                        wfr, ref, L = next(dataset_iter)
                    except StopIteration:
                        dataset_iter = iter(dataset.iter_epoch(rng))
                        wfr, ref, L = next(dataset_iter)
                else:
                    grid_n = rng.choice([128, 256])
                    wfr, L = generate_universal_wavefront(rng, nx=grid_n, nz=grid_n)

                    # Guaranteed-correct reference via adaptive BPM
                    try:
                        ref = compute_reference_adaptive(wfr, L)
                    except Exception:
                        continue

                spatial = prepare_spatial_maps(wfr)
                prior = prepare_analytical_prior(wfr, L)

                # Forward + sample
                mode, resize_deltas, log_prob, entropy, value, mode_probs = \
                    self.agent.sample_action(spatial, prior)

                params = action_to_params(mode, resize_deltas, prior)

                # Evaluate proposed parameters via propagation
                try:
                    result = self.propagate_fn(wfr, L, params)
                    acc = compute_accuracy(result, ref)
                except Exception:
                    acc = 0.0

                cst = compute_cost(params)
                reward = compute_reward(acc, cst, self.lambda_cost)
                reward_t = torch.tensor(reward, dtype=torch.float32)

                # REINFORCE loss with value baseline
                advantage = reward_t - value.detach()
                policy_loss = -(advantage * log_prob)
                value_loss = 0.5 * (value - reward_t).pow(2)
                entropy_loss = -self.entropy_coeff * entropy

                batch_loss = batch_loss + policy_loss + value_loss + entropy_loss
                n_valid += 1

                batch_rewards.append(reward)
                batch_accuracies.append(acc)
                batch_costs.append(cst)
                batch_modes.append(mode)

            if n_valid == 0:
                continue

            (batch_loss / n_valid).backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.optimizer.step()

            mean_reward = np.mean(batch_rewards)
            mean_acc = np.mean(batch_accuracies)
            mean_cost = np.mean(batch_costs)

            self.history.append({
                'episode': ep,
                'mean_reward': mean_reward,
                'mean_accuracy': mean_acc,
                'mean_cost': mean_cost,
                'mode_dist': np.bincount(batch_modes, minlength=N_MODES).tolist(),
            })

            step = ep // batch_size
            self.writer.add_scalar('train/reward', mean_reward, step)
            self.writer.add_scalar('train/accuracy', mean_acc, step)
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
                      f"acc={mean_acc:.3f} | cost={mean_cost:.2f} | "
                      f"modes: {mode_str} | {elapsed:.0f}s")

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


def evaluate(agent, n_test=30, verbose=True, writer=None, global_step=None):
    """Evaluate agent vs analytical baseline on new configurations."""
    agent.eval()
    rng = np.random.RandomState(999)

    agent_accs, baseline_accs = [], []
    agent_costs, baseline_costs = [], []

    for i in range(n_test):
        wfr, L = generate_universal_wavefront(rng, nx=rng.choice([128, 256]),
                                                nz=rng.choice([128, 256]))
        try:
            ref = compute_reference_adaptive(wfr, L)
        except Exception:
            continue

        # Agent prediction
        pred = predict(agent, wfr, L)
        pred_params = {'analyt_treat': pred.analyt_treat,
                       'pxm': pred.pxm, 'pxd': pred.pxd,
                       'pzm': pred.pzm, 'pzd': pred.pzd}
        try:
            res_a = srw_propagate(wfr, L, pred_params)
            acc_a = compute_accuracy(res_a, ref)
        except Exception:
            acc_a = 0.0
        cost_a = compute_cost(pred_params)

        # Baseline: analytical params, no correction
        prior = prepare_analytical_prior(wfr, L)
        ap = get_analytical_params(prior)
        base_params = {'analyt_treat': ap['AT'],
                       'pxm': ap['pxm'], 'pxd': ap['pxd'],
                       'pzm': ap['pzm'], 'pzd': ap['pzd']}
        try:
            res_b = srw_propagate(wfr, L, base_params)
            acc_b = compute_accuracy(res_b, ref)
        except Exception:
            acc_b = 0.0
        cost_b = compute_cost(ap)

        agent_accs.append(acc_a)
        baseline_accs.append(acc_b)
        agent_costs.append(cost_a)
        baseline_costs.append(cost_b)

    if verbose and agent_accs:
        print(f"\nEvaluation ({len(agent_accs)} test cases):")
        print(f"  {'':20s} {'Accuracy':>10s} {'Cost':>8s} {'Reward':>10s}")
        print(f"  {'Analytical baseline':20s} {np.mean(baseline_accs):10.4f} "
              f"{np.mean(baseline_costs):8.2f} "
              f"{np.mean([a - 0.1*c for a,c in zip(baseline_accs, baseline_costs)]):10.4f}")
        print(f"  {'RL agent':20s} {np.mean(agent_accs):10.4f} "
              f"{np.mean(agent_costs):8.2f} "
              f"{np.mean([a - 0.1*c for a,c in zip(agent_accs, agent_costs)]):10.4f}")
        imp = np.sum(np.array(agent_accs) > np.array(baseline_accs))
        print(f"  Agent beats baseline: {imp}/{len(agent_accs)} cases")

    if writer is not None and agent_accs:
        s = global_step if global_step is not None else 0
        writer.add_scalar('eval/agent_accuracy', np.mean(agent_accs), s)
        writer.add_scalar('eval/baseline_accuracy', np.mean(baseline_accs), s)
        writer.add_scalar('eval/agent_cost', np.mean(agent_costs), s)
        writer.add_scalar('eval/agent_reward',
                          np.mean([a - 0.1*c for a, c in zip(agent_accs, agent_costs)]), s)
        writer.add_scalar('eval/win_rate', np.sum(np.array(agent_accs) > np.array(baseline_accs)) / len(agent_accs), s)

    return agent_accs, baseline_accs


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
