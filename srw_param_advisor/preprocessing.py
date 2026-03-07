"""
Physics-normalised spatial map preparation for the ViT encoder.

All phase-gradient channels are expressed in Nyquist-fraction units,
making the CNN/ViT scale-invariant across energies and grid sizes.

Channels:
  0: Normalised intensity |E|²/max → [0, 1]
  1: θ_x / θ_nyquist clipped to [-1, +1]
  2: θ_z / θ_nyquist clipped to [-1, +1]
  3: Sampling quality map: 1 - |Δφ_max|/π per pixel
  4: Validity mask
"""

import numpy as np
from typing import Tuple

PATCH_SIZE = 128
N_CHANNELS = 5


def prepare_spatial_maps(wfr) -> np.ndarray:
    """
    Prepare 5-channel spatial maps in resolution-element units.

    Parameters
    ----------
    wfr : WavefrontSnapshot
        Input wavefront.

    Returns
    -------
    np.ndarray, shape (5, nz, nx), float32
    """
    Ix = np.sum(np.abs(wfr.Ex) ** 2)
    Iz = np.sum(np.abs(wfr.Ez) ** 2)
    E = wfr.Ex if Ix >= Iz else wfr.Ez

    intensity = np.abs(E) ** 2
    peak = intensity.max()
    if peak == 0:
        return np.zeros((N_CHANNELS, wfr.nz, wfr.nx), dtype=np.float32)

    lambda_m = wfr.wavelength
    k = 2 * np.pi / lambda_m

    # Ch0: normalised intensity
    ch0 = intensity / peak

    # Local phase gradients via complex conjugate trick (no unwrapping)
    dEdx = np.zeros_like(E)
    dEdx[:, 1:-1] = (E[:, 2:] - E[:, :-2]) / (2 * wfr.x_step)
    dEdx[:, 0] = (E[:, 1] - E[:, 0]) / wfr.x_step
    dEdx[:, -1] = (E[:, -1] - E[:, -2]) / wfr.x_step

    dEdz = np.zeros_like(E)
    dEdz[1:-1, :] = (E[2:, :] - E[:-2, :]) / (2 * wfr.z_step)
    dEdz[0, :] = (E[1, :] - E[0, :]) / wfr.z_step
    dEdz[-1, :] = (E[-1, :] - E[-2, :]) / wfr.z_step

    threshold = 0.01 * peak
    mask = intensity > threshold
    I_safe = np.where(mask, intensity, 1.0)

    theta_x = np.where(mask, np.imag(np.conj(E) * dEdx) / (k * I_safe), 0.0)
    theta_z = np.where(mask, np.imag(np.conj(E) * dEdz) / (k * I_safe), 0.0)

    # Normalise by Nyquist angular limit: θ_nyq = λ/(2·dx)
    nyq_x = lambda_m / (2 * wfr.x_step)
    nyq_z = lambda_m / (2 * wfr.z_step)
    ch1 = np.clip(theta_x / nyq_x, -1, 1) * mask
    ch2 = np.clip(theta_z / nyq_z, -1, 1) * mask

    # Ch3: sampling quality from inter-pixel phase change
    dphi_x = np.abs(np.angle(E[:, 1:] * np.conj(E[:, :-1])))
    dphi_z = np.abs(np.angle(E[1:, :] * np.conj(E[:-1, :])))
    dphi_x_full = np.pad(dphi_x, ((0, 0), (0, 1)), mode='edge')
    dphi_z_full = np.pad(dphi_z, ((0, 1), (0, 0)), mode='edge')
    dphi_max = np.maximum(dphi_x_full, dphi_z_full)
    ch3 = np.clip(1.0 - dphi_max / np.pi, 0, 1)

    # Ch4: validity mask (has signal AND not aliased)
    aliased = (np.abs(ch1) > 0.95) | (np.abs(ch2) > 0.95)
    ch4 = (mask & ~aliased).astype(np.float32)

    return np.stack([ch0, ch1, ch2, ch3, ch4], axis=0).astype(np.float32)


def extract_patches(spatial_maps: np.ndarray, patch_size: int = PATCH_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split spatial maps into non-overlapping patches.

    Parameters
    ----------
    spatial_maps : (C, H, W)
    patch_size : int

    Returns
    -------
    patches : (N, C, P, P)
    positions : (N, 4) — [x_frac, z_frac, x_edge, z_edge]
    """
    C, H, W = spatial_maps.shape
    P = patch_size

    pad_h = (P - H % P) % P if H % P != 0 else 0
    pad_w = (P - W % P) % P if W % P != 0 else 0
    if pad_h > 0 or pad_w > 0:
        spatial_maps = np.pad(spatial_maps, ((0, 0), (0, pad_h), (0, pad_w)),
                              mode='constant')

    _, Hp, Wp = spatial_maps.shape
    nh, nw = Hp // P, Wp // P
    N = nh * nw

    patches = (spatial_maps
               .reshape(C, nh, P, nw, P)
               .transpose(1, 3, 0, 2, 4)
               .reshape(N, C, P, P))

    positions = np.zeros((N, 4), dtype=np.float32)
    idx = 0
    for i in range(nh):
        for j in range(nw):
            positions[idx] = [
                (j + 0.5) / nw, (i + 0.5) / nh,
                j / max(nw - 1, 1), i / max(nh - 1, 1),
            ]
            idx += 1

    return patches, positions


def sinusoidal_position_encoding(positions: np.ndarray, D: int) -> np.ndarray:
    """
    Sinusoidal encoding of physical positions.

    Parameters
    ----------
    positions : (N, 4)
    D : int — embedding dimension

    Returns
    -------
    (N, D) encoding
    """
    N, n_coords = positions.shape
    d = D // (2 * n_coords)
    freqs = 2.0 ** np.linspace(0, 6, d)

    enc_parts = []
    for c in range(n_coords):
        coord = positions[:, c:c + 1]
        angles = coord * freqs[None, :] * 2 * np.pi
        enc_parts.append(np.sin(angles))
        enc_parts.append(np.cos(angles))

    enc = np.concatenate(enc_parts, axis=1)
    if enc.shape[1] < D:
        enc = np.pad(enc, ((0, 0), (0, D - enc.shape[1])))
    return enc[:, :D].astype(np.float32)
