"""
Adaptive BPM Drift Propagator (PyTorch / GPU-accelerated)
==========================================================

Free-space drift propagation using the angular spectrum method with
automatic grid range management. Uses PyTorch for GPU-accelerated FFT.

Algorithm:
  1. Analyse the input field's spectral bandwidth
  2. Estimate how much the beam will spread over the drift distance
  3. Pre-expand the grid to accommodate the spread
  4. Propagate in a SINGLE angular spectrum step (exact for free space)
  5. Downsample if the grid exceeds the cap

The angular spectrum method is exact for homogeneous free-space
propagation in a single step -- no multi-step iteration needed.
The only requirement is that the grid is large enough to contain
the output beam without aliasing.

Usage:
    from training.adaptive_bpm import adaptive_drift_propagate

    E_out, x_out, z_out = adaptive_drift_propagate(
        E_in, x_coords, z_coords,
        drift_length=5.0,
        photon_energy_eV=12000.0,
    )
"""

import numpy as np
import torch
from typing import Tuple


def _get_device():
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _build_transfer_function(nx, nz, dx, dz, dL, lambda_m, device):
    """
    Build the angular spectrum transfer function on the given device.

    Parameters
    ----------
    dL : float
        FULL propagation distance (not a sub-step).

    Returns a complex64 tensor of shape (nz, nx).
    """
    k = 2 * np.pi / lambda_m

    qx = torch.fft.fftfreq(nx, dx, device=device, dtype=torch.float32)
    qz = torch.fft.fftfreq(nz, dz, device=device, dtype=torch.float32)
    Qz, Qx = torch.meshgrid(qz, qx, indexing='ij')
    qr2 = Qx ** 2 + Qz ** 2

    lam_qr2 = lambda_m ** 2 * qr2
    propagating = lam_qr2 < 1.0

    transfer = torch.zeros(nz, nx, dtype=torch.complex64, device=device)

    # Propagating waves
    # Use identity sqrt(1-x) - 1 = -x / (sqrt(1-x) + 1) to avoid
    # catastrophic cancellation in fp32 for small spatial frequencies
    lam_qr2_prop = lam_qr2[propagating]
    prop_arg = torch.sqrt(1.0 - lam_qr2_prop)
    phase = k * dL * (-lam_qr2_prop / (prop_arg + 1.0))
    transfer[propagating] = torch.exp(1j * phase)

    # Evanescent waves
    evanescent = ~propagating
    if evanescent.any():
        decay = k * abs(dL) * torch.sqrt(lam_qr2[evanescent] - 1.0)
        transfer[evanescent] = torch.exp(-decay).to(torch.complex64)

    return transfer


def _estimate_spectral_bandwidth(E, dx, dz, energy_fraction=0.999):
    """
    Estimate the spectral bandwidth of the field using cumulative energy.

    Sorts spectral bins by |q| and finds the radius that contains
    energy_fraction of the total energy. This is robust for chirped beams
    where power is spread thinly across many frequency bins.

    Returns (q_max_x, q_max_z) -- the maximum spatial frequencies with
    significant energy, in cycles/metre.
    """
    E_ft = torch.fft.fft2(E)
    power = torch.abs(E_ft) ** 2

    nz, nx = E.shape
    qx = torch.fft.fftfreq(nx, dx, device=E.device, dtype=torch.float32)
    qz = torch.fft.fftfreq(nz, dz, device=E.device, dtype=torch.float32)

    # X bandwidth: marginal spectrum, sorted by |qx|
    power_x = power.sum(dim=0)  # shape (nx,)
    abs_qx = torch.abs(qx)
    order_x = torch.argsort(abs_qx)
    cum_x = torch.cumsum(power_x[order_x], dim=0)
    total_x = cum_x[-1]
    if total_x > 0:
        idx_x = torch.searchsorted(cum_x, energy_fraction * total_x).item()
        idx_x = min(idx_x, nx - 1)
        q_max_x = abs_qx[order_x[idx_x]].item()
    else:
        q_max_x = 0.0

    # Z bandwidth: marginal spectrum, sorted by |qz|
    power_z = power.sum(dim=1)  # shape (nz,)
    abs_qz = torch.abs(qz)
    order_z = torch.argsort(abs_qz)
    cum_z = torch.cumsum(power_z[order_z], dim=0)
    total_z = cum_z[-1]
    if total_z > 0:
        idx_z = torch.searchsorted(cum_z, energy_fraction * total_z).item()
        idx_z = min(idx_z, nz - 1)
        q_max_z = abs_qz[order_z[idx_z]].item()
    else:
        q_max_z = 0.0

    return q_max_x, q_max_z


def _pad_for_propagation(E, dx, dz, x_start, z_start, drift_length,
                          q_max_x, q_max_z, lambda_m, safety_factor=2.0,
                          max_n=8192):
    """
    Pre-expand the grid to accommodate beam spread during propagation.

    The beam spread is estimated from the spectral bandwidth:
        spread_x = |L| * lambda * q_max_x  (paraxial approximation)

    For wide-angle components (lambda*q close to 1), uses the exact formula:
        spread = |L| * sin(theta) = |L| * lambda * q / sqrt(1 - (lambda*q)^2)
    """
    nz, nx = E.shape
    L = abs(drift_length)

    # Compute beam spread from spectral bandwidth
    # Use exact formula: tan(theta) = lambda*q / sqrt(1 - (lambda*q)^2)
    lqx = lambda_m * q_max_x
    lqz = lambda_m * q_max_z

    if lqx < 1.0 and lqx > 0:
        spread_x = L * lqx / np.sqrt(1.0 - lqx ** 2)
    else:
        spread_x = 0.0

    if lqz < 1.0 and lqz > 0:
        spread_z = L * lqz / np.sqrt(1.0 - lqz ** 2)
    else:
        spread_z = 0.0

    # Number of extra pixels needed on each side
    pad_x = int(np.ceil(spread_x * safety_factor / dx))
    pad_z = int(np.ceil(spread_z * safety_factor / dz))

    new_nx = nx + 2 * pad_x
    new_nz = nz + 2 * pad_z

    # Round up to even numbers (good for FFT)
    if new_nx % 2 != 0:
        new_nx += 1
    if new_nz % 2 != 0:
        new_nz += 1

    # Cap at max grid size
    new_nx = min(new_nx, max_n)
    new_nz = min(new_nz, max_n)

    # Recompute actual padding after capping
    actual_pad_x = (new_nx - nx) // 2
    actual_pad_z = (new_nz - nz) // 2
    pad_x_right = new_nx - nx - actual_pad_x
    pad_z_bottom = new_nz - nz - actual_pad_z

    if actual_pad_x > 0 or actual_pad_z > 0:
        E = torch.nn.functional.pad(
            E, (actual_pad_x, pad_x_right, actual_pad_z, pad_z_bottom))
        x_start = x_start - actual_pad_x * dx
        z_start = z_start - actual_pad_z * dz

    return E, dx, dz, x_start, z_start


def _downsample_field(E, dx, dz, x_start, z_start, target_n, lambda_m):
    """
    Downsample the field by cropping high frequencies in Fourier space.
    Keeps the physical range but uses fewer grid points (coarser dx/dz).
    Only downsamples axes that exceed target_n.
    """
    nz, nx = E.shape
    new_nx = nx
    new_dx = dx
    new_nz = nz
    new_dz = dz

    if nx > target_n:
        new_nx = target_n
        if new_nx % 2 != 0:
            new_nx += 1
        new_dx = dx * nx / new_nx

    if nz > target_n:
        new_nz = target_n
        if new_nz % 2 != 0:
            new_nz += 1
        new_dz = dz * nz / new_nz

    if new_nx == nx and new_nz == nz:
        return E, dx, dz, x_start, z_start

    E_ft = torch.fft.fft2(E)
    E_ft_shifted = torch.fft.fftshift(E_ft)

    cz = nz // 2
    cx = nx // 2
    hz = new_nz // 2
    hx = new_nx // 2

    E_ft_cropped = E_ft_shifted[cz - hz:cz - hz + new_nz,
                                 cx - hx:cx - hx + new_nx]

    E_ft_cropped = torch.fft.ifftshift(E_ft_cropped)
    E_new = torch.fft.ifft2(E_ft_cropped)

    # Scale to preserve field amplitude
    E_new = E_new * (new_nx * new_nz) / (nx * nz)

    return E_new, new_dx, new_dz, x_start, z_start


def adaptive_drift_propagate(
    E_in: np.ndarray,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    drift_length: float,
    photon_energy_eV: float,
    energy_fraction: float = 0.999,
    max_grid: int = 8192,
    verbose: bool = False,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate a complex E-field through a free-space drift using the
    angular spectrum method. Single-step, exact, GPU-accelerated.

    Algorithm:
      1. Estimate spectral bandwidth of input field (cumulative energy)
      2. Pre-expand grid to accommodate beam spread
      3. Single angular spectrum propagation step (exact for free space)
      4. Downsample if grid exceeds max_grid

    Parameters
    ----------
    E_in : np.ndarray, complex, shape (nz, nx)
        Input electric field.
    x_coords : np.ndarray, shape (nx,)
        Horizontal coordinates (metres).
    z_coords : np.ndarray, shape (nz,)
        Vertical coordinates (metres).
    drift_length : float
        Drift space length (metres). Can be negative.
    photon_energy_eV : float
        Photon energy in eV.
    energy_fraction : float
        Fraction of spectral energy to include in bandwidth estimate.
        Default 0.999 (99.9%).
    max_grid : int
        Maximum grid size per axis.
    verbose : bool
        Print progress information.
    device : torch.device, optional
        Torch device to use. Default: CUDA if available, else CPU.

    Returns
    -------
    E_out : np.ndarray, complex, shape (nz_out, nx_out)
    x_out : np.ndarray, shape (nx_out,)
    z_out : np.ndarray, shape (nz_out,)
    """
    if device is None:
        device = _get_device()

    lambda_m = 1.239842e-06 / photon_energy_eV
    L = drift_length

    nz, nx = E_in.shape
    dx = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1e-6
    dz = float(z_coords[1] - z_coords[0]) if len(z_coords) > 1 else 1e-6
    x_start = float(x_coords[0])
    z_start = float(z_coords[0])

    if verbose:
        print(f"Angular spectrum drift propagation ({device}):")
        print(f"  Drift length: {L:.4g} m")
        print(f"  Wavelength: {lambda_m:.4g} m ({photon_energy_eV:.0f} eV)")
        print(f"  Input grid: {nx}x{nz}, dx={dx:.3g} m, dz={dz:.3g} m")

    E_gpu = torch.from_numpy(E_in.astype(np.complex64)).to(device)

    # Step 1: Estimate spectral bandwidth
    q_max_x, q_max_z = _estimate_spectral_bandwidth(
        E_gpu, dx, dz, energy_fraction=energy_fraction)

    if verbose:
        print(f"  Spectral bandwidth: q_max_x={q_max_x:.2e}, q_max_z={q_max_z:.2e} cycles/m")

    # Step 2: Pre-expand grid to accommodate beam spread
    E_padded, dx, dz, x_start, z_start = _pad_for_propagation(
        E_gpu, dx, dz, x_start, z_start, L,
        q_max_x, q_max_z, lambda_m, max_n=max_grid)

    nz_pad, nx_pad = E_padded.shape
    if verbose:
        print(f"  Padded grid: {nx_pad}x{nz_pad}")

    # Step 3: Single-step angular spectrum propagation (exact for free space)
    transfer = _build_transfer_function(nx_pad, nz_pad, dx, dz, L, lambda_m, device)
    E_ft = torch.fft.fft2(E_padded)
    E_ft *= transfer
    E_out = torch.fft.ifft2(E_ft)
    del E_ft, transfer, E_padded

    # Step 4: Downsample if needed
    E_out, dx, dz, x_start, z_start = _downsample_field(
        E_out, dx, dz, x_start, z_start, max_grid, lambda_m)

    nz_out, nx_out = E_out.shape
    E_out_np = E_out.cpu().numpy()
    x_out = x_start + np.arange(nx_out) * dx
    z_out = z_start + np.arange(nz_out) * dz

    if verbose:
        print(f"  Output grid: {nx_out}x{nz_out}, "
              f"x=[{x_out[0]:.4g}, {x_out[-1]:.4g}] m, "
              f"z=[{z_out[0]:.4g}, {z_out[-1]:.4g}] m")

    return E_out_np, x_out, z_out


# ============================================================================
# Convenience wrapper matching WavefrontSnapshot interface
# ============================================================================

def propagate_drift_adaptive(wfr, drift_length, verbose=False):
    """
    Propagate a WavefrontSnapshot through a drift using angular spectrum.

    Parameters
    ----------
    wfr : WavefrontSnapshot
    drift_length : float
    verbose : bool

    Returns
    -------
    WavefrontSnapshot with propagated field on adapted grid
    """
    from srw_param_advisor.wavefront import WavefrontSnapshot

    x_coords = wfr.x_coords
    z_coords = wfr.z_coords

    Ex_out, x_out, z_out = adaptive_drift_propagate(
        wfr.Ex, x_coords, z_coords,
        drift_length, wfr.photon_energy_eV,
        verbose=verbose,
    )

    if np.any(np.abs(wfr.Ez) > 0):
        Ez_out, _, _ = adaptive_drift_propagate(
            wfr.Ez, x_coords, z_coords,
            drift_length, wfr.photon_energy_eV,
            verbose=False,
        )
    else:
        Ez_out = np.zeros_like(Ex_out)

    nx_out = len(x_out)
    nz_out = len(z_out)
    dx_out = x_out[1] - x_out[0] if nx_out > 1 else wfr.x_step
    dz_out = z_out[1] - z_out[0] if nz_out > 1 else wfr.z_step

    return WavefrontSnapshot(
        Ex=Ex_out, Ez=Ez_out,
        x_start=x_out[0], x_step=dx_out,
        z_start=z_out[0], z_step=dz_out,
        nx=nx_out, nz=nz_out,
        photon_energy_eV=wfr.photon_energy_eV,
        Robs_x=(wfr.Robs_x + drift_length) if wfr.Robs_x is not None else None,
        Robs_z=(wfr.Robs_z + drift_length) if wfr.Robs_z is not None else None,
    )


# ============================================================================
# Reference computation for RL training
# ============================================================================

def compute_reference_adaptive(wfr, drift_length, verbose=False):
    """
    Compute a guaranteed-correct reference propagation using angular spectrum.

    Parameters
    ----------
    wfr : WavefrontSnapshot
    drift_length : float
    verbose : bool

    Returns
    -------
    WavefrontSnapshot -- the reference result
    """
    return propagate_drift_adaptive(wfr, drift_length, verbose=verbose)


def batch_compute_references(wfr_list, drift_lengths, device=None, verbose=False):
    """
    Compute reference propagations for a batch of wavefronts efficiently.

    Keeps the GPU warm by processing all samples back-to-back without
    returning to CPU between samples. Groups samples by grid size for
    potential future batched-FFT optimization.

    Parameters
    ----------
    wfr_list : list of WavefrontSnapshot
    drift_lengths : list of float
    device : torch.device, optional
    verbose : bool

    Returns
    -------
    list of (WavefrontSnapshot or None) -- None for failed samples
    """
    from srw_param_advisor.wavefront import WavefrontSnapshot

    if device is None:
        device = _get_device()

    # Pre-warm CUDA (first kernel launch has overhead)
    if device.type == 'cuda':
        torch.zeros(1, device=device)

    results = [None] * len(wfr_list)

    for i, (wfr, L) in enumerate(zip(wfr_list, drift_lengths)):
        try:
            x_coords = wfr.x_coords
            z_coords = wfr.z_coords
            dx = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1e-6
            dz = float(z_coords[1] - z_coords[0]) if len(z_coords) > 1 else 1e-6
            x_start = float(x_coords[0])
            z_start = float(z_coords[0])
            lambda_m = wfr.wavelength

            # Move to GPU once
            E_gpu = torch.from_numpy(wfr.Ex.astype(np.complex64)).to(device)

            # Spectral bandwidth
            q_max_x, q_max_z = _estimate_spectral_bandwidth(E_gpu, dx, dz)

            # Pad
            E_padded, dx_p, dz_p, xs_p, zs_p = _pad_for_propagation(
                E_gpu, dx, dz, x_start, z_start, L,
                q_max_x, q_max_z, lambda_m)
            del E_gpu

            # Propagate
            nz_pad, nx_pad = E_padded.shape
            transfer = _build_transfer_function(nx_pad, nz_pad, dx_p, dz_p, L, lambda_m, device)
            E_ft = torch.fft.fft2(E_padded)
            E_ft *= transfer
            E_out = torch.fft.ifft2(E_ft)
            del E_ft, transfer, E_padded

            # Downsample
            E_out, dx_o, dz_o, xs_o, zs_o = _downsample_field(
                E_out, dx_p, dz_p, xs_p, zs_p, 8192, lambda_m)

            # Copy back to CPU and free GPU memory
            nz_out, nx_out = E_out.shape
            E_out_np = E_out.cpu().numpy()
            del E_out
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            x_out = xs_o + np.arange(nx_out) * dx_o
            z_out = zs_o + np.arange(nz_out) * dz_o

            Ez_out = np.zeros_like(E_out_np)

            results[i] = WavefrontSnapshot(
                Ex=E_out_np, Ez=Ez_out,
                x_start=x_out[0], x_step=dx_o,
                z_start=z_out[0], z_step=dz_o,
                nx=nx_out, nz=nz_out,
                photon_energy_eV=wfr.photon_energy_eV,
                Robs_x=(wfr.Robs_x + L) if wfr.Robs_x is not None else None,
                Robs_z=(wfr.Robs_z + L) if wfr.Robs_z is not None else None,
            )
        except Exception as e:
            if verbose:
                print(f"  batch_compute_references: sample {i} failed: {e}")
            results[i] = None

    return results


# ============================================================================
# Demo / test
# ============================================================================

def _make_test_wavefront(nx=512, nz=256, dx=1e-6, dz=1e-6,
                          energy_eV=12000.0, R_x=10.0, R_z=10.0,
                          sigma_x=80e-6, sigma_z=40e-6):
    """Create a test Gaussian wavefront."""
    x = (np.arange(nx) - nx // 2) * dx
    z = (np.arange(nz) - nz // 2) * dz
    X, Z = np.meshgrid(x, z)

    k = 2 * np.pi / (1.239842e-06 / energy_eV)
    amp = np.exp(-X ** 2 / (2 * sigma_x ** 2) - Z ** 2 / (2 * sigma_z ** 2))
    phase = np.zeros_like(X)
    if abs(R_x) < 1e20:
        phase += (k / (2 * R_x)) * X ** 2
    if abs(R_z) < 1e20:
        phase += (k / (2 * R_z)) * Z ** 2

    E = amp * np.exp(1j * phase)
    return E, x, z


def main():
    import time

    print("=" * 65)
    print("Angular Spectrum Drift Propagator (PyTorch) -- Demo")
    print("=" * 65)
    print(f"Device: {_get_device()}")

    test_cases = [
        {
            'name': "Gaussian beam, moderate drift",
            'params': dict(nx=256, nz=128, dx=1e-6, dz=1e-6,
                           energy_eV=12000, R_x=20, R_z=20,
                           sigma_x=60e-6, sigma_z=30e-6),
            'drift': 3.0,
        },
        {
            'name': "Converging beam (to waist)",
            'params': dict(nx=256, nz=256, dx=0.5e-6, dz=0.5e-6,
                           energy_eV=8000, R_x=-2.0, R_z=-2.0,
                           sigma_x=30e-6, sigma_z=30e-6),
            'drift': 1.5,
        },
        {
            'name': "Heavily wrapped phase (small R)",
            'params': dict(nx=256, nz=256, dx=2e-6, dz=2e-6,
                           energy_eV=12000, R_x=3.0, R_z=3.0,
                           sigma_x=100e-6, sigma_z=100e-6),
            'drift': 5.0,
        },
        {
            'name': "Large beam, long drift",
            'params': dict(nx=256, nz=256, dx=5e-6, dz=5e-6,
                           energy_eV=500, R_x=50, R_z=50,
                           sigma_x=400e-6, sigma_z=400e-6),
            'drift': 20.0,
        },
    ]

    for tc in test_cases:
        print(f"\n{'--' * 32}")
        print(f"  {tc['name']}")
        print(f"{'--' * 32}")

        E_in, x_in, z_in = _make_test_wavefront(**tc['params'])

        dphi_x = np.abs(np.angle(E_in[:, 1:] * np.conj(E_in[:, :-1])))
        max_dphi = dphi_x.max() / np.pi
        print(f"  Input max |dphi|: {max_dphi:.3f}pi per pixel")

        t0 = time.time()
        E_out, x_out, z_out = adaptive_drift_propagate(
            E_in, x_in, z_in,
            drift_length=tc['drift'],
            photon_energy_eV=tc['params']['energy_eV'],
            verbose=True,
        )
        elapsed = time.time() - t0

        E_in_total = np.sum(np.abs(E_in) ** 2)
        E_out_total = np.sum(np.abs(E_out) ** 2) * (
            (x_out[1] - x_out[0]) * (z_out[1] - z_out[0]) /
            (x_in[1] - x_in[0]) / (z_in[1] - z_in[0])
        ) if len(x_out) > 1 else 0

        energy_change = abs(E_out_total - E_in_total) / max(E_in_total, 1e-30)

        dphi_x_out = np.abs(np.angle(E_out[:, 1:] * np.conj(E_out[:, :-1])))
        I_out = np.abs(E_out) ** 2
        mask = 0.5 * (I_out[:, 1:] + I_out[:, :-1]) > 0.01 * I_out.max()
        max_dphi_out = dphi_x_out[mask].max() / np.pi if mask.any() else 0

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Energy conservation: {energy_change:.2e} relative change")
        print(f"  Output grid: {len(x_out)}x{len(z_out)}")
        print(f"  Output max |dphi|: {max_dphi_out:.3f}pi per pixel")
        print(f"  Output range x: [{x_out[0]*1e3:.3f}, {x_out[-1]*1e3:.3f}] mm")
        print(f"  Output range z: [{z_out[0]*1e3:.3f}, {z_out[-1]*1e3:.3f}] mm")


if __name__ == "__main__":
    main()
