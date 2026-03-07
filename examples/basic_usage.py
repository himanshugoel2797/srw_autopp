"""
Basic usage example for the SRW Parameter Advisor.

Demonstrates:
  1. Analytical parameter estimation
  2. Post-propagation validation
  3. RL agent prediction (with untrained agent)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from srw_param_advisor.wavefront import WavefrontSnapshot
from srw_param_advisor.analytical import AnalyticalDriftEstimator
from srw_param_advisor.validator import PropagationValidator


def make_test_wavefront():
    """Create a simple Gaussian test wavefront."""
    nx, nz = 256, 256
    dx, dz = 1e-6, 1e-6
    energy = 12000.0  # 12 keV
    R_x, R_z = 20.0, 20.0
    sigma_x, sigma_z = 80e-6, 40e-6

    x = (np.arange(nx) - nx // 2) * dx
    z = (np.arange(nz) - nz // 2) * dz
    X, Z = np.meshgrid(x, z)

    k = 2 * np.pi / (1.239842e-06 / energy)
    amp = np.exp(-X**2 / (2 * sigma_x**2) - Z**2 / (2 * sigma_z**2))
    phase = (k / (2 * R_x)) * X**2 + (k / (2 * R_z)) * Z**2
    E = amp * np.exp(1j * phase)

    return WavefrontSnapshot(
        Ex=E, Ez=np.zeros_like(E),
        x_start=x[0], x_step=dx, z_start=z[0], z_step=dz,
        nx=nx, nz=nz, photon_energy_eV=energy,
        Robs_x=R_x, Robs_z=R_z,
    )


def main():
    print("=" * 60)
    print("SRW Parameter Advisor — Basic Usage Example")
    print("=" * 60)

    wfr = make_test_wavefront()
    drift_length = 5.0  # metres

    # --- Stage 1: Analytical estimation ---
    print("\n1. Analytical Parameter Estimation")
    print("-" * 40)

    estimator = AnalyticalDriftEstimator()
    estimate = estimator.estimate(
        R_x=wfr.Robs_x, R_z=wfr.Robs_z,
        sigma_x=80e-6, sigma_z=40e-6,
        drift_length=drift_length,
        photon_energy_eV=wfr.photon_energy_eV,
        dx=wfr.x_step, dz=wfr.z_step,
        nx=wfr.nx, nz=wfr.nz,
    )

    print(f"  AnalTreatment: {estimate.AT}")
    print(f"  pxm: {estimate.pxm}, pzm: {estimate.pzm}")
    print(f"  pxd: {estimate.pxd}, pzd: {estimate.pzd}")
    print(f"  Expected quality: {estimate.expected_quality}")

    # --- Build SRW prop params from estimate ---
    print(f"\n  SRW prop params list:")
    params = [0] * 17
    params[0] = 1  # auto resize before
    params[1] = 1  # auto resize after
    params[2] = 1.0
    params[3] = estimate.AT
    params[11] = estimate.pxm
    params[12] = estimate.pxd
    params[13] = estimate.pzm
    params[14] = estimate.pzd
    print(f"  {params}")

    # --- Stage 2: Validation (after propagation) ---
    print("\n2. Post-Propagation Validation")
    print("-" * 40)
    print("  (Simulating propagation with toy angular-spectrum method)")

    # Simple angular-spectrum propagation for demo
    wavelength = wfr.wavelength
    k = 2 * np.pi / wavelength
    qx = np.fft.fftfreq(wfr.nx, wfr.x_step)
    qz = np.fft.fftfreq(wfr.nz, wfr.z_step)
    Qx, Qz = np.meshgrid(qx, qz)
    qr2 = Qx**2 + Qz**2
    lam_qr2 = wavelength**2 * qr2
    mask = lam_qr2 < 1.0
    phase_tf = np.zeros_like(qr2)
    phase_tf[mask] = k * drift_length * (np.sqrt(1.0 - lam_qr2[mask]) - 1.0)
    transfer = np.exp(1j * phase_tf) * mask

    Ex_prop = np.fft.ifft2(np.fft.fft2(wfr.Ex) * transfer)
    wfr_after = WavefrontSnapshot(
        Ex=Ex_prop, Ez=np.zeros_like(Ex_prop),
        x_start=wfr.x_start, x_step=wfr.x_step,
        z_start=wfr.z_start, z_step=wfr.z_step,
        nx=wfr.nx, nz=wfr.nz,
        photon_energy_eV=wfr.photon_energy_eV,
    )

    validator = PropagationValidator()
    report = validator.validate(wfr, wfr_after)
    print(f"\n{report}")

    # --- Pre-flight check ---
    print("\n3. Pre-Flight Check")
    print("-" * 40)
    preflight = validator.preflight_check(
        wfr, drift_length=drift_length,
        prop_params={'pxm': estimate.pxm, 'pzm': estimate.pzm}
    )
    print(f"{preflight}")


if __name__ == "__main__":
    main()
