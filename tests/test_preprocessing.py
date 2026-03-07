"""Basic tests for the SRW Parameter Advisor."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from srw_param_advisor.wavefront import WavefrontSnapshot
from srw_param_advisor.preprocessing import (
    prepare_spatial_maps, extract_patches, sinusoidal_position_encoding,
    PATCH_SIZE, N_CHANNELS,
)
from srw_param_advisor.analytical import AnalyticalDriftEstimator


def _make_wavefront(nx=256, nz=128, R_x=10.0, R_z=10.0):
    dx, dz = 1e-6, 1e-6
    energy = 12000.0
    x = (np.arange(nx) - nx // 2) * dx
    z = (np.arange(nz) - nz // 2) * dz
    X, Z = np.meshgrid(x, z)
    k = 2 * np.pi / (1.239842e-06 / energy)
    amp = np.exp(-X**2 / (2 * (60e-6)**2) - Z**2 / (2 * (30e-6)**2))
    phase = (k / (2 * R_x)) * X**2 + (k / (2 * R_z)) * Z**2
    E = amp * np.exp(1j * phase)
    return WavefrontSnapshot(
        Ex=E, Ez=np.zeros_like(E),
        x_start=x[0], x_step=dx, z_start=z[0], z_step=dz,
        nx=nx, nz=nz, photon_energy_eV=energy,
        Robs_x=R_x, Robs_z=R_z)


def test_wavefront_properties():
    wfr = _make_wavefront()
    assert wfr.intensity.shape == (128, 256)
    assert wfr.total_energy > 0
    assert wfr.wavelength > 0
    assert len(wfr.x_coords) == 256
    assert len(wfr.z_coords) == 128
    print("✓ WavefrontSnapshot properties")


def test_spatial_maps():
    wfr = _make_wavefront()
    maps = prepare_spatial_maps(wfr)
    assert maps.shape == (N_CHANNELS, 128, 256)
    assert maps.dtype == np.float32
    # Ch0: intensity in [0, 1]
    assert maps[0].min() >= 0
    assert maps[0].max() <= 1.0
    # Ch1-2: phase gradient in [-1, 1]
    assert maps[1].min() >= -1.0
    assert maps[1].max() <= 1.0
    # Ch3: sampling quality in [0, 1]
    assert maps[3].min() >= 0
    assert maps[3].max() <= 1.0
    # Ch4: binary mask
    assert set(np.unique(maps[4])).issubset({0.0, 1.0})
    print("✓ Spatial map preparation")


def test_patch_extraction():
    wfr = _make_wavefront(nx=256, nz=256)
    maps = prepare_spatial_maps(wfr)
    patches, positions = extract_patches(maps)
    expected_patches = (256 // PATCH_SIZE) * (256 // PATCH_SIZE)
    assert patches.shape == (expected_patches, N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    assert positions.shape == (expected_patches, 4)
    # Positions in [0, 1]
    assert positions.min() >= 0
    assert positions.max() <= 1.0
    print(f"✓ Patch extraction: {expected_patches} patches of {PATCH_SIZE}×{PATCH_SIZE}")


def test_patch_extraction_non_divisible():
    """Test that non-divisible grid sizes get padded correctly."""
    wfr = _make_wavefront(nx=300, nz=200)
    maps = prepare_spatial_maps(wfr)
    patches, positions = extract_patches(maps)
    assert patches.shape[0] > 0
    assert patches.shape[2] == PATCH_SIZE
    assert patches.shape[3] == PATCH_SIZE
    print(f"✓ Patch extraction (non-divisible): {patches.shape[0]} patches from 300×200 grid")


def test_position_encoding():
    positions = np.array([[0.5, 0.5, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    D = 64
    enc = sinusoidal_position_encoding(positions, D)
    assert enc.shape == (2, D)
    assert np.isfinite(enc).all()
    print("✓ Sinusoidal position encoding")


def test_analytical_estimator_gaussian():
    est = AnalyticalDriftEstimator()
    result = est.estimate(
        R_x=20.0, R_z=20.0, sigma_x=80e-6, sigma_z=40e-6,
        drift_length=5.0, photon_energy_eV=12000,
        dx=1e-6, dz=1e-6, nx=512, nz=256)
    assert result.AT in [0, 1, 2, 3, 4, 5]
    assert result.pxm > 0
    assert result.pzm > 0
    assert result.pxd > 0
    assert result.pzd > 0
    assert 0 <= result.expected_quality <= 1
    print(f"✓ Analytical estimator: AT={result.AT}, pxm={result.pxm}, pzm={result.pzm}")


def test_analytical_estimator_to_waist():
    est = AnalyticalDriftEstimator()
    result = est.estimate(
        R_x=-3.0, R_z=-3.0, sigma_x=30e-6, sigma_z=30e-6,
        drift_length=2.5, photon_energy_eV=8000,
        dx=0.5e-6, dz=0.5e-6, nx=256, nz=256)
    assert result.AT == 4, f"Expected AT=4 for converging beam, got {result.AT}"
    print(f"✓ Analytical estimator (to waist): AT={result.AT}")


def test_analytical_estimator_flat():
    est = AnalyticalDriftEstimator()
    result = est.estimate(
        R_x=1e23, R_z=1e23, sigma_x=500e-6, sigma_z=500e-6,
        drift_length=0.5, photon_energy_eV=10.0,
        dx=20e-6, dz=20e-6, nx=256, nz=256)
    assert result.AT == 0, f"Expected AT=0 for flat phase, got {result.AT}"
    print(f"✓ Analytical estimator (flat phase): AT={result.AT}")


def test_prior_vector():
    est = AnalyticalDriftEstimator()
    result = est.estimate(
        R_x=20.0, R_z=20.0, sigma_x=80e-6, sigma_z=40e-6,
        drift_length=5.0, photon_energy_eV=12000,
        dx=1e-6, dz=1e-6, nx=512, nz=256)
    prior = result.to_prior_vector(5.0)
    assert prior.shape == (12,)
    assert np.isfinite(prior).all()
    print("✓ Prior vector generation")


if __name__ == "__main__":
    test_wavefront_properties()
    test_spatial_maps()
    test_patch_extraction()
    test_patch_extraction_non_divisible()
    test_position_encoding()
    test_analytical_estimator_gaussian()
    test_analytical_estimator_to_waist()
    test_analytical_estimator_flat()
    test_prior_vector()
    print("\nAll tests passed.")
