# Universal Parametric Source for Training Data Generation

## Goal

A single parametric wavefront generator with ~12 tunable parameters that
produces the full range of beam profiles encountered in synchrotron beamlines,
without simulating actual undulators or optical elements. Sweep the parameters
→ generate diverse training wavefronts → propagate through a focusing element
and drift → label with the validator → train the ViT.

## The Distribution

The electric field is defined as:

```
E(x, z) = A(x, z) · exp(i · Φ(x, z))
```

where the amplitude `A` and phase `Φ` are each parametric functions.

### Amplitude: Generalized Gaussian + Modal Perturbation

```
A(x, z) = A_base(x, z) · (1 + η · A_modes(x, z))
```

**Base amplitude** — a super-Gaussian with independent x/z profiles:

```
A_base(x, z) = exp( -(|x - x_c| / w_x)^p_x / 2  -  (|z - z_c| / w_z)^p_z / 2 )
```

Parameters:
- `w_x, w_z`: beam widths (metres)
- `p_x, p_z`: shape exponents
  - p = 2: standard Gaussian
  - p = 1: exponential (cusp-like, Lorentzian tails) 
  - p = 4–6: super-Gaussian (flat-top with steep edges, like an undulator
    central cone illuminating a finite aperture)
  - p = 10+: approaches a hard-edged flat-top
- `x_c, z_c`: beam center offsets

**Modal perturbation** — adds higher-order structure:

```
A_modes(x, z) = Σ_n  c_n · H_n(x/w_x) · exp(-(x/w_x)²/2)  ·  H_m(z/w_z) · exp(-(z/w_z)²/2)
```

where `H_n` are Hermite polynomials (or simpler: a cosine-ring model).

For practical generation, a simpler ring model captures the key structure:

```
A_modes(x, z) = c_ring · cos(2π · r / λ_ring + φ_ring)
```

where `r = sqrt((x/w_x)² + (z/w_z)²)` and `λ_ring` is the ring spacing.

Parameters:
- `η`: modal mixing strength (0 = pure base, 0.5 = heavy structure)
- `c_ring`: ring amplitude
- `λ_ring`: ring period (in beam-width units)
- `φ_ring`: ring phase (shifts the pattern)

### Phase: Curvature + Aberrations

```
Φ(x, z) = Φ_quad(x, z) + Φ_aberr(x, z) + Φ_tilt(x, z)
```

**Quadratic phase** (curvature):
```
Φ_quad = k/(2·R_x) · (x - x_c)² + k/(2·R_z) · (z - z_c)²
```

Parameters:
- `R_x, R_z`: radii of curvature (metres, signed, ±∞ for flat)

**Aberration phase** (Zernike-like, low order):
```
Φ_aberr = Σ  a_j · Z_j(ρ, θ)
```

where `ρ = r / r_max`, `θ = atan2(z, x)`, and we use just the first few
non-trivial Zernike terms:

- `a_coma`: Z₃₁ — coma (asymmetric tail, common from off-axis mirrors)
- `a_astig`: Z₂₂ — astigmatism (different focus in x vs z, from KB mismatch)
- `a_spher`: Z₄₀ — spherical aberration (from CRL thickness errors)
- `a_trefoil`: Z₃₃ — trefoil (from 3-fold symmetric optics errors)

Parameters:
- `a_coma, a_astig, a_spher, a_trefoil`: aberration amplitudes (radians)

**Tilt:**
```
Φ_tilt = k · (θ_x · x + θ_z · z)
```

Parameters:
- `θ_x, θ_z`: tilt angles (radians)

## Complete Parameter Set (15 parameters)

| # | Parameter | Range | What it controls |
|---|-----------|-------|-----------------|
| 1 | `w_x` | 1 μm – 10 mm | Beam width, horizontal |
| 2 | `w_z` | 1 μm – 10 mm | Beam width, vertical |
| 3 | `p_x` | 1.0 – 10.0 | Shape: Gaussian(2) → flat-top(10) |
| 4 | `p_z` | 1.0 – 10.0 | Shape: Gaussian(2) → flat-top(10) |
| 5 | `R_x` | ±0.1 m – ±∞ | Curvature, horizontal |
| 6 | `R_z` | ±0.1 m – ±∞ | Curvature, vertical |
| 7 | `η` | 0.0 – 0.5 | Modal structure strength |
| 8 | `λ_ring` | 0.1 – 5.0 | Ring period (beam-width units) |
| 9 | `a_coma` | 0 – 5 rad | Coma aberration |
| 10 | `a_astig` | 0 – 5 rad | Astigmatism |
| 11 | `a_spher` | 0 – 5 rad | Spherical aberration |
| 12 | `θ_x` | -1 – 1 mrad | Tilt, horizontal |
| 13 | `θ_z` | -1 – 1 mrad | Tilt, vertical |
| 14 | `x_c` | offset | Beam center x |
| 15 | `z_c` | offset | Beam center z |

## What Each Parameter Regime Produces

### Clean Gaussian beam (TEM₀₀)
```python
p_x = p_z = 2, η = 0, all a_j = 0
```
→ Standard Gaussian. The baseline case.

### Undulator-like (central cone + structure)
```python
p_x = 3–5, p_z = 2–3   # broader in x (horizontal fan)
η = 0.1–0.3             # some ring/lobe structure
λ_ring = 1–3            # period of angular structure
a_coma = 0, a_spher = 0 # no aberrations (ideal source)
```
→ Super-Gaussian core with faint rings. Mimics the central cone of an
undulator with contributions from off-axis radiation.

Undulators naturally produce broader horizontal profiles (p_x > p_z)
due to the larger horizontal source size and emission angle.

### Post-zone-plate (multiple focal orders)
```python
p_x = p_z = 2           # Gaussian base
η = 0.2–0.5             # strong ring structure
λ_ring = 0.3–1.0        # closely spaced rings (diffraction orders)
R_x, R_z finite         # focused
```
→ Central peak with prominent Airy-like rings from the overlapping
diffraction orders. The ring spacing relates to the zone plate geometry.

### Post-CRL (aberrated Gaussian)
```python
p_x = p_z = 2–3         # slightly super-Gaussian from aperture
η = 0–0.1               # little ring structure
a_spher = 1–3            # spherical aberration from lens shape
a_coma = 0–1             # slight coma from misalignment
```
→ Gaussian with phase distortion. Intensity looks clean but phase has
higher-order structure that affects propagation parameter needs.

### Post-KB mirror (astigmatic + figure errors)
```python
p_x = 2, p_z = 2        # Gaussian in each plane
R_x ≠ R_z               # different curvatures (astigmatic)
a_astig = 0.5–2          # residual astigmatism
a_coma = 0–2             # from alignment errors
η = 0.05–0.15            # mild speckle-like structure
λ_ring = 0.5–2           # figure error correlation length
```
→ Astigmatic beam with some speckle. The different R_x, R_z produce
different wrapping rates in the two planes.

### Post-slit (Fresnel diffraction)
```python
p_x = 8–10              # steep-edged (slit defines aperture)
p_z = 2                  # Gaussian in other plane
η = 0.3–0.5             # strong fringe structure at edges
λ_ring = 0.5–1.5        # Fresnel fringe period
```
→ Flat-top profile with oscillatory fringes near the edges. The super-
Gaussian with ring perturbation at the edges mimics Fresnel diffraction
without solving the integral.

### Hard-edged flat-top (e.g., uniformly illuminated aperture)
```python
p_x = p_z = 10+         # hard edges
η = 0                    # no structure (ideal)
```
→ Flat intensity across the aperture, sharp cutoff.

## Implementation

```python
import numpy as np
from scipy.special import hermite

def generate_universal_wavefront(
    # Grid
    nx=512, nz=256, dx=1e-6, dz=1e-6,
    photon_energy_eV=12000.0,
    # Amplitude shape
    w_x=100e-6, w_z=50e-6,
    p_x=2.0, p_z=2.0,
    x_c=0.0, z_c=0.0,
    # Curvature
    R_x=1e23, R_z=1e23,
    # Modal structure
    eta=0.0, ring_period=1.0, ring_phase=0.0,
    # Aberrations (radians at beam edge)
    a_coma=0.0, a_astig=0.0, a_spher=0.0,
    # Tilt
    theta_x=0.0, theta_z=0.0,
    # Noise
    noise_level=0.0,
):
    """
    Generate a wavefront from the universal parametric distribution.
    
    Parameters
    ----------
    nx, nz : int — grid points
    dx, dz : float — pixel size (m)
    photon_energy_eV : float
    w_x, w_z : float — beam widths (m), 1/e amplitude radius
    p_x, p_z : float — shape exponent (2=Gaussian, >2=super-Gaussian, <2=cusp)
    x_c, z_c : float — beam center offset (m)
    R_x, R_z : float — curvature radii (m), signed, ±1e23 for flat
    eta : float — modal/ring structure strength (0–0.5)
    ring_period : float — ring spacing in beam-width units
    ring_phase : float — ring pattern phase offset (rad)
    a_coma, a_astig, a_spher : float — aberration amplitudes (rad)
    theta_x, theta_z : float — tilt angles (rad)
    noise_level : float — relative noise amplitude (0–0.1)
    
    Returns
    -------
    WavefrontSnapshot
    """
    lambda_m = 1.239842e-06 / photon_energy_eV
    k = 2 * np.pi / lambda_m
    
    x = (np.arange(nx) - nx // 2) * dx
    z = (np.arange(nz) - nz // 2) * dz
    X, Z = np.meshgrid(x, z)
    
    # Centered coordinates
    Xc = X - x_c
    Zc = Z - z_c
    
    # Normalised coordinates
    xn = Xc / w_x
    zn = Zc / w_z
    
    # --- Amplitude ---
    # Super-Gaussian base
    A_base = np.exp(-0.5 * (np.abs(xn)**p_x + np.abs(zn)**p_z))
    
    # Ring/modal perturbation
    r_norm = np.sqrt(xn**2 + zn**2)
    A_modes = np.cos(2 * np.pi * r_norm / max(ring_period, 0.01) + ring_phase)
    
    amplitude = A_base * (1.0 + eta * A_modes)
    amplitude = np.maximum(amplitude, 0)  # ensure non-negative
    
    # --- Phase ---
    phase = np.zeros_like(X)
    
    # Quadratic (curvature)
    if np.isfinite(R_x) and abs(R_x) < 1e20:
        phase += (k / (2 * R_x)) * Xc**2
    if np.isfinite(R_z) and abs(R_z) < 1e20:
        phase += (k / (2 * R_z)) * Zc**2
    
    # Aberrations (Zernike-like on normalised pupil)
    r_pupil = np.sqrt(xn**2 + zn**2)
    r_max = r_pupil.max()
    if r_max > 0:
        rho = r_pupil / r_max  # normalised radius [0, 1]
        theta = np.arctan2(zn, xn)
        
        # Coma: Z_3^1 ≈ (3ρ³ - 2ρ)·cos(θ)
        if a_coma != 0:
            phase += a_coma * (3 * rho**3 - 2 * rho) * np.cos(theta)
        
        # Astigmatism: Z_2^2 ≈ ρ²·cos(2θ)
        if a_astig != 0:
            phase += a_astig * rho**2 * np.cos(2 * theta)
        
        # Spherical: Z_4^0 ≈ 6ρ⁴ - 6ρ² + 1
        if a_spher != 0:
            phase += a_spher * (6 * rho**4 - 6 * rho**2 + 1)
    
    # Tilt
    if theta_x != 0:
        phase += k * theta_x * Xc
    if theta_z != 0:
        phase += k * theta_z * Zc
    
    # Random global phase (augmentation)
    phase += np.random.uniform(0, 2 * np.pi)
    
    # --- Assemble ---
    E = amplitude * np.exp(1j * phase)
    
    # Optional noise
    if noise_level > 0:
        noise = noise_level * amplitude.max() * (
            np.random.randn(nz, nx) + 1j * np.random.randn(nz, nx)
        )
        E += noise
    
    return WavefrontSnapshot(
        Ex=E,
        Ez=np.zeros_like(E),
        x_start=x[0], x_step=dx,
        z_start=z[0], z_step=dz,
        nx=nx, nz=nz,
        photon_energy_eV=photon_energy_eV,
        Robs_x=R_x, Robs_z=R_z,
    )


def sample_random_config(rng):
    """
    Sample random parameters from the universal distribution.
    Returns a dict of kwargs for generate_universal_wavefront.
    """
    energy = 10 ** rng.uniform(1.5, 5)  # 30 eV to 100 keV
    lambda_m = 1.239842e-06 / energy
    
    # Grid
    nx = rng.choice([128, 256, 512, 1024, 2048])
    nz = rng.choice([128, 256, 512, 1024])
    dx = 10 ** rng.uniform(-8, -4)  # 10 nm to 100 μm
    dz = dx * 10 ** rng.uniform(-0.5, 0.5)
    
    # Beam width (relative to grid for controlled fill)
    grid_half_x = nx * dx / 2
    grid_half_z = nz * dz / 2
    fill = 10 ** rng.uniform(-0.7, 0.2)  # 0.2× to 1.5× grid half-width
    w_x = fill * grid_half_x * 0.3
    w_z = w_x * 10 ** rng.uniform(-0.5, 0.5)  # allow asymmetry
    
    # Shape exponent
    # Weighted toward p=2 (Gaussian) with tails to flat-top
    p_x = 2.0 + abs(rng.standard_cauchy()) * 1.5  # heavy tail toward high p
    p_x = np.clip(p_x, 1.0, 12.0)
    p_z = 2.0 + abs(rng.standard_cauchy()) * 1.5
    p_z = np.clip(p_z, 1.0, 12.0)
    
    # Curvature
    if rng.random() < 0.1:  # 10% flat
        R_x = rng.choice([-1, 1]) * 1e23
        R_z = rng.choice([-1, 1]) * 1e23
    else:
        R_x = rng.choice([-1, 1]) * 10 ** rng.uniform(-0.5, 5)
        R_z = rng.choice([-1, 1]) * 10 ** rng.uniform(-0.5, 5)
    
    # Modal structure (most beams are clean, some are heavily structured)
    if rng.random() < 0.6:
        eta = 0.0  # 60% clean
    else:
        eta = rng.uniform(0.02, 0.5)  # 40% with some structure
    ring_period = 10 ** rng.uniform(-0.5, 0.7)  # 0.3 to 5 beam widths
    ring_phase = rng.uniform(0, 2 * np.pi)
    
    # Aberrations (most beams are unaberrated)
    if rng.random() < 0.5:
        a_coma = a_astig = a_spher = 0.0
    else:
        a_coma = rng.exponential(0.5) if rng.random() < 0.5 else 0.0
        a_astig = rng.exponential(0.5) if rng.random() < 0.5 else 0.0
        a_spher = rng.exponential(0.5) if rng.random() < 0.5 else 0.0
    
    # Tilt (small, random)
    theta_x = rng.normal(0, 1e-4)
    theta_z = rng.normal(0, 1e-4)
    
    # Center offset (usually small)
    x_c = rng.normal(0, grid_half_x * 0.05)
    z_c = rng.normal(0, grid_half_z * 0.05)
    
    # Noise (rare)
    noise_level = 0.0
    if rng.random() < 0.15:
        noise_level = 10 ** rng.uniform(-3, -1)
    
    # Drift length for subsequent propagation
    drift = rng.choice([-1, 1]) * 10 ** rng.uniform(-1, 2.5)
    
    return {
        'wfr_params': dict(
            nx=nx, nz=nz, dx=dx, dz=dz,
            photon_energy_eV=energy,
            w_x=max(w_x, dx), w_z=max(w_z, dz),
            p_x=p_x, p_z=p_z,
            x_c=x_c, z_c=z_c,
            R_x=R_x, R_z=R_z,
            eta=eta, ring_period=ring_period, ring_phase=ring_phase,
            a_coma=a_coma, a_astig=a_astig, a_spher=a_spher,
            theta_x=theta_x, theta_z=theta_z,
            noise_level=noise_level,
        ),
        'drift_length': drift,
    }
```

## Sampling Strategy for Training Data

The parameter space is large, so we use structured sampling that ensures
coverage of the physically important regimes:

| Regime | % of samples | Key parameter settings |
|--------|-------------|----------------------|
| Clean Gaussian | 25% | p=2, η=0, no aberrations |
| Super-Gaussian / undulator-like | 15% | p=3–6, small η, asymmetric w |
| Ring structure (zone plate-like) | 15% | η=0.1–0.5, small ring_period |
| Aberrated (CRL-like) | 15% | p=2, η=0, moderate aberrations |
| Astigmatic (KB-like) | 10% | R_x ≠ R_z, some aberrations |
| Flat-top / post-slit | 10% | p=6–12, optional edge fringes |
| Pathological (stress tests) | 10% | high η, strong aberrations, clipped |

Within each regime, curvature and drift length are varied independently
across their full range.

## Why This Works

The key insight: what matters for propagation parameters is not the
physical origin of the wavefront features (undulator vs zone plate vs
mirror), but their **effect on the field**:

- Ring/lobe structure in intensity → needs more grid range (pxm/pzm)
- Rapid phase variation → needs more resolution (pxd/pzd) or AT=1
- Asymmetric curvature → different x/z parameters needed
- Aberrated phase → AT=1 may subtract wrong quadratic, reconsider AT=0

The universal distribution generates fields with these features directly,
parameterised by their amplitude and character, without needing to model
the physical optics that produced them. A beam with η=0.3 and ring_period=1.0
behaves similarly whether it came from a zone plate or from some other
diffractive element — and needs similar propagation parameters.
