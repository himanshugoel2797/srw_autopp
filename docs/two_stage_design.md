# Refined Design: Analytical Prior + CNN Refinement

## The Insight

The problem naturally splits into two stages with different strengths:

**Stage 1 — Analytical (no AI):** Trace Gaussian beam parameters through the
optical layout using ABCD matrices. This gives R_x, R_z, waist positions, beam
sizes, and divergences at every drift space in the beamline. This is fast, exact
for ideal optics, and already partially implemented in SRW.

**Stage 2 — CNN (AI):** Given the analytical predictions AND the actual wavefront
on the grid, predict the correction to the analytical parameters that accounts for
what the optics actually did vs what they were supposed to do.

The CNN doesn't predict propagation parameters from scratch. It predicts the
**delta** between what the analytical model suggests and what actually works.

## Why This Split Makes Sense

| Aspect | Analytical (Stage 1) | CNN (Stage 2) |
|--------|---------------------|---------------|
| Ideal Gaussian beam through ideal optics | Perfect | Unnecessary |
| KB mirror with figure errors | Uses design R | Corrects for local phase distortion |
| Zone plate (multiple orders) | Uses 1st order R | Recognises multi-order structure, adjusts resize |
| Post-aperture Fresnel fringes | Ignores diffraction | Detects fringe structure, adjusts resolution |
| CRL with thickness errors | Uses design focal length | Sees aberrated phase, corrects |
| Near waist | Knows waist location from layout | Handles the actual beam shape at waist |
| Heavily wrapped phase | Knows R, computes expected wrapping | Confirms or corrects based on actual sampling quality |

The analytical stage handles 80%+ of cases correctly. The CNN adds value precisely
in the 20% where the real wavefront deviates from the design.

## Architecture

```
STAGE 1: Analytical Beamline Trace (no AI)
═══════════════════════════════════════════

  Optical layout definition
  (source params + list of elements with distances)
         │
         ▼
  ABCD matrix propagation through each element
         │
         ▼
  At each drift space, compute:
    • R_x, R_z (wavefront curvature radii)
    • σ_x, σ_z (beam sizes)
    • waist_x, waist_z (distance to waist)
    • σ'_x, σ'_z (divergences)
         │
         ▼
  Rule-based parameter estimator:
    • AnalTreatment from R, waist position, drift length
    • pxm, pzm from beam growth ratio
    • pxd, pzd from phase sampling requirement
         │
         ▼
  "Analytical suggestion" — 6 values:
    [AT_suggested, pxm_0, pxd_0, pzm_0, pzd_0, expected_quality]


STAGE 2: CNN Refinement (AI)
═══════════════════════════════════════════

  Inputs:
  ┌─────────────────────────────┐  ┌──────────────────────────┐
  │ Spatial maps (5ch, 64×64)   │  │ Analytical priors (12)    │
  │  ch0: intensity             │  │  R_x, R_z (from Stage 1)  │
  │  ch1: θ_x / θ_nyquist      │  │  σ_x, σ_z                │
  │  ch2: θ_z / θ_nyquist      │  │  waist_dist_x, waist_z   │
  │  ch3: sampling quality      │  │  AT_suggested             │
  │  ch4: validity mask         │  │  pxm_0, pxd_0, pzm_0,   │
  │                             │  │  pzd_0                    │
  └──────────┬──────────────────┘  │  drift_length             │
             │                     └────────────┬──────────────┘
             ▼                                  │
        CNN encoder                             ▼
        (spatial features)               Dense encoder
             │                          (prior features)
             ▼                                  │
        (256-dim)                          (64-dim)
             │                                  │
             └──────────┬───────────────────────┘
                        ▼
                   Concat (320)
                        │
                        ▼
                   Dense 320→128, ReLU
                   Dense 128→64, ReLU
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          AT head    Resize head  Confidence
          (2)       (4, log Δ)    (1)
```

### Key Change: Resize Head Predicts Corrections, Not Absolutes

```python
# The resize head outputs log-space CORRECTIONS to the analytical suggestion
delta_log_pxm = resize_head[0]   # correction to log(pxm_0)
delta_log_pxd = resize_head[1]
delta_log_pzm = resize_head[2]
delta_log_pzd = resize_head[3]

# Final prediction
pxm = pxm_0 * exp(delta_log_pxm)
pxd = pxd_0 * exp(delta_log_pxd)
pzm = pzm_0 * exp(delta_log_pzm)
pzd = pzd_0 * exp(delta_log_pzd)
```

Why this matters:

- For a clean Gaussian beam, the deltas are all ~0 (no correction needed).
  The CNN just needs to learn "this looks normal, don't change anything."
  Much easier than learning the absolute mapping from scratch.

- For a zone plate wavefront, the delta might be +0.3 for pxm ("the analytical
  estimate underestimates the range needed because there are extra diffraction
  orders expanding the beam").

- The CNN's task is now bounded: learn small corrections to a good starting point,
  not arbitrary parameter values from raw data.

### AT Head: Override or Confirm

```python
# AT head produces 3 outputs:
#   [0] logit for "agree with analytical suggestion"
#   [1] logit for "switch to AT=0"
#   [2] logit for "switch to AT=1"

# If the analytical stage suggests AT=1 and the CNN agrees → AT=1
# If the CNN sees something unusual → override to AT=0 or AT=1
```

In practice, the CNN should agree with the analytical suggestion >85% of the time.
The value is in the exceptions — cases where the wavefront structure makes the
analytically-suggested mode fail.

## Stage 1: Analytical Parameter Estimator (Detailed)

This is deterministic code, no ML. Based on the SRW C++ logic we analysed.

```python
class AnalyticalDriftEstimator:
    """
    Estimate propagation parameters from beam parameters at the drift entrance.
    
    These parameters can come from:
    - ABCD matrix trace through the beamline
    - SRW's own Robs/beam moments (if available)
    - User specification
    """
    
    def estimate(self, R_x, R_z, sigma_x, sigma_z, 
                 drift_length, photon_energy_eV,
                 dx, dz, nx, nz):
        
        lambda_m = 1.239842e-06 / photon_energy_eV
        L = drift_length
        
        # --- AnalTreatment selection ---
        # Based on ChooseLocalPropMode logic from sroptdrf.h
        
        new_Rx = R_x + L
        new_Rz = R_z + L
        
        # Check if approaching waist
        waist_x = abs(new_Rx) < 0.3 * abs(R_x) if R_x != 0 else False
        waist_z = abs(new_Rz) < 0.3 * abs(R_z) if R_z != 0 else False
        
        # Check quadratic phase wrapping across grid
        grid_half_x = nx * dx / 2
        grid_half_z = nz * dz / 2
        phase_cycles_x = abs(grid_half_x**2 / (lambda_m * R_x)) if R_x != 0 else 0
        phase_cycles_z = abs(grid_half_z**2 / (lambda_m * R_z)) if R_z != 0 else 0
        
        if waist_x and waist_z:
            AT = 4  # propagation to waist
        elif phase_cycles_x > 2 or phase_cycles_z > 2:
            AT = 1  # analytical quad-phase treatment
        else:
            AT = 0  # standard angular representation
        
        # --- Range magnification (pxm, pzm) ---
        # Beam size growth through drift
        if R_x != 0:
            growth_x = abs(new_Rx / R_x)
        else:
            # At waist: divergence-limited growth
            div_x = lambda_m / (4 * pi * sigma_x) if sigma_x > 0 else 1e-3
            growth_x = sqrt(1 + (L * div_x / max(sigma_x, 1e-15))**2)
        
        if R_z != 0:
            growth_z = abs(new_Rz / R_z)
        else:
            div_z = lambda_m / (4 * pi * sigma_z) if sigma_z > 0 else 1e-3
            growth_z = sqrt(1 + (L * div_z / max(sigma_z, 1e-15))**2)
        
        # Margin factor (SRW uses 1.1 as DiffractionFactor)
        margin = 1.15
        pxm = margin * growth_x if growth_x > 1.15 else 1.0
        pzm = margin * growth_z if growth_z > 1.15 else 1.0
        
        # Check if grid is over-sized (beam much smaller than grid)
        fill_x = 4 * sigma_x / (nx * dx) if nx * dx > 0 else 1
        fill_z = 4 * sigma_z / (nz * dz) if nz * dz > 0 else 1
        if fill_x < 0.2 and pxm >= 1.0:
            pxm = max(0.5, fill_x * 3)  # shrink
        if fill_z < 0.2 and pzm >= 1.0:
            pzm = max(0.5, fill_z * 3)
        
        # --- Resolution (pxd, pzd) ---
        # Needed if phase sampling is inadequate after propagation
        # In angular rep, max phase per pixel ≈ L·λ/(2·dx·N·dx)
        phase_per_pixel_x = abs(L) * lambda_m / (2 * dx**2 * nx)
        phase_per_pixel_z = abs(L) * lambda_m / (2 * dz**2 * nz)
        
        # Target: < 0.5π per pixel
        target = 0.5 * pi
        pxd = phase_per_pixel_x / target if phase_per_pixel_x > target else 1.0
        pzd = phase_per_pixel_z / target if phase_per_pixel_z > target else 1.0
        
        # If using AT=1, quad phase is subtracted → less sampling pressure
        if AT == 1:
            pxd = max(1.0, pxd * 0.3)  # much reduced need
            pzd = max(1.0, pzd * 0.3)
        
        # --- Expected quality ---
        # Rough estimate of how well this will work
        quality = 1.0
        if fill_x * pxm * growth_x > 0.8:
            quality -= 0.3  # risk of clipping
        if fill_z * pzm * growth_z > 0.8:
            quality -= 0.3
        if phase_per_pixel_x / max(pxd, 1) > 0.8 * pi:
            quality -= 0.2
        if phase_per_pixel_z / max(pzd, 1) > 0.8 * pi:
            quality -= 0.2
        quality = max(0.0, min(1.0, quality))
        
        # Distances to waist (useful context for CNN)
        waist_dist_x = -R_x if R_x != 0 else float('inf')
        waist_dist_z = -R_z if R_z != 0 else float('inf')
        
        return AnalyticalEstimate(
            AT=AT,
            pxm=round(pxm, 2), pxd=round(pxd, 2),
            pzm=round(pzm, 2), pzd=round(pzd, 2),
            expected_quality=quality,
            R_x=R_x, R_z=R_z,
            sigma_x=sigma_x, sigma_z=sigma_z,
            waist_dist_x=waist_dist_x, waist_dist_z=waist_dist_z,
        )
```

### Where this analytical estimator fails (motivating the CNN)

| Scenario | Analytical estimate | What actually happens | CNN correction |
|----------|-------------------|----------------------|----------------|
| Zone plate, 1st+3rd order | pxm based on 1st order beam size | 3rd order is 3× more divergent, clips | CNN sees ring structure in intensity, increases pxm |
| KB with 5 nm figure error | pxm, pzd from design curvature | Speckle creates local phase hotspots needing more resolution | CNN sees speckle in sampling quality map, increases pxd |
| Post-slit Fresnel fringes | Based on geometric beam size | Fringes extend beyond geometric shadow | CNN sees fringe pattern at edges, increases pxm |
| CRL with lens alignment error | AT=1 based on design R | Actual R differs → AT=1 subtracts wrong curvature | CNN detects mismatch in phase gradient pattern, may override to AT=0 |
| Beam at waist | AT=4 (to waist) | Works fine | CNN confirms, delta ≈ 0 |
| Clean Gaussian | All params correct | Works fine | CNN confirms, delta ≈ 0 |

## Stage 2: CNN Specification

### Spatial Input Channels (5 channels, 64×64)

Same as previous design document:

- **ch0:** Normalised intensity `|E|²/max`
- **ch1:** `θ_x / θ_nyquist` clipped to [-1, +1]
- **ch2:** `θ_z / θ_nyquist` clipped to [-1, +1]
- **ch3:** Sampling quality `1 - |Δφ_max|/π` per pixel
- **ch4:** Validity mask

### Analytical Prior Input (12 scalars)

```python
prior = [
    log10(|R_x|) * sign(R_x),   # signed log curvature x
    log10(|R_z|) * sign(R_z),   # signed log curvature z
    log10(sigma_x),              # beam size x
    log10(sigma_z),              # beam size z  
    log10(|waist_dist_x|) * sign(waist_dist_x),  # signed log waist distance x
    log10(|waist_dist_z|) * sign(waist_dist_z),  # signed log waist distance z
    float(AT_suggested),         # analytical suggestion (0, 1, or 4)
    log(pxm_0),                  # analytical resize suggestion
    log(pxd_0),
    log(pzm_0),
    log(pzd_0),
    drift_length (signed, log-scaled),
]
```

### CNN Architecture

```
Spatial: 5ch → Conv(5,32,3) BN ReLU → Conv(32,32,3) ReLU MaxPool2
         → Conv(32,64,3) BN ReLU → Conv(64,64,3) ReLU MaxPool2
         → Conv(64,128,3) BN ReLU → Conv(128,128,3) ReLU MaxPool2
         → Conv(128,256,3) BN ReLU → GlobalAvgPool → (256,)

Priors:  12 → Dense(12,64) ReLU → (64,)

Combined: Concat(320) → Dense(320,128) ReLU Dropout(0.2)
          → Dense(128,64) ReLU Dropout(0.1)

Heads:
  AT override:     Dense(64,3) → softmax → {agree, switch_to_0, switch_to_1}
  Resize delta:    Dense(64,4) → unconstrained (log-space correction)
  Confidence:      Dense(64,1) → sigmoid
```

### Output Interpretation

```python
# AT decision
at_probs = softmax(at_logits)  # [agree, force_0, force_1]
if argmax(at_probs) == 0:
    final_AT = AT_suggested  # keep analytical suggestion
elif argmax(at_probs) == 1:
    final_AT = 0
else:
    final_AT = 1

# Resize: apply correction to analytical base
final_pxm = pxm_0 * exp(delta[0])
final_pxd = pxd_0 * exp(delta[1])
final_pzm = pzm_0 * exp(delta[2])
final_pzd = pzd_0 * exp(delta[3])

# Confidence: if low, flag for user review
confidence = sigmoid(conf_logit)
```

### Loss Function

```python
# AT: cross-entropy (3-class)
L_at = cross_entropy(at_logits, at_target)

# Resize: Smooth L1 on the DELTA (not the absolute value)
# Target delta = log(optimal_param / analytical_param)
L_resize = smooth_l1(delta_pred, delta_target)

# Confidence: calibrated against actual post-propagation quality
L_conf = binary_cross_entropy(conf_pred, actual_quality)

loss = L_at + L_resize + 0.5 * L_conf
```

Because the CNN predicts deltas, the target values are centered around 0 for
most training examples (where the analytical estimate is correct). This makes
training easier and more stable than predicting absolutes.

## Training Data Generation

### For each training example:

```python
1. Sample random beamline configuration
2. Run Stage 1 analytical estimator → get prior params
3. Generate actual wavefront (with imperfections)
4. Prepare CNN inputs (spatial maps + prior scalars)
5. Try ~30 parameter candidates via propagation + validator
6. Best candidate → compute delta from analytical suggestion → label
```

### What the delta labels look like:

For a clean Gaussian beam:
```
delta_target = [log(1.0/pxm_0), log(1.0/pxd_0), ...]  ≈ [0, 0, 0, 0]
AT_target = [1, 0, 0]  (agree with analytical)
```

For a zone plate with unexpected 3rd order:
```
delta_target = [log(2.5/pxm_0), 0, log(2.5/pzm_0), 0]  ≈ [+0.5, 0, +0.5, 0]
AT_target = [1, 0, 0]  (analytical AT is fine, just need more range)
```

For a CRL where analytical R is wrong:
```
delta_target = [0, log(1.5/pxd_0), 0, log(1.5/pzd_0)]  ≈ [0, +0.2, 0, +0.2]
AT_target = [0, 1, 0]  (override: switch to AT=0 because AT=1 subtracted wrong R)
```

## What the CNN Learns

In order of importance:

1. **"Is the wavefront well-described by the analytical model?"**
   If intensity is Gaussian-like and sampling quality map matches expected
   pattern for the given R → agree with analytical suggestion, small deltas.

2. **"Are there features the analytical model doesn't know about?"**
   Ring structure (zone plate), speckle (figure errors), fringes (diffraction)
   → increase range or resolution accordingly.

3. **"Is the sampling quality consistent with the stated R?"**
   If the prior says R=10m but the sampling quality map shows much denser
   wrapping → the actual R is different, analytical suggestion may be wrong.

4. **"Is the beam about to clip?"**
   Intensity at grid edges + beam growth estimate → increase pxm/pzm.

## Why This Justifies AI Over Pure Estimation

The analytical estimator handles standard beamlines perfectly. The CNN adds
value only when the wavefront has structure that deviates from the analytical
model. This deviation is:

- **Optics-dependent:** zone plates look different from CRLs
- **Instance-dependent:** two KB mirrors with different figure error PSDs
  produce different speckle patterns
- **Not easily parameterised:** you can't write a closed-form rule for
  "how much to increase pxm when a zone plate has 15% third-order content
  and the beam is slightly off-axis"

This is the classic ML sweet spot: the input-output relationship exists and is
deterministic, but the rule is too complex to write by hand for all cases.
The CNN learns it from examples.

For a clean beamline with ideal optics, the CNN is unnecessary — the analytical
estimator is sufficient and the CNN just passes through its suggestion unchanged.
The AI earns its keep on the hard cases.
