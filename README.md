# SRW Parameter Advisor

**AI-driven propagation parameter optimization for the Synchrotron Radiation Workshop (SRW)**

Automatically selects propagation mode (`AnalTreatment` 0–5) and resize parameters (`pxm`, `pxd`, `pzm`, `pzd`) for SRW drift space propagation, trained via reinforcement learning to maximize accuracy while minimizing computational cost.

## The Problem

SRW's drift space propagator has 6 propagation modes, each optimal for a different physical regime. Choosing wrong produces artifacts, numerical instability, or incorrect results. Currently, parameter selection requires expert knowledge of:
- Wavefront curvature and its relation to the grid sampling
- Whether the beam is approaching, at, or past a waist
- How much the beam will expand through the drift
- Whether the quadratic phase can be resolved on the grid or needs analytical treatment

This project automates that selection.

## Approach

### Two-Stage Architecture

**Stage 1 — Analytical Estimator** (no ML): Traces Gaussian beam parameters through the optical layout using ABCD matrices. Produces a baseline parameter suggestion that handles ~80% of cases correctly.

**Stage 2 — RL Agent** (trained): A Vision Transformer processes the actual wavefront spatial structure and predicts *corrections* to the analytical suggestion. Handles the remaining ~20% where the wavefront deviates from the analytical model (zone plate higher orders, mirror figure errors, post-slit diffraction, aberrated CRL beams).

### Reinforcement Learning Formulation

- **Contextual bandit**: single-step decision per drift space (not sequential RL)
- **State**: 5-channel physics-normalised spatial maps (128×128 patches) + 12 analytical prior scalars
- **Action**: propagation mode (6-class categorical) + resize corrections (4 continuous, log-space) — jointly selected via mode-conditional policy
- **Reward**: complex field correlation against high-resolution reference − λ × computational cost
- **Training oracle**: SRW itself at high resolution — no proxy metrics needed

### Key Design Decisions

**Physics-normalised inputs**: Phase gradient channels are expressed in Nyquist fractions (θ/θ_nyquist), making the CNN scale-invariant across photon energies (10 eV – 100 keV) and grid scales (nm – m).

**Mode-conditional resize**: Each propagation mode has its own resize distribution. The agent learns that AT=1 needs less resolution (pxd) than AT=0 because it subtracts quadratic phase before FFT — an interaction too complex to hand-code.

**Delta prediction**: The resize head outputs corrections to analytical estimates, not absolute values. Clean Gaussian beams get zero correction; structured wavefronts get targeted adjustments.

## Repository Structure

```
srw-parameter-advisor/
├── srw_param_advisor/           # Core package
│   ├── __init__.py
│   ├── wavefront.py             # WavefrontSnapshot data structure + SRW interface
│   ├── preprocessing.py         # Physics-normalised spatial maps, patch extraction
│   ├── analytical.py            # Stage 1: ABCD-based parameter estimation
│   └── validator.py             # Post-propagation quality validator
│
├── training/                    # RL training infrastructure
│   └── rl_bandit_agent.py       # ViT encoder + mode-conditional policy + training loop
│
├── docs/                        # Design documents
│   ├── rl_design.md             # RL contextual bandit formulation
│   ├── vit_architecture.md      # ViT encoder + mode-conditional policy
│   ├── two_stage_design.md      # Analytical + RL refinement approach
│   ├── universal_source.md      # Parametric wavefront generator
│   ├── phase_analysis_27m.png   # Real SRW phase data analysis
│   └── phase_analysis_kb_mirror.png  # KB mirror phase analysis
│
├── examples/                    # Usage examples
│   └── basic_usage.py
├── tests/                       # Tests
│   └── test_preprocessing.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/srw-parameter-advisor.git
cd srw-parameter-advisor

# Install (CPU, for inference and NumPy-based training)
pip install -e .

# With PyTorch (for production training)
pip install -e ".[training]"
```

## Quick Start

### Post-propagation validation (works now, no training needed)

```python
from srw_param_advisor.validator import PropagationValidator
from srw_param_advisor.wavefront import WavefrontSnapshot

validator = PropagationValidator()

# After an SRW propagation
wfr_before = WavefrontSnapshot.from_srw(wfr_input)
wfr_after = WavefrontSnapshot.from_srw(wfr_output)

report = validator.validate(wfr_before, wfr_after)
print(report)
# → FAIL: beam clipped at output grid edges (right: 3.2% of energy).
#   → Try increasing pxm from 1.0 to at least 2.5
```

### Analytical parameter estimation (works now, no training needed)

```python
from srw_param_advisor.analytical import AnalyticalDriftEstimator

estimator = AnalyticalDriftEstimator()
estimate = estimator.estimate(
    R_x=20.0, R_z=20.0,           # from beamline trace
    sigma_x=80e-6, sigma_z=40e-6, # beam sizes
    drift_length=5.0,
    photon_energy_eV=12000,
    dx=1e-6, dz=1e-6,
    nx=1024, nz=512,
)
print(f"Suggested: AT={estimate.AT}, pxm={estimate.pxm}, pzm={estimate.pzm}")
```

### RL agent prediction (requires trained model)

```python
from training.rl_bandit_agent import BanditAgent, predict

agent = BanditAgent(D=256, n_transformer_blocks=4)
# agent.load("trained_model.npz")  # load trained weights

result = predict(agent, wfr, drift_length=5.0, R_x=20.0, R_z=20.0)
print(result)
# → AnalTreatment=1 (Quad-phase, moment-based)
#   pxm=1.82, pxd=1.00, pzm=1.65, pzd=1.00
#   Confidence: 87%
#   Mode probabilities:
#     Quad-phase (moment)    87% █████████████████
#     Standard angular        8% █
#     To waist                3%
#     ...
```

## Current Status

### Working Now
- ✅ Post-propagation validator with 8 diagnostics and actionable recommendations
- ✅ Analytical parameter estimator (Stage 1)
- ✅ Physics-normalised spatial map preparation
- ✅ ViT architecture with mode-conditional policy (forward pass validated)
- ✅ Universal parametric wavefront generator
- ✅ Complete RL training loop (contextual bandit with REINFORCE)
- ✅ Evaluation pipeline comparing agent vs analytical baseline

### Needs PyTorch for Production
- ⬜ Proper backpropagation (currently using weight perturbation in NumPy)
- ⬜ 4-layer CNN patch embedding (currently using statistical features)
- ⬜ Training on actual SRW propagation (currently using toy angular-spectrum)
- ⬜ Curriculum learning (λ=0 → λ=0.05 → λ=0.2)
- ⬜ Pre-computed reference cache

### Future Extensions
- ⬜ Support for AnalTreatment modes 2, 3, 5 (require SRW-specific training data)
- ⬜ Integration with OASYS / Sirepo for beamline-level parameter optimization
- ⬜ Attention visualization for debugging predictions
- ⬜ Continuous learning from deployed predictions

## Design Rationale

See the [docs/](docs/) directory for detailed design documents covering:
- Why RL over supervised learning (ground truth problem)
- Why mode-conditional resize (joint optimization)
- Why physics-normalised inputs (scale invariance)
- Why ViT over CNN (variable resolution handling)
- Analysis of real SRW phase data (heavily wrapped but propagates fine)

## References

- SRW: O. Chubar, P. Elleaume, "Accurate And Efficient Computation Of Synchrotron Radiation In The Near Field Region", EPAC 1998
- SRW source code: `sroptdrf.h`, `sroptdrf.cpp` (drift space propagation)
- OASYS: L. Rebuffi, M. Sanchez del Rio, J. Synchrotron Rad. 24 (2017)

## License

MIT
