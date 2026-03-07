# SRW Parameter Advisor

**AI-driven propagation parameter optimization for the Synchrotron Radiation Workshop (SRW)**

Automatically selects propagation mode (`AnalTreatment` 0–4) and resize parameters (`pxm`, `pxd`, `pzm`, `pzd`) for SRW drift space propagation, trained via reinforcement learning to maximize accuracy while minimizing computational cost.

## The Problem

SRW's drift space propagator has 5 propagation modes, each optimal for a different physical regime. Choosing wrong produces artifacts, numerical instability, or incorrect results. Currently, parameter selection requires expert knowledge of:
- Wavefront curvature and its relation to the grid sampling
- Whether the beam is approaching, at, or past a waist
- How much the beam will expand through the drift
- Whether the quadratic phase can be resolved on the grid or needs analytical treatment

This project automates that selection.

## Propagation Modes

| Mode | Name | Use case |
|------|------|----------|
| 0 | Standard angular | Default FFT-based propagation |
| 1 | Quadratic term (moment) | Removes quadratic phase before FFT; for beams with strong curvature |
| 2 | Quadratic term (special) | Alternative quadratic phase treatment |
| 3 | From waist | Beam propagating away from a waist |
| 4 | To waist | Beam converging toward a waist |

## Approach

### Two-Stage Architecture

**Stage 1 -- Analytical Estimator** (no ML): Computes beam parameters (curvature, beam size, phase sampling) to produce a baseline parameter suggestion that handles ~80% of cases correctly.

**Stage 2 -- RL Agent** (trained): A CNN + Vision Transformer processes the actual wavefront spatial structure and predicts *corrections* to the analytical suggestion. Handles the remaining ~20% where the wavefront deviates from the analytical model (zone plate higher orders, mirror figure errors, post-slit diffraction, aberrated CRL beams).

### Reinforcement Learning Formulation

- **Contextual bandit**: single-step decision per drift space (not sequential RL)
- **State**: 5-channel physics-normalised spatial maps (128x128 patches) + 12 analytical prior scalars
- **Action**: propagation mode (5-class categorical) + resize corrections (4 continuous, log-space) -- jointly selected via mode-conditional policy
- **Reward**: complex field correlation against high-resolution reference - lambda x computational cost
- **Training oracle**: SRW itself is used to propagate with the agent's chosen parameters; an adaptive angular spectrum method provides the ground-truth reference

### Key Design Decisions

**CNN patch encoder**: Raw 5-channel patches (128x128) are processed by a 3-layer CNN (Conv2d -> BatchNorm -> ReLU, with stride downsampling and adaptive average pooling) before being fed to the transformer. This learns spatial features directly from the wavefront data rather than relying on hand-crafted statistics.

**Physics-normalised inputs**: Phase gradient channels are expressed in Nyquist fractions (theta/theta_nyquist), making the encoder scale-invariant across photon energies (10 eV -- 100 keV) and grid scales (nm -- m).

**Mode-conditional resize**: Each propagation mode has its own resize distribution. The agent learns that AT=1 needs less resolution (pxd) than AT=0 because it subtracts quadratic phase before FFT -- an interaction too complex to hand-code.

**Delta prediction**: The resize head outputs corrections to analytical estimates, not absolute values. Clean Gaussian beams get zero correction; structured wavefronts get targeted adjustments.

**SRW propagation parameter list** (12 elements):

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0 | Auto-Resize before | 1=yes, 0=no |
| 1 | Auto-Resize after | 1=yes, 0=no |
| 2 | Precision | Relative precision for auto-resizing (1.0 nominal) |
| 3 | AnalTreatment | Semi-analytical quadratic phase mode (0--4) |
| 4 | Fourier resize | Resize on Fourier side using FFT (1=yes, 0=no) |
| 5 | pxm | Horizontal range modification factor |
| 6 | pxd | Horizontal resolution modification factor |
| 7 | pzm | Vertical range modification factor |
| 8 | pzd | Vertical resolution modification factor |
| 9 | Shift type | Wavefront shift type before resizing |
| 10 | x center | New horizontal center after shift |
| 11 | z center | New vertical center after shift |

The agent controls indices 3 and 5--8. All factors are multipliers applied to the input wavefront mesh.

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
│   ├── __init__.py
│   ├── __main__.py              # python -m training entry point
│   ├── cli.py                   # CLI for dataset generation and training
│   ├── rl_bandit_agent.py       # CNN+ViT encoder + mode-conditional policy + training loop
│   └── adaptive_bpm.py          # Adaptive angular spectrum reference propagator
│
├── tests/                       # Tests
│   ├── test_preprocessing.py
│   └── test_training.py         # Agent, training, SRW cross-validation tests
│
├── docs/                        # Design documents
├── examples/                    # Usage examples
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/srw-parameter-advisor.git
cd srw-parameter-advisor

# Install (core only)
pip install -e .

# With training dependencies (PyTorch, TensorBoard)
pip install -e ".[training]"
```

Requires `srwpy` for SRW propagation (training and evaluation).

## Quick Start

### Dataset Generation

```bash
# Precompute training dataset (wavefront + reference pairs)
python -m training generate-dataset --output-dir data/train --n-samples 500

# With custom grid sizes
python -m training generate-dataset -o data/train -n 1000 --grid-sizes 128,256
```

### Training

```bash
# Train with precomputed dataset
python -m training train --dataset data/train --n-episodes 1000 --save model.pt

# Train with on-the-fly generation
python -m training train --n-episodes 200 --batch-size 8

# Resume from checkpoint
python -m training train --dataset data/train --resume model.pt --n-episodes 500 --save model_v2.pt

# With TensorBoard logging
python -m training train --dataset data/train --n-episodes 1000 --log-dir runs/exp1
```

### RL Agent Prediction

```python
import torch
from training.rl_bandit_agent import BanditAgent, predict
from srw_param_advisor.wavefront import WavefrontSnapshot

agent = BanditAgent(D=256, n_transformer_blocks=2)
checkpoint = torch.load("model.pt", map_location="cpu", weights_only=False)
agent.load_state_dict(checkpoint["agent_state_dict"])

wfr = WavefrontSnapshot.from_srw(srw_wavefront)
result = predict(agent, wfr, drift_length=5.0, R_x=20.0, R_z=20.0)
print(result)
# -> AnalTreatment=1 (Quad-phase, moment-based)
#    pxm=1.82, pxd=1.00, pzm=1.65, pzd=1.00
#    Confidence: 87%

# Get SRW propagation parameters directly
pp = result.to_srw_prop_params()  # 12-element list ready for SRW
```

### Post-propagation Validation

```python
from srw_param_advisor.validator import PropagationValidator
from srw_param_advisor.wavefront import WavefrontSnapshot

validator = PropagationValidator()

wfr_before = WavefrontSnapshot.from_srw(wfr_input)
wfr_after = WavefrontSnapshot.from_srw(wfr_output)

report = validator.validate(wfr_before, wfr_after)
print(report)
# -> FAIL: beam clipped at output grid edges (right: 3.2% of energy).
#    -> Try increasing pxm from 1.0 to at least 2.5
```

## Architecture

```
Input Wavefront (nz x nx complex field)
    |
    v
prepare_spatial_maps() -> 5 channels (C, H, W):
    [0] Normalised intensity |E|^2/max
    [1] theta_x / theta_nyquist (horizontal phase gradient)
    [2] theta_z / theta_nyquist (vertical phase gradient)
    [3] Sampling quality (1 - |dphi_max|/pi)
    [4] Validity mask
    |
    v
extract_patches() -> (N, 5, 128, 128) non-overlapping patches
    |
    v
CNN Patch Encoder (per patch):
    Conv2d(5, 32, 7, stride=4) -> BN -> ReLU
    Conv2d(32, 64, 3, stride=2) -> BN -> ReLU
    Conv2d(64, 128, 3, stride=2) -> BN -> ReLU
    AdaptiveAvgPool2d(1) -> Linear(128, D)
    |
    v                          prepare_analytical_prior()
    + sinusoidal position enc       |
    |                               v
    v                          Prior MLP: Linear(12, 32) -> ReLU -> Linear(32, D)
    |                               |
    +-------------------------------+
    |
    v
Transformer Encoder (sequence = [prior_token, patch_1, ..., patch_N])
    |
    v
Attention pool + Max pool -> combined (3*D)
    |
    +---> Policy trunk -> Mode logits (5-class)
    |                  -> Per-mode resize (mean, log_std) x 5
    |
    +---> Value head -> scalar baseline
```

## Current Status

- Post-propagation validator with 8 diagnostics and actionable recommendations
- Analytical parameter estimator (Stage 1)
- Physics-normalised spatial map preparation (Nyquist-fraction phase gradients)
- CNN + ViT architecture with mode-conditional policy (PyTorch, autograd)
- Universal parametric wavefront generator (super-Gaussian, aberrations, ring structure)
- Complete RL training loop (REINFORCE with value baseline)
- SRW integration for training propagation
- Adaptive angular spectrum reference propagator (GPU-accelerated)
- Precomputed dataset generation and loading
- CLI for dataset generation and training with checkpoint save/resume
- TensorBoard logging
- SRW cross-validation tests (angular spectrum agreement > 98%)

## Design Rationale

See the [docs/](docs/) directory for detailed design documents covering:
- Why RL over supervised learning (ground truth problem)
- Why mode-conditional resize (joint optimization)
- Why physics-normalised inputs (scale invariance)
- Why CNN + ViT over pure CNN (variable resolution handling + global context)
- Analysis of real SRW phase data (heavily wrapped but propagates fine)

## References

- SRW: O. Chubar, P. Elleaume, "Accurate And Efficient Computation Of Synchrotron Radiation In The Near Field Region", EPAC 1998
- SRW source code: `sroptdrf.h`, `sroptdrf.cpp` (drift space propagation)
- OASYS: L. Rebuffi, M. Sanchez del Rio, J. Synchrotron Rad. 24 (2017)

## License

MIT
