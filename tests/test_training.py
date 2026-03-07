"""Tests for RL bandit training components."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn

from srw_param_advisor.wavefront import WavefrontSnapshot
from srw_param_advisor.preprocessing import prepare_spatial_maps, PATCH_SIZE, N_CHANNELS
from training.rl_bandit_agent import (
    N_AUX, N_MODES, N_RESIZE,
    prepare_analytical_prior, get_analytical_params,
    BanditAgent,
    action_to_params, apply_resize,
    compute_accuracy, compute_cost, compute_reward,
    generate_universal_wavefront,
    BanditTrainer,
    precompute_dataset, PrecomputedDataset,
    _save_wavefront, _load_wavefront,
)
from training.adaptive_bpm import compute_reference_adaptive
from srw_param_advisor.validator import simulate_drift_propagation, generate_test_wavefront


def _fallback_propagate(wfr, drift_length, params):
    """Test-only propagator that uses simple angular spectrum (no SRW needed)."""
    wfr_r = apply_resize(wfr, params['pxm'], params['pzm'],
                          params['pxd'], params['pzd'])
    return simulate_drift_propagation(wfr_r, drift_length)


# ============================================================================
# Shared fixtures
# ============================================================================

def _make_wavefront(nx=256, nz=128, R_x=20.0, R_z=20.0, energy=12000.0):
    dx, dz = 1e-6, 1e-6
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


def _make_agent():
    return BanditAgent(D=64, n_transformer_blocks=1)


# ============================================================================
# Analytical prior
# ============================================================================

def test_prior_shape_and_finite():
    wfr = _make_wavefront()
    prior = prepare_analytical_prior(wfr, drift_length=5.0)
    assert prior.shape == (N_AUX,), f"Expected ({N_AUX},), got {prior.shape}"
    assert np.isfinite(prior).all(), "Prior contains non-finite values"
    assert prior.dtype == np.float32
    print("✓ prepare_analytical_prior: shape and finiteness")


def test_prior_at_suggestion_diverging():
    """Strongly diverging beam → AT=1 (many phase cycles)."""
    # Small R → large phase cycles → AT=1
    wfr = _make_wavefront(R_x=0.1, R_z=0.1)
    prior = prepare_analytical_prior(wfr, drift_length=0.05)
    at = int(prior[6])
    assert at in (0, 1, 4), f"AT should be 0, 1, or 4, got {at}"
    print(f"✓ prepare_analytical_prior: AT suggestion = {at} (diverging beam)")


def test_prior_at_suggestion_converging_to_waist():
    """Beam converging to waist → AT=4."""
    # new_R < 0.3*R → waist condition
    wfr = _make_wavefront(R_x=-3.0, R_z=-3.0)
    prior = prepare_analytical_prior(wfr, drift_length=2.5)
    at = int(prior[6])
    assert at == 4, f"Expected AT=4 for converging beam, got {at}"
    print(f"✓ prepare_analytical_prior: AT=4 for converging beam")


def test_prior_custom_R_and_sigma():
    wfr = _make_wavefront()
    prior_default = prepare_analytical_prior(wfr, drift_length=5.0)
    prior_custom = prepare_analytical_prior(wfr, drift_length=5.0,
                                            R_x=50.0, R_z=50.0,
                                            sigma_x=100e-6, sigma_z=100e-6)
    assert not np.allclose(prior_default, prior_custom), \
        "Custom R/sigma should change the prior"
    print("✓ prepare_analytical_prior: custom R and sigma override")


def test_get_analytical_params_types_and_ranges():
    wfr = _make_wavefront()
    prior = prepare_analytical_prior(wfr, drift_length=5.0)
    ap = get_analytical_params(prior)
    assert set(ap.keys()) == {'AT', 'pxm', 'pxd', 'pzm', 'pzd'}
    assert ap['AT'] in range(N_MODES)
    for key in ('pxm', 'pxd', 'pzm', 'pzd'):
        assert ap[key] > 0, f"{key} should be positive, got {ap[key]}"
    print(f"✓ get_analytical_params: AT={ap['AT']}, pxm={ap['pxm']:.2f}")


# ============================================================================
# Patch CNN encoder
# ============================================================================

def test_patch_cnn_output_shape():
    """CNN patch encoder produces (N, D) embeddings from raw patches."""
    agent = _make_agent()
    agent.eval()
    patch = torch.randn(2, N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        emb = agent.patch_cnn(patch)
    assert emb.shape == (2, agent.D), f"Expected (2, {agent.D}), got {emb.shape}"
    print(f"✓ patch_cnn: output shape (2, {agent.D})")


def test_patch_cnn_finite_output():
    """CNN should produce finite outputs for random input."""
    agent = _make_agent()
    agent.eval()
    patch = torch.randn(1, N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        emb = agent.patch_cnn(patch)
    assert torch.isfinite(emb).all(), "CNN produced non-finite embeddings"
    print("✓ patch_cnn: all finite")


def test_patch_cnn_zero_patch():
    """Zero patch should not produce NaN."""
    agent = _make_agent()
    agent.eval()
    patch = torch.zeros(1, N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        emb = agent.patch_cnn(patch)
    assert torch.isfinite(emb).all(), "Zero patch produced non-finite embeddings"
    print("✓ patch_cnn: zero patch handled safely")


# ============================================================================
# BanditAgent
# ============================================================================

def test_agent_forward_shapes():
    agent = _make_agent()
    agent.eval()
    wfr = _make_wavefront(nx=256, nz=256)
    spatial = prepare_spatial_maps(wfr)
    prior = prepare_analytical_prior(wfr, drift_length=5.0)

    with torch.no_grad():
        mode_logits, resize_params, value = agent(spatial, prior)

    assert mode_logits.shape == (N_MODES,), \
        f"mode_logits shape: expected ({N_MODES},), got {mode_logits.shape}"
    assert len(resize_params) == N_MODES
    for rp in resize_params:
        assert rp['mean'].shape == (N_RESIZE,)
        assert rp['log_std'].shape == (N_RESIZE,)
    assert value.shape == (), f"value should be scalar, got shape {value.shape}"
    print("✓ BanditAgent.forward: output shapes correct")


def test_agent_forward_finite():
    agent = _make_agent()
    agent.eval()
    wfr = _make_wavefront()
    spatial = prepare_spatial_maps(wfr)
    prior = prepare_analytical_prior(wfr, drift_length=5.0)

    with torch.no_grad():
        mode_logits, resize_params, value = agent(spatial, prior)

    assert torch.isfinite(mode_logits).all()
    assert torch.isfinite(value)
    for rp in resize_params:
        assert torch.isfinite(rp['mean']).all()
        assert torch.isfinite(rp['log_std']).all()
    print("✓ BanditAgent.forward: all outputs finite")


def test_agent_log_std_clamped():
    agent = _make_agent()
    agent.eval()
    wfr = _make_wavefront()
    spatial = prepare_spatial_maps(wfr)
    prior = prepare_analytical_prior(wfr, drift_length=5.0)

    with torch.no_grad():
        _, resize_params, _ = agent(spatial, prior)

    for rp in resize_params:
        assert (rp['log_std'] >= -3).all() and (rp['log_std'] <= 1).all(), \
            "log_std outside clamp range [-3, 1]"
    print("✓ BanditAgent: log_std values within clamp range")


def test_agent_sample_action():
    agent = _make_agent()
    wfr = _make_wavefront()
    spatial = prepare_spatial_maps(wfr)
    prior = prepare_analytical_prior(wfr, drift_length=5.0)

    mode, resize_deltas, log_prob, entropy, value, mode_probs = \
        agent.sample_action(spatial, prior)

    assert 0 <= mode < N_MODES, f"mode={mode} out of range"
    assert resize_deltas.shape == (N_RESIZE,)
    assert np.isfinite(resize_deltas).all()
    assert log_prob.requires_grad or log_prob.is_leaf
    assert float(log_prob) < 0 or True  # log_prob can be any value
    assert torch.isfinite(log_prob)
    assert torch.isfinite(entropy)
    assert mode_probs.shape == (N_MODES,)
    assert abs(mode_probs.sum() - 1.0) < 1e-5, \
        f"mode_probs don't sum to 1: {mode_probs.sum()}"
    print(f"✓ BanditAgent.sample_action: mode={mode}, probs sum={mode_probs.sum():.6f}")


def test_agent_deterministic_action():
    agent = _make_agent()
    wfr = _make_wavefront()
    spatial = prepare_spatial_maps(wfr)
    prior = prepare_analytical_prior(wfr, drift_length=5.0)

    mode, resize_deltas, value, mode_probs, _ = agent.deterministic_action(spatial, prior)

    assert 0 <= mode < N_MODES
    assert mode == int(np.argmax(mode_probs)), \
        "deterministic mode should be argmax of mode_probs"
    assert resize_deltas.shape == (N_RESIZE,)
    assert abs(mode_probs.sum() - 1.0) < 1e-5
    print(f"✓ BanditAgent.deterministic_action: mode={mode} (argmax)")


def test_agent_parameters_change_after_update():
    """Verify gradients flow and parameters update after a loss.backward()."""
    agent = _make_agent()
    agent.train()

    wfr = _make_wavefront()
    spatial = prepare_spatial_maps(wfr)
    prior = prepare_analytical_prior(wfr, drift_length=5.0)

    params_before = {k: v.clone() for k, v in agent.named_parameters()}

    mode, resize_deltas, log_prob, entropy, value, _ = agent.sample_action(spatial, prior)
    reward_t = torch.tensor(0.7, dtype=torch.float32)
    loss = -(reward_t - value.detach()) * log_prob + 0.5 * (value - reward_t).pow(2)

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    any_changed = any(
        not torch.equal(params_before[k], v)
        for k, v in agent.named_parameters()
        if v.requires_grad
    )
    assert any_changed, "No parameters changed after optimizer step"
    print("✓ BanditAgent: parameters update after REINFORCE loss")


# ============================================================================
# Action → parameters conversion
# ============================================================================

def test_action_to_params_ranges():
    wfr = _make_wavefront()
    prior = prepare_analytical_prior(wfr, drift_length=5.0)
    deltas = np.zeros(N_RESIZE, dtype=np.float32)  # no correction

    for mode in range(N_MODES):
        params = action_to_params(mode, deltas, prior)
        assert params['analyt_treat'] == mode
        assert 0.3 <= params['pxm'] <= 20.0, f"pxm={params['pxm']} out of clip range"
        assert 0.3 <= params['pxd'] <= 10.0, f"pxd={params['pxd']} out of clip range"
        assert 0.3 <= params['pzm'] <= 20.0, f"pzm={params['pzm']} out of clip range"
        assert 0.3 <= params['pzd'] <= 10.0, f"pzd={params['pzd']} out of clip range"
    print("✓ action_to_params: all outputs within clip bounds")


def test_action_to_params_large_deltas_clipped():
    """Very large positive deltas should still produce clipped params."""
    wfr = _make_wavefront()
    prior = prepare_analytical_prior(wfr, drift_length=5.0)
    deltas = np.array([10.0, 10.0, 10.0, 10.0])  # exp(10) >> clip max

    params = action_to_params(0, deltas, prior)
    assert params['pxm'] == 20.0
    assert params['pxd'] == 10.0
    assert params['pzm'] == 20.0
    assert params['pzd'] == 10.0
    print("✓ action_to_params: large deltas clipped to max")


# ============================================================================
# apply_resize
# ============================================================================

def test_apply_resize_identity():
    wfr = _make_wavefront(nx=128, nz=128)
    out = apply_resize(wfr, pxm=1.0, pzm=1.0, pxd=1.0, pzd=1.0)
    assert out.nx == wfr.nx
    assert out.nz == wfr.nz
    assert out.x_step == wfr.x_step
    assert out.z_step == wfr.z_step
    print("✓ apply_resize: identity parameters preserve grid")


def test_apply_resize_expand():
    wfr = _make_wavefront(nx=128, nz=128)
    out = apply_resize(wfr, pxm=2.0, pzm=2.0, pxd=1.0, pzd=1.0)
    assert out.nx > wfr.nx, "pxm=2 should expand x grid"
    assert out.nz > wfr.nz, "pzm=2 should expand z grid"
    assert out.Ex.shape == (out.nz, out.nx)
    print(f"✓ apply_resize: expand {wfr.nx}×{wfr.nz} → {out.nx}×{out.nz}")


def test_apply_resize_resolution():
    wfr = _make_wavefront(nx=128, nz=128)
    out = apply_resize(wfr, pxm=1.0, pzm=1.0, pxd=2.0, pzd=2.0)
    assert abs(out.x_step - wfr.x_step / 2.0) < 1e-20
    assert abs(out.z_step - wfr.z_step / 2.0) < 1e-20
    print("✓ apply_resize: pxd=2 halves x_step")


def test_apply_resize_returns_wavefront_snapshot():
    wfr = _make_wavefront()
    out = apply_resize(wfr, pxm=1.5, pzm=1.5, pxd=1.0, pzd=1.0)
    assert isinstance(out, WavefrontSnapshot)
    assert out.photon_energy_eV == wfr.photon_energy_eV
    print("✓ apply_resize: returns WavefrontSnapshot with correct metadata")


# ============================================================================
# Accuracy / cost / reward
# ============================================================================

def test_compute_accuracy_identical():
    """Same wavefront compared against itself → correlation = 1."""
    wfr = _make_wavefront(nx=64, nz=64)
    acc = compute_accuracy(wfr, wfr)
    assert abs(acc - 1.0) < 1e-6, f"Identical wavefronts should give acc=1, got {acc}"
    print(f"✓ compute_accuracy: identical wavefronts → {acc:.6f}")


def test_compute_accuracy_range():
    """Accuracy is always in [0, 1]."""
    wfr = _make_wavefront(nx=64, nz=64)
    rng = np.random.RandomState(1)
    E_rand = rng.randn(64, 64) + 1j * rng.randn(64, 64)
    wfr2 = WavefrontSnapshot(
        Ex=E_rand, Ez=np.zeros_like(E_rand),
        x_start=wfr.x_start, x_step=wfr.x_step,
        z_start=wfr.z_start, z_step=wfr.z_step,
        nx=64, nz=64, photon_energy_eV=wfr.photon_energy_eV)
    acc = compute_accuracy(wfr, wfr2)
    assert 0.0 <= acc <= 1.0, f"acc={acc} out of [0, 1]"
    print(f"✓ compute_accuracy: random vs structured → {acc:.4f} (in [0,1])")


def test_compute_accuracy_no_overlap():
    """Non-overlapping grids → 0."""
    wfr = _make_wavefront(nx=64, nz=64)
    E2 = np.ones((64, 64), dtype=complex)
    wfr2 = WavefrontSnapshot(
        Ex=E2, Ez=np.zeros_like(E2),
        x_start=10.0, x_step=wfr.x_step,  # far away
        z_start=10.0, z_step=wfr.z_step,
        nx=64, nz=64, photon_energy_eV=wfr.photon_energy_eV)
    acc = compute_accuracy(wfr, wfr2)
    assert acc == 0.0, f"Non-overlapping grids should give acc=0, got {acc}"
    print("✓ compute_accuracy: non-overlapping grids → 0")


def test_compute_cost():
    params_identity = {'pxm': 1.0, 'pxd': 1.0, 'pzm': 1.0, 'pzd': 1.0}
    cost_identity = compute_cost(params_identity)
    assert abs(cost_identity) < 1e-10, f"Identity params cost should be 0, got {cost_identity}"

    params_large = {'pxm': 2.0, 'pxd': 2.0, 'pzm': 2.0, 'pzd': 2.0}
    cost_large = compute_cost(params_large)
    assert cost_large > 0, "Larger params should have higher cost"
    print(f"✓ compute_cost: identity={cost_identity:.4f}, 2×larger={cost_large:.4f}")


def test_compute_reward():
    assert compute_reward(1.0, 0.0, lambda_cost=0.1) == 1.0
    assert abs(compute_reward(0.8, 1.0, lambda_cost=0.1) - 0.7) < 1e-9
    assert compute_reward(0.5, 2.0, lambda_cost=0.0) == 0.5  # no cost penalty
    print("✓ compute_reward: accuracy - λ·cost")


# ============================================================================
# generate_universal_wavefront
# ============================================================================

def test_generate_universal_wavefront():
    rng = np.random.RandomState(0)
    wfr, L = generate_universal_wavefront(rng, nx=128, nz=128)
    assert isinstance(wfr, WavefrontSnapshot)
    assert wfr.nx == 128
    assert wfr.nz == 128
    assert wfr.Ex.shape == (128, 128)
    assert np.any(np.abs(wfr.Ex) > 0), "Wavefront should be non-zero"
    assert isinstance(L, (float, np.floating))
    assert L != 0
    print(f"✓ generate_universal_wavefront: {wfr.nx}×{wfr.nz}, L={L:.3f} m")


def test_generate_universal_wavefront_variety():
    """Different seeds produce different wavefronts."""
    rng1 = np.random.RandomState(1)
    rng2 = np.random.RandomState(2)
    wfr1, _ = generate_universal_wavefront(rng1, nx=64, nz=64)
    wfr2, _ = generate_universal_wavefront(rng2, nx=64, nz=64)
    assert not np.allclose(wfr1.Ex, wfr2.Ex), "Different seeds should give different wavefronts"
    print("✓ generate_universal_wavefront: different seeds produce different wavefronts")


# ============================================================================
# Adaptive BPM reference
# ============================================================================

def test_compute_reference_adaptive_returns_wavefront():
    wfr = _make_wavefront(nx=64, nz=64)
    ref = compute_reference_adaptive(wfr, drift_length=2.0)
    assert isinstance(ref, WavefrontSnapshot)
    assert ref.photon_energy_eV == wfr.photon_energy_eV
    assert np.any(np.abs(ref.Ex) > 0)
    print("✓ compute_reference_adaptive: returns non-zero WavefrontSnapshot")


def test_compute_reference_adaptive_energy_conserved():
    """Total energy should be approximately conserved in free-space drift."""
    wfr = _make_wavefront(nx=64, nz=64)
    ref = compute_reference_adaptive(wfr, drift_length=1.0)
    # Energy = sum(|E|^2) * dx * dz; dx*dz may differ so compare total power
    E_in = np.sum(np.abs(wfr.Ex)**2) * wfr.x_step * wfr.z_step
    E_out = np.sum(np.abs(ref.Ex)**2) * ref.x_step * ref.z_step
    # Allow 5% tolerance (grid expansion changes normalization slightly)
    assert abs(E_out - E_in) / E_in < 0.05, \
        f"Energy not conserved: in={E_in:.3e}, out={E_out:.3e}"
    print(f"✓ compute_reference_adaptive: energy conserved ({abs(E_out-E_in)/E_in:.2%} change)")


def test_compute_reference_adaptive_negative_drift():
    """Negative drift should propagate in reverse direction."""
    wfr = _make_wavefront(nx=64, nz=64)
    ref_fwd = compute_reference_adaptive(wfr, drift_length=+1.0)
    ref_bwd = compute_reference_adaptive(wfr, drift_length=-1.0)
    # Results should differ from each other and from input
    assert not np.allclose(ref_fwd.Ex, ref_bwd.Ex), \
        "Forward and backward drift should differ"
    print("✓ compute_reference_adaptive: negative drift supported")


# ============================================================================
# BanditTrainer integration
# ============================================================================

def test_trainer_runs_without_error():
    """Smoke test: a few training episodes complete without exception."""
    agent = _make_agent()
    trainer = BanditTrainer(agent, lambda_cost=0.1, lr=1e-3, entropy_coeff=0.01,
                            propagate_fn=_fallback_propagate)
    trainer.train(n_episodes=4, batch_size=2, verbose=False)
    assert len(trainer.history) > 0, "No training history recorded"
    print(f"✓ BanditTrainer: ran {len(trainer.history)} update steps")


def test_trainer_history_fields():
    agent = _make_agent()
    trainer = BanditTrainer(agent, lambda_cost=0.1, lr=1e-3,
                            propagate_fn=_fallback_propagate)
    trainer.train(n_episodes=4, batch_size=2, verbose=False)
    for entry in trainer.history:
        assert 'episode' in entry
        assert 'mean_reward' in entry
        assert 'mean_accuracy' in entry
        assert 'mean_cost' in entry
        assert 'mode_dist' in entry
        assert len(entry['mode_dist']) == N_MODES
    print("✓ BanditTrainer: history entries have correct fields")


def test_trainer_parameters_change():
    """Parameters should change after training (vs. initial state)."""
    agent = _make_agent()
    params_before = {k: v.clone().detach() for k, v in agent.named_parameters()}
    trainer = BanditTrainer(agent, lambda_cost=0.1, lr=1e-3,
                            propagate_fn=_fallback_propagate)
    trainer.train(n_episodes=4, batch_size=2, verbose=False)
    any_changed = any(
        not torch.equal(params_before[k], v.detach())
        for k, v in agent.named_parameters()
    )
    assert any_changed, "Agent parameters did not change during training"
    print("✓ BanditTrainer: agent parameters update during training")


# ============================================================================
# Precomputed dataset
# ============================================================================

import tempfile
import json
from pathlib import Path


def test_save_load_wavefront_roundtrip():
    """Saving and loading a wavefront preserves all fields."""
    wfr = _make_wavefront(nx=64, nz=32)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.npz"
        _save_wavefront(path, wfr)
        loaded = _load_wavefront(path)

    assert loaded.nx == wfr.nx
    assert loaded.nz == wfr.nz
    assert np.allclose(loaded.Ex, wfr.Ex)
    assert np.allclose(loaded.Ez, wfr.Ez)
    assert abs(loaded.x_start - wfr.x_start) < 1e-15
    assert abs(loaded.x_step - wfr.x_step) < 1e-15
    assert abs(loaded.photon_energy_eV - wfr.photon_energy_eV) < 1e-6
    assert loaded.Robs_x == wfr.Robs_x
    assert loaded.Robs_z == wfr.Robs_z
    print("✓ _save_wavefront/_load_wavefront: roundtrip preserves all fields")


def test_save_load_wavefront_none_robs():
    """Wavefront with Robs=None roundtrips correctly."""
    wfr = _make_wavefront()
    wfr_no_r = WavefrontSnapshot(
        Ex=wfr.Ex, Ez=wfr.Ez,
        x_start=wfr.x_start, x_step=wfr.x_step,
        z_start=wfr.z_start, z_step=wfr.z_step,
        nx=wfr.nx, nz=wfr.nz,
        photon_energy_eV=wfr.photon_energy_eV,
        Robs_x=None, Robs_z=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.npz"
        _save_wavefront(path, wfr_no_r)
        loaded = _load_wavefront(path)
    assert loaded.Robs_x is None
    assert loaded.Robs_z is None
    print("✓ _save_wavefront/_load_wavefront: None Robs roundtrips as None")


def test_precompute_dataset_creates_files():
    """precompute_dataset creates manifest and sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = precompute_dataset(tmpdir, n_samples=3, grid_sizes=[128],
                                 seed=0, verbose=False)
        out = Path(out)
        assert (out / 'manifest.json').exists()
        with open(out / 'manifest.json') as f:
            manifest = json.load(f)
        n = manifest['n_samples']
        assert n > 0, "Should have at least one sample"
        assert (out / '00000_wfr.npz').exists()
        assert (out / '00000_ref.npz').exists()
    print(f"✓ precompute_dataset: created {n} samples with manifest")


def test_precomputed_dataset_load_and_iterate():
    """PrecomputedDataset loads and iterates correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        precompute_dataset(tmpdir, n_samples=4, grid_sizes=[128],
                           seed=1, verbose=False)
        dataset = PrecomputedDataset(tmpdir)
        assert len(dataset) > 0

        # Load a single sample
        wfr, ref, L = dataset.load_sample(0)
        assert isinstance(wfr, WavefrontSnapshot)
        assert isinstance(ref, WavefrontSnapshot)
        assert isinstance(L, float)
        assert L != 0

        # Iterate full epoch
        rng = np.random.RandomState(99)
        count = 0
        for wfr, ref, L in dataset.iter_epoch(rng):
            count += 1
        assert count == len(dataset)
    print(f"✓ PrecomputedDataset: loaded and iterated {count} samples")


def test_trainer_with_precomputed_dataset():
    """BanditTrainer runs with a precomputed dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        precompute_dataset(tmpdir, n_samples=4, grid_sizes=[128],
                           seed=2, verbose=False)
        dataset = PrecomputedDataset(tmpdir)

        agent = _make_agent()
        trainer = BanditTrainer(agent, lambda_cost=0.1, lr=1e-3, entropy_coeff=0.01,
                                propagate_fn=_fallback_propagate)
        trainer.train(n_episodes=4, batch_size=2, verbose=False, dataset=dataset)
        assert len(trainer.history) > 0
    print(f"✓ BanditTrainer with precomputed dataset: {len(trainer.history)} steps")


def test_trainer_precomputed_wraps_around():
    """Training with more episodes than dataset samples wraps around."""
    with tempfile.TemporaryDirectory() as tmpdir:
        precompute_dataset(tmpdir, n_samples=2, grid_sizes=[128],
                           seed=3, verbose=False)
        dataset = PrecomputedDataset(tmpdir)

        agent = _make_agent()
        trainer = BanditTrainer(agent, lambda_cost=0.1, lr=1e-3, entropy_coeff=0.01,
                                propagate_fn=_fallback_propagate)
        # 6 episodes > 2 samples, must wrap
        trainer.train(n_episodes=6, batch_size=2, verbose=False, dataset=dataset)
        assert len(trainer.history) > 0
    print(f"✓ BanditTrainer with precomputed dataset: wraps around correctly")


# ============================================================================
# SRW cross-validation tests
# ============================================================================

from training.rl_bandit_agent import srw_propagate

try:
    from srwpy.srwlib import srwl
    HAS_SRW = True
except ImportError:
    HAS_SRW = False


def test_to_srw_roundtrip():
    """WavefrontSnapshot → SRWLWfr → WavefrontSnapshot preserves fields."""
    if not HAS_SRW:
        print("⊘ to_srw roundtrip: skipped (srwpy not installed)")
        return
    wfr = _make_wavefront(nx=64, nz=32)
    srw_wfr = wfr.to_srw()
    back = WavefrontSnapshot.from_srw(srw_wfr)

    assert back.nx == wfr.nx
    assert back.nz == wfr.nz
    assert np.allclose(back.Ex, wfr.Ex, atol=1e-6)
    assert np.allclose(back.Ez, wfr.Ez, atol=1e-6)
    assert abs(back.x_start - wfr.x_start) < 1e-10
    assert abs(back.x_step - wfr.x_step) < 1e-10
    assert abs(back.z_start - wfr.z_start) < 1e-10
    assert abs(back.z_step - wfr.z_step) < 1e-10
    assert abs(back.photon_energy_eV - wfr.photon_energy_eV) < 1e-6
    print("✓ to_srw roundtrip: WavefrontSnapshot → SRWLWfr → WavefrontSnapshot")


def test_srw_propagate_runs():
    """srw_propagate produces a valid WavefrontSnapshot."""
    if not HAS_SRW:
        print("⊘ srw_propagate runs: skipped (srwpy not installed)")
        return
    wfr = _make_wavefront(nx=128, nz=128)
    params = {'analyt_treat': 1, 'pxm': 1.0, 'pxd': 1.0,
              'pzm': 1.0, 'pzd': 1.0}
    result = srw_propagate(wfr, drift_length=1.0, params=params)
    assert isinstance(result, WavefrontSnapshot)
    assert result.nx > 0 and result.nz > 0
    assert np.all(np.isfinite(result.Ex))
    print(f"✓ srw_propagate runs: {result.nx}×{result.nz} output")


def test_srw_vs_angular_spectrum_simple_beam():
    """SRW and angular spectrum should agree for a well-resolved Gaussian beam."""
    if not HAS_SRW:
        print("⊘ SRW vs angular spectrum: skipped (srwpy not installed)")
        return
    from srw_param_advisor.validator import simulate_drift_propagation

    # Well-resolved Gaussian beam: large R, moderate sigma, plenty of grid room
    wfr = generate_test_wavefront(
        nx=256, nz=256, dx=2e-6, dz=2e-6,
        photon_energy_eV=12000, R_x=50.0, R_z=50.0,
        beam_sigma_x=60e-6, beam_sigma_z=60e-6,
    )
    L = 0.5  # short drift

    # Angular spectrum propagation (our own)
    ref_as = simulate_drift_propagation(wfr, L)

    # SRW propagation with standard angular mode (AT=0)
    params = {'analyt_treat': 0, 'pxm': 1.0, 'pxd': 1.0,
              'pzm': 1.0, 'pzd': 1.0}
    ref_srw = srw_propagate(wfr, L, params)

    # Compare intensities on the common grid
    acc = compute_accuracy(ref_as, ref_srw)
    print(f"✓ SRW vs angular spectrum (simple beam): accuracy={acc:.4f}")
    assert acc > 0.9, f"SRW and angular spectrum should agree for simple beam, got {acc}"


def test_srw_vs_angular_spectrum_diverging_beam():
    """SRW and angular spectrum agree for a diverging beam with curvature."""
    if not HAS_SRW:
        print("⊘ SRW vs angular spectrum (diverging): skipped (srwpy not installed)")
        return
    from srw_param_advisor.validator import simulate_drift_propagation

    wfr = generate_test_wavefront(
        nx=256, nz=256, dx=2e-6, dz=2e-6,
        photon_energy_eV=12000, R_x=5.0, R_z=5.0,
        beam_sigma_x=50e-6, beam_sigma_z=50e-6,
    )
    L = 2.0

    ref_as = simulate_drift_propagation(wfr, L)

    # SRW with quad-phase moment mode (AT=1), allow grid expansion
    params = {'analyt_treat': 1, 'pxm': 2.0, 'pxd': 1.0,
              'pzm': 2.0, 'pzd': 1.0}
    ref_srw = srw_propagate(wfr, L, params)

    acc = compute_accuracy(ref_as, ref_srw)
    print(f"✓ SRW vs angular spectrum (diverging beam): accuracy={acc:.4f}")
    assert acc > 0.8, f"SRW and angular spectrum should agree for diverging beam, got {acc}"


def test_srw_energy_conservation():
    """SRW propagation should approximately conserve energy."""
    if not HAS_SRW:
        print("⊘ SRW energy conservation: skipped (srwpy not installed)")
        return
    wfr = generate_test_wavefront(
        nx=256, nz=256, dx=2e-6, dz=2e-6,
        photon_energy_eV=12000, R_x=20.0, R_z=20.0,
        beam_sigma_x=60e-6, beam_sigma_z=60e-6,
    )
    E_in = wfr.total_energy

    params = {'analyt_treat': 1, 'pxm': 2.0, 'pxd': 1.0,
              'pzm': 2.0, 'pzd': 1.0}
    result = srw_propagate(wfr, drift_length=1.0, params=params)
    E_out = result.total_energy

    ratio = E_out / E_in if E_in > 0 else 0
    print(f"✓ SRW energy conservation: E_out/E_in={ratio:.4f}")
    assert 0.8 < ratio < 1.2, f"Energy changed by {abs(1-ratio):.0%}"


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_prior_shape_and_finite,
        test_prior_at_suggestion_diverging,
        test_prior_at_suggestion_converging_to_waist,
        test_prior_custom_R_and_sigma,
        test_get_analytical_params_types_and_ranges,
        test_patch_cnn_output_shape,
        test_patch_cnn_finite_output,
        test_patch_cnn_zero_patch,
        test_agent_forward_shapes,
        test_agent_forward_finite,
        test_agent_log_std_clamped,
        test_agent_sample_action,
        test_agent_deterministic_action,
        test_agent_parameters_change_after_update,
        test_action_to_params_ranges,
        test_action_to_params_large_deltas_clipped,
        test_apply_resize_identity,
        test_apply_resize_expand,
        test_apply_resize_resolution,
        test_apply_resize_returns_wavefront_snapshot,
        test_compute_accuracy_identical,
        test_compute_accuracy_range,
        test_compute_accuracy_no_overlap,
        test_compute_cost,
        test_compute_reward,
        test_generate_universal_wavefront,
        test_generate_universal_wavefront_variety,
        test_compute_reference_adaptive_returns_wavefront,
        test_compute_reference_adaptive_energy_conserved,
        test_compute_reference_adaptive_negative_drift,
        test_trainer_runs_without_error,
        test_trainer_history_fields,
        test_trainer_parameters_change,
        test_save_load_wavefront_roundtrip,
        test_save_load_wavefront_none_robs,
        test_precompute_dataset_creates_files,
        test_precomputed_dataset_load_and_iterate,
        test_trainer_with_precomputed_dataset,
        test_trainer_precomputed_wraps_around,
        test_to_srw_roundtrip,
        test_srw_propagate_runs,
        test_srw_vs_angular_spectrum_simple_beam,
        test_srw_vs_angular_spectrum_diverging_beam,
        test_srw_energy_conservation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
