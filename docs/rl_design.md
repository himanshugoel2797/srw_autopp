# Reinforcement Learning for SRW Propagation Parameter Optimization

## 1. The Clean Formulation

The problem has a natural RL structure with an unusually clean reward signal:

**Ground truth is obtainable.** For any wavefront + drift configuration, we can
run SRW at very high resolution (oversampled grid with generous range) to produce
a reference result. This is expensive — but it's a training cost, not an inference
cost. Once the agent is trained, it predicts parameters instantly.

**The reward is objective and verifiable.** No proxy metrics, no validator
heuristics, no assumptions about what "good" looks like. Direct pixel-level
comparison against the known-good reference, penalised by computational cost.

**The problem is a contextual bandit.** Single-step decision (observe wavefront →
choose parameters → get reward), no sequential structure. This is simpler than
full RL and avoids challenges like credit assignment, temporal discounting, and
exploration-exploitation in long horizons.

## 2. Formal Problem Definition

### State (Context)

The agent observes two types of information:

**Spatial maps** (from the ViT architecture):
```
s_spatial = (5, H, W)  — patchified into 128×128 patches
  ch0: normalised intensity
  ch1: θ_x / θ_nyquist (phase gradient in Nyquist fractions)
  ch2: θ_z / θ_nyquist
  ch3: sampling quality map
  ch4: validity mask
```

**Analytical prior** (from ABCD beamline trace):
```
s_prior = (R_x, R_z, σ_x, σ_z, waist_dist_x, waist_dist_z,
           AT_suggested, pxm_0, pxd_0, pzm_0, pzd_0, drift_length)
```

### Action

The agent outputs a hybrid discrete-continuous action:

```
a = (AT_mode, Δlog_pxm, Δlog_pxd, Δlog_pzm, Δlog_pzd)
```

Where:
- `AT_mode` ∈ {0, 1, 2, 3, 4, 5} — full SRW AnalTreatment parameter:
  - 0: Standard angular representation
  - 1: Analytical quad-phase, moment-based radius estimation
  - 2: Analytical quad-phase, error-bar-based radius estimation
  - 3: Propagation from waist
  - 4: Propagation to waist
  - 5: Propagation to waist, beyond paraxial approximation
- `Δlog_p*` — log-space corrections to the analytical resize suggestions

The agent selects the mode and resize parameters jointly. This is critical
because the optimal resize depends on the mode:
- AT=0 needs enough resolution to sample the full quadratic phase → large pxd
- AT=1 subtracts the quadratic phase first → much less pxd needed
- AT=4 (to waist) changes the grid through propagation → different pxm logic
- AT=3 (from waist) starts at a waist → beam grows, needs large pxm

By making mode and resize part of the same action, the agent learns these
mode-dependent parameter relationships directly from the reward signal.

Final parameters:
```
analyt_treat = AT_mode
pxm = pxm_0 · exp(Δlog_pxm)
pxd = pxd_0 · exp(Δlog_pxd)
pzm = pzm_0 · exp(Δlog_pzm)
pzd = pzd_0 · exp(Δlog_pzd)
```

### Reward

```
R = accuracy(result, reference) - λ · cost(params)
```

Where:

**Accuracy** — normalised field correlation between the propagated result
and the high-resolution reference:

```python
def accuracy(result, reference):
    """
    Compare propagated wavefront against high-res reference.
    
    Uses complex field correlation, not just intensity — this catches
    phase errors that intensity comparison would miss.
    
    Both fields must be interpolated onto a common grid for comparison.
    Returns value in [0, 1] where 1 = perfect match.
    """
    # Interpolate result onto reference grid (or vice versa)
    E_res = interpolate_to_common_grid(result.Ex, result, reference)
    E_ref = reference.Ex
    
    # Complex correlation coefficient
    numerator = np.abs(np.sum(E_res * np.conj(E_ref)))**2
    denominator = np.sum(np.abs(E_res)**2) * np.sum(np.abs(E_ref)**2)
    
    if denominator == 0:
        return 0.0
    
    correlation = numerator / denominator  # ∈ [0, 1]
    return float(correlation)
```

**Cost** — normalised computational expense:

```python
def cost(params):
    """
    Computational cost relative to the base configuration.
    
    The dominant cost is the FFT, which scales as N·log(N).
    Total grid points = nx·pxm·pxd × nz·pzm·pzd.
    """
    grid_factor = params.pxm * params.pxd * params.pzm * params.pzd
    # Log scale so the penalty is proportional to relative increase
    return np.log(max(grid_factor, 0.1))
```

**λ (cost weight):** Controls the accuracy-vs-speed tradeoff.
- λ = 0: pure accuracy, no cost penalty → agent learns to always max out grid
- λ = 0.1: mild preference for efficiency
- λ = 0.5: strong efficiency pressure → agent finds minimum viable parameters
- Curriculum: start with λ=0 (learn what accuracy requires), gradually increase

### Episode Structure

```
1. Sample random wavefront configuration
2. Generate wavefront using universal parametric source
3. Compute reference: propagate at high resolution
4. Agent observes state (spatial maps + prior)
5. Agent selects action (parameters)
6. Propagate with selected parameters
7. Compute reward = accuracy(result, reference) - λ·cost(params)
8. Update agent
```

This is a single-step episode — pure contextual bandit. No temporal structure.

## 3. Why Contextual Bandit, Not Full RL

A contextual bandit is the right abstraction because:

- **No sequential decisions.** Each drift space in a beamline is independent:
  you choose parameters once, propagate once, done. There's no "next state"
  that depends on your current choice.

- **No delayed rewards.** The quality of the propagation is known immediately
  after the single propagation step.

- **No exploration-exploitation dilemma in the traditional sense.** The context
  (wavefront) is different every episode, so "exploring" a parameter choice for
  one wavefront doesn't inform choices for a different wavefront. The agent
  simply needs to learn the mapping from context to optimal action.

This means we can use simpler, more sample-efficient algorithms than full RL
(no need for PPO, SAC, etc. in their general form).

## 4. Agent Architecture

The agent reuses the ViT architecture from the previous design, with the
output head modified for the bandit setting:

```
┌─────────────────────────────────────────────────────┐
│  ViT Encoder (same as previous design)              │
│                                                      │
│  Spatial maps → Patches → CNN embedding              │
│  + Prior token                                       │
│  → Transformer (4 layers, 8 heads)                   │
│  → [prior_out, patch_attn_pool, patch_max_pool]      │
│  → Concat (3D = 768)                                │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Policy Head (stochastic for exploration)            │
│                                                      │
│  Dense 768→256, ReLU                                 │
│  Dense 256→128, ReLU                                 │
│                                                      │
│  ┌────────────────┐  ┌────────────────────────────┐  │
│  │ AT head        │  │ Resize head                │  │
│  │ Dense 128→3    │  │ Dense 128→8                │  │
│  │ → Categorical  │  │ → 4 means + 4 log-stds    │  │
│  │   distribution │  │ → 4 independent Gaussians  │  │
│  └────────────────┘  └────────────────────────────┘  │
│                                                      │
│  Action = (AT_sample, Δlog_pxm, Δlog_pxd,           │
│            Δlog_pzm, Δlog_pzd)                       │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Value Head (baseline for variance reduction)        │
│                                                      │
│  Dense 768→128, ReLU                                 │
│  Dense 128→1                                         │
│  → V(s) — predicted expected reward                  │
└─────────────────────────────────────────────────────┘
```

### Policy Distribution

**AT selection:** Categorical distribution over {agree, force_AT0, force_AT1}.
Temperature-controlled softmax for exploration.

**Resize corrections:** 4 independent Gaussian distributions, one per parameter.
The network outputs both mean μ and log-standard-deviation log(σ) for each.

```python
class PolicyHead(nn.Module):
    """
    Outputs a joint distribution over propagation mode (6-class categorical)
    and resize corrections (4 continuous Gaussians, conditioned on mode).
    
    Mode-conditional resize: each mode has different resize requirements.
    The network outputs mode-specific resize parameters, and the sampled
    mode selects which set to use. This lets the agent learn that AT=1
    needs less pxd than AT=0, that AT=4 needs different pxm logic, etc.
    """
    
    N_MODES = 6  # AT = {0, 1, 2, 3, 4, 5}
    N_RESIZE = 4  # (pxm, pxd, pzm, pzd)
    
    def __init__(self, input_dim=768):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        
        # Mode selection (6-class categorical)
        self.mode_logits = nn.Linear(128, self.N_MODES)
        
        # Mode-conditional resize parameters
        # Each mode gets its own mean/std predictions for the 4 resize params
        # This is key: AT=1 will learn different resize priors than AT=0
        self.resize_per_mode = nn.ModuleList([
            nn.Linear(128, self.N_RESIZE * 2)  # 4 means + 4 log_stds
            for _ in range(self.N_MODES)
        ])
        
        # Value baseline (mode-agnostic)
        self.value = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, features):
        h = self.shared(features)
        
        # Mode distribution
        mode_dist = Categorical(logits=self.mode_logits(h))
        
        # Per-mode resize distributions
        resize_dists = []
        for mode_head in self.resize_per_mode:
            params = mode_head(h)
            means = params[:, :self.N_RESIZE]
            log_stds = params[:, self.N_RESIZE:].clamp(-3, 1)
            resize_dists.append(Normal(means, log_stds.exp()))
        
        # Value
        V = self.value(features)
        
        return mode_dist, resize_dists, V
    
    def sample_action(self, features):
        mode_dist, resize_dists, V = self.forward(features)
        
        # Sample mode
        mode = mode_dist.sample()  # (batch,) ∈ {0,...,5}
        
        # Sample resize from the selected mode's distribution
        # Gather the distribution for the sampled mode
        batch_size = features.shape[0]
        resize_means = torch.stack([d.mean for d in resize_dists], dim=1)  # (B, 6, 4)
        resize_stds = torch.stack([d.stddev for d in resize_dists], dim=1)  # (B, 6, 4)
        
        # Index by sampled mode
        mode_idx = mode.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.N_RESIZE)
        selected_means = resize_means.gather(1, mode_idx).squeeze(1)  # (B, 4)
        selected_stds = resize_stds.gather(1, mode_idx).squeeze(1)    # (B, 4)
        
        selected_dist = Normal(selected_means, selected_stds)
        resize_action = selected_dist.rsample()  # reparametrised sample
        
        # Log probability: log p(mode) + log p(resize | mode)
        log_prob_mode = mode_dist.log_prob(mode)
        log_prob_resize = selected_dist.log_prob(resize_action).sum(-1)
        log_prob = log_prob_mode + log_prob_resize
        
        return mode, resize_action, log_prob, V
    
    def deterministic_action(self, features):
        """For inference: take the most probable mode, then mean resize for that mode."""
        mode_dist, resize_dists, V = self.forward(features)
        mode = mode_dist.probs.argmax(-1)
        
        # Get mean resize for the selected mode
        batch_size = features.shape[0]
        resize_means = torch.stack([d.mean for d in resize_dists], dim=1)
        mode_idx = mode.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.N_RESIZE)
        selected_means = resize_means.gather(1, mode_idx).squeeze(1)
        
        return mode, selected_means, V
    
    def entropy(self, features):
        """Total policy entropy (for exploration bonus)."""
        mode_dist, resize_dists, _ = self.forward(features)
        
        # Mode entropy
        mode_entropy = mode_dist.entropy()  # (B,)
        
        # Expected resize entropy across modes (weighted by mode probability)
        mode_probs = mode_dist.probs  # (B, 6)
        resize_entropies = torch.stack(
            [d.entropy().sum(-1) for d in resize_dists], dim=1
        )  # (B, 6)
        expected_resize_entropy = (mode_probs * resize_entropies).sum(-1)  # (B,)
        
        return mode_entropy + expected_resize_entropy
```

### Why Stochastic Policy

Exploration is critical: the agent needs to discover that e.g. pxm=3.0 works
better than pxm=1.0 for a particular wavefront type. The Gaussian policy
naturally explores around the current best guess, with the standard deviation
shrinking as the agent becomes more confident.

## 5. Training Algorithm: REINFORCE with Baseline

For a contextual bandit, REINFORCE (policy gradient with learned baseline)
is simple, well-understood, and sufficient. No need for actor-critic temporal
difference methods since there's no temporal structure.

```python
class BanditTrainer:
    def __init__(self, agent, lr=3e-4, gamma_cost=0.1):
        self.agent = agent
        self.optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=1e-4)
        self.gamma_cost = gamma_cost  # cost penalty weight λ
    
    def train_episode(self, wavefront, drift_length, reference):
        """
        One training episode:
        1. Agent observes wavefront
        2. Agent samples mode + resize parameters
        3. Propagate with sampled parameters
        4. Compute reward against reference
        5. Update agent via policy gradient
        """
        # Prepare state
        spatial = prepare_spatial_maps(wavefront)
        prior = compute_analytical_prior(wavefront, drift_length)
        
        # Agent samples action (mode + resize jointly)
        mode, resize_action, log_prob, V = self.agent.sample_action(
            spatial, prior
        )
        
        # Convert to SRW parameters
        params = action_to_params(mode, resize_action, prior)
        
        # Propagate with selected parameters
        try:
            result = srw_propagate(wavefront, drift_length, params)
            acc = accuracy(result, reference)
        except Exception:
            # Propagation failed (invalid params, memory error, etc.)
            # Strong negative signal
            acc = 0.0
        
        # Compute reward
        cst = cost(params)
        reward = acc - self.gamma_cost * cst
        
        # Policy gradient update
        advantage = reward - V.detach()  # baseline subtraction
        policy_loss = -(log_prob * advantage).mean()
        value_loss = F.mse_loss(V.squeeze(), torch.tensor(reward))
        
        # Entropy bonus from joint mode+resize distribution
        entropy = self.agent.entropy(torch.cat([spatial, prior]))
        
        loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'reward': reward,
            'accuracy': acc,
            'cost': cst,
            'mode': mode.item(),
            'params': params,
            'loss': loss.item(),
        }
```

### Batch Training (More Efficient)

In practice, collect a batch of episodes before updating:

```python
def train_batch(self, batch_size=32):
    """Collect batch_size episodes, update once."""
    log_probs = []
    rewards = []
    values = []
    
    for _ in range(batch_size):
        config = sample_random_config()
        wavefront = generate_wavefront(config)
        reference = compute_reference(wavefront, config.drift_length)
        
        spatial = prepare_spatial_maps(wavefront)
        prior = compute_analytical_prior(wavefront, config.drift_length)
        
        at_act, resize_act, lp, V = self.agent.sample_action(spatial, prior)
        params = action_to_params(at_act, resize_act, prior)
        result = srw_propagate(wavefront, config.drift_length, params)
        
        acc = accuracy(result, reference)
        cst = cost(params)
        r = acc - self.gamma_cost * cst
        
        log_probs.append(lp)
        rewards.append(r)
        values.append(V)
    
    # Stack
    log_probs = torch.stack(log_probs)
    rewards = torch.tensor(rewards)
    values = torch.cat(values).squeeze()
    
    # Normalise rewards (reduces variance)
    rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Advantages
    advantages = rewards_norm - values.detach()
    
    # Losses
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, rewards_norm)
    
    # Entropy bonus (encourages exploration)
    # Computed from the policy distribution, not shown for brevity
    entropy_bonus = ...
    
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
    
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
    self.optimizer.step()
    
    return {
        'mean_reward': rewards.mean().item(),
        'mean_accuracy': ...,
        'mean_cost': ...,
    }
```

## 6. Reference Computation Strategy

The reference is the expensive part. For each training wavefront, we need a
high-resolution SRW propagation that we trust is correct. Strategies to
manage this cost:

### What Makes a Good Reference

```python
def compute_reference(wavefront, drift_length):
    """
    Compute high-resolution reference propagation.
    
    Strategy: use generous resize factors and AnalTreatment=1,
    then verify with energy conservation and sampling checks.
    """
    # Very generous parameters
    ref_params = {
        'analyt_treat': 1,
        'pxm': 8.0,   # 8× grid range
        'pxd': 4.0,   # 4× resolution
        'pzm': 8.0,
        'pzd': 4.0,
    }
    
    result = srw_propagate(wavefront, drift_length, ref_params)
    
    # Verify reference quality
    validator = PropagationValidator()
    report = validator.validate(wavefront, result)
    
    if report.overall_quality < 0.95:
        # Reference itself is not good enough — increase further
        ref_params = {
            'analyt_treat': 1,
            'pxm': 16.0,
            'pxd': 8.0,
            'pzm': 16.0,
            'pzd': 8.0,
        }
        result = srw_propagate(wavefront, drift_length, ref_params)
        report = validator.validate(wavefront, result)
        
        if report.overall_quality < 0.90:
            # Skip this configuration — can't establish reference
            return None
    
    return result
```

### Amortising Reference Cost

References are expensive but reusable:

```python
class ReferenceCache:
    """
    Pre-compute and cache references for a bank of configurations.
    Each reference is computed once and reused across many training episodes.
    """
    
    def __init__(self, cache_dir="reference_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_or_compute(self, config_hash, wavefront, drift_length):
        cache_path = os.path.join(self.cache_dir, f"{config_hash}.npz")
        
        if os.path.exists(cache_path):
            return load_reference(cache_path)
        
        reference = compute_reference(wavefront, drift_length)
        if reference is not None:
            save_reference(cache_path, reference)
        
        return reference
```

### Training Data Scale

| Phase | Configurations | Reference cost per config | Total reference cost |
|-------|---------------|--------------------------|---------------------|
| Proof of concept | 500 | ~30s (toy propagator) | ~4 hours |
| Initial SRW | 5,000 | ~2 min (SRW, generous grid) | ~7 days (parallelisable) |
| Production | 50,000 | ~2 min | ~70 days → ~3 days on 24-core cluster |

References can be pre-computed in parallel across a cluster, then the RL
training runs on a single GPU reading from the reference cache.

## 7. Accuracy Metric — Complex Field Correlation

The accuracy metric deserves careful thought. Several options:

### Option 1: Complex field correlation (recommended)

```python
def complex_correlation(E_result, E_reference):
    """
    |⟨E_result | E_reference⟩|² / (⟨E_result|E_result⟩ · ⟨E_ref|E_ref⟩)
    
    Invariant to global phase shift.
    Catches both amplitude and phase errors.
    Returns 1.0 for perfect match, 0.0 for uncorrelated.
    """
    inner = np.sum(E_result * np.conj(E_reference))
    norm_res = np.sum(np.abs(E_result)**2)
    norm_ref = np.sum(np.abs(E_reference)**2)
    
    if norm_res == 0 or norm_ref == 0:
        return 0.0
    
    return float(np.abs(inner)**2 / (norm_res * norm_ref))
```

**Why this is right:** Propagation errors manifest as both intensity errors
(beam shape wrong) and phase errors (curvature wrong). Intensity-only metrics
would miss phase errors that produce correct-looking intensity but wrong
wavefront curvature — which would then fail at the next optical element.

### Option 2: Multi-scale accuracy (for extra robustness)

```python
def multiscale_accuracy(result, reference):
    """
    Combine correlation at multiple resolutions to catch both
    global structure errors and fine detail errors.
    """
    scales = [1, 2, 4, 8]  # downsampling factors
    weights = [0.4, 0.3, 0.2, 0.1]  # more weight on full resolution
    
    total = 0
    for scale, weight in zip(scales, weights):
        E_res_ds = downsample(result.Ex, scale)
        E_ref_ds = downsample(reference.Ex, scale)
        total += weight * complex_correlation(E_res_ds, E_ref_ds)
    
    return total
```

### Grid Interpolation for Comparison

The result and reference have different grids (different pxm/pxd). They must
be interpolated to a common grid for comparison:

```python
def interpolate_to_common_grid(result, reference):
    """
    Interpolate both fields to the finer of the two grids,
    cropped to the smaller of the two ranges.
    """
    # Common range: intersection of both grids
    x_min = max(result.x_start, reference.x_start)
    x_max = min(result.x_start + result.nx * result.x_step,
                reference.x_start + reference.nx * reference.x_step)
    z_min = max(result.z_start, reference.z_start)
    z_max = min(result.z_start + result.nz * result.z_step,
                reference.z_start + reference.nz * reference.z_step)
    
    # Common step: finer of the two
    dx = min(result.x_step, reference.x_step)
    dz = min(result.z_step, reference.z_step)
    
    # Interpolate both onto common grid
    from scipy.interpolate import RegularGridInterpolator
    
    x_common = np.arange(x_min, x_max, dx)
    z_common = np.arange(z_min, z_max, dz)
    
    E_res = interpolate_complex_field(result.Ex, result, x_common, z_common)
    E_ref = interpolate_complex_field(reference.Ex, reference, x_common, z_common)
    
    return E_res, E_ref
```

## 8. Curriculum Learning

Train in stages of increasing difficulty and increasing cost pressure:

### Stage 1: Learn What Accuracy Requires (λ = 0)

No cost penalty. The agent learns which parameters produce high accuracy
for each wavefront type. This is essentially supervised learning on the
reference data, but via policy gradient.

- Duration: ~1000 episodes
- Expected outcome: agent always picks large resize factors (safe but expensive)
- Purpose: establish that the ViT can read wavefront structure and map it to
  parameter requirements

### Stage 2: Introduce Mild Cost Pressure (λ = 0.05)

Small penalty for large grids. The agent starts finding cases where it can
reduce pxm/pxd without losing accuracy.

- Duration: ~2000 episodes
- Expected outcome: agent reduces parameters for easy cases (clean Gaussian)
  but keeps them large for hard cases (zone plate, figure errors)

### Stage 3: Full Cost Pressure (λ = 0.2)

Moderate penalty. The agent must find the Pareto-optimal tradeoff.

- Duration: ~5000 episodes
- Expected outcome: agent has learned per-wavefront-type parameter strategies
  that are significantly cheaper than the reference parameters while maintaining
  >95% accuracy

### Stage 4: Fine-tuning (λ = 0.1, diverse data)

Broaden the training distribution, lower cost pressure slightly, refine.

- Duration: ~5000 episodes
- Expected outcome: robust performance across all wavefront types

## 9. Exploration Strategy

### Initial Exploration

Large initial standard deviations on the Gaussian policy:
```python
# Initial log_std = 0 → std = 1.0 → explore ±1 in log space → 0.37× to 2.7× multiplicative
```

### Entropy Bonus

Add `- β · H(π)` to the loss, where H is the policy entropy. This prevents
premature convergence to a single parameter choice.

```python
def entropy_bonus(at_dist, resize_dist):
    """Encourage diverse actions."""
    at_entropy = at_dist.entropy().mean()
    resize_entropy = resize_dist.entropy().sum(-1).mean()
    return at_entropy + resize_entropy
```

β starts at 0.05 and decays to 0.001 over training.

### Mode Exploration

With 6 modes, the agent needs sufficient exploration to discover when each
mode is useful. The entropy bonus on the mode categorical distribution
prevents premature collapse to a single mode. Early in training, the agent
tries all 6 modes for each wavefront type and learns which ones get rewarded.

Expected mode discovery timeline:
- **Early training:** Agent tries all modes roughly equally. AT=0 and AT=1
  get positive rewards for most cases (they're the safest). AT=3, 4, 5 mostly
  fail (wrong context) but occasionally succeed spectacularly (near-waist cases).
- **Mid training:** Agent has learned the basic split — AT=1 for general use,
  AT=4 for converging beams approaching waist. AT=0 reserved for very
  well-sampled cases where it's cheapest. AT=3 for post-waist diverging beams.
- **Late training:** Agent fine-tunes the boundaries. Discovers that AT=5
  (beyond paraxial) gives better accuracy than AT=4 when angular spread is
  large. Learns that AT=2 is sometimes preferable to AT=1 when the analytical
  R estimate is uncertain (the sampling quality map is inconsistent with the
  prior R value).

The mode-conditional resize head means each mode develops its own resize
strategy independently:
- AT=0 resize head: learns to predict large pxd (needs to resolve full phase)
- AT=1 resize head: learns to predict small pxd (quadratic phase subtracted)
  but similar pxm (range requirements don't change)
- AT=4 resize head: learns different pxm logic (the grid transforms through
  the waist, final range depends on post-waist divergence)

This joint mode-resize learning is something no rule-based system captures —
the interactions are too complex to hand-code but emerge naturally from the
reward signal.

### Targeted Exploration for Hard Cases

If the agent consistently gets low reward for certain wavefront types,
increase the sampling frequency of those types:

```python
def adaptive_sampling(config_history, reward_history):
    """
    Sample more from configurations where the agent performs poorly.
    Priority proportional to (1 - mean_reward) for each wavefront type.
    """
    type_rewards = defaultdict(list)
    for config, reward in zip(config_history, reward_history):
        type_rewards[config.wfr_type].append(reward)
    
    weights = {}
    for wfr_type, rewards in type_rewards.items():
        mean_r = np.mean(rewards[-100:])  # last 100 episodes of this type
        weights[wfr_type] = max(1.0 - mean_r, 0.1)  # higher weight for poor performance
    
    # Normalise
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}
```

## 10. Inference

At inference time, the agent uses its deterministic policy (no sampling):

```python
class TrainedAgent:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()
    
    def predict(self, wavefront, drift_length, R_x, R_z, sigma_x, sigma_z):
        """
        Predict optimal propagation mode AND parameters.
        
        Returns the full AnalTreatment (0–5), resize parameters,
        confidence, and diagnostic information including the probability
        distribution over all modes.
        """
        # Prepare inputs
        spatial = prepare_spatial_maps(wavefront)
        prior = compute_analytical_prior(
            wavefront, drift_length, R_x, R_z, sigma_x, sigma_z
        )
        
        with torch.no_grad():
            mode_dist, resize_dists, V = self.model(spatial, prior)
        
        # Deterministic action (mode of distributions)
        mode = mode_dist.probs.argmax(-1).item()
        
        # Get resize params for selected mode
        resize_means = resize_dists[mode].mean[0].numpy()
        resize_stds = resize_dists[mode].stddev[0].numpy()
        
        # Mode probabilities (full distribution for diagnostics)
        mode_probs = mode_dist.probs[0].numpy()
        
        # Confidence: combination of mode certainty and resize precision
        mode_confidence = float(mode_probs[mode])  # how sure about the mode
        resize_confidence = float(np.exp(-resize_stds.sum()))  # how precise the resize
        overall_confidence = mode_confidence * resize_confidence
        
        # Convert to parameters
        prior_params = prior.to_dict()
        pxm = prior_params['pxm_0'] * np.exp(resize_means[0])
        pxd = prior_params['pxd_0'] * np.exp(resize_means[1])
        pzm = prior_params['pzm_0'] * np.exp(resize_means[2])
        pzd = prior_params['pzd_0'] * np.exp(resize_means[3])
        
        # Mode names for display
        MODE_NAMES = {
            0: "Standard angular representation",
            1: "Analytical quad-phase (moment-based)",
            2: "Analytical quad-phase (error-bar-based)",
            3: "Propagation from waist",
            4: "Propagation to waist",
            5: "To waist beyond paraxial",
        }
        
        return PredictionResult(
            analyt_treat=mode,
            mode_name=MODE_NAMES[mode],
            pxm=round(pxm, 3), pxd=round(pxd, 3),
            pzm=round(pzm, 3), pzd=round(pzd, 3),
            confidence=overall_confidence,
            expected_reward=V.item(),
            diagnostics={
                'mode_probabilities': {
                    MODE_NAMES[i]: float(p) for i, p in enumerate(mode_probs)
                },
                'analytical_suggestion': {
                    'AT': prior_params['AT_suggested'],
                    'pxm': prior_params['pxm_0'],
                    'pzm': prior_params['pzm_0'],
                },
                'resize_corrections': {
                    'Δlog_pxm': float(resize_means[0]),
                    'Δlog_pxd': float(resize_means[1]),
                    'Δlog_pzm': float(resize_means[2]),
                    'Δlog_pzd': float(resize_means[3]),
                },
                'resize_uncertainty': {
                    'σ_pxm': float(resize_stds[0]),
                    'σ_pxd': float(resize_stds[1]),
                    'σ_pzm': float(resize_stds[2]),
                    'σ_pzd': float(resize_stds[3]),
                },
            },
        )
```

## 11. Advantages Over Supervised Learning

The RL formulation has specific advantages over the supervised approach
we designed earlier:

| Aspect | Supervised (previous design) | RL (this design) |
|--------|----------------------------|-------------------|
| Ground truth | Validator heuristics (proxy) | Direct field comparison (exact) |
| Label quality | Only as good as the validator | As good as the reference propagation |
| Mode selection | Binary (AT=0 or AT=1) | Full 6-mode selection with mode-conditional resize |
| Optimisation target | Match the best-of-N candidates | Find truly optimal mode+params (continuous) |
| Mode-resize coupling | Independent (mode chosen, then resize chosen) | Joint (agent learns AT=1 needs less pxd than AT=0) |
| Cost awareness | Manual penalty term in labeling | Learned tradeoff via reward shaping |
| Exploration | Fixed candidate grid, may miss optima | Continuous exploration discovers novel parameter combos |
| Generalisation | Limited to training candidate grid | Continuous action space generalises to any parameters |
| New modes | Requires relabeling all training data | Just add to action space, agent discovers when to use them |

The supervised approach was limited by the discrete candidate grid — if the
optimal pxm was 1.7 but the grid only had {1.5, 2.0}, the label would be
one of those. The RL agent can discover 1.7 on its own.

## 12. Implementation Roadmap

### Phase 1: Infrastructure (1 week)
- ViT model with policy and value heads (PyTorch)
- `compute_reference()` with validation
- `accuracy()` with grid interpolation
- `cost()` function
- Reference cache system

### Phase 2: Proof of Concept with Toy Propagator (1 week)
- Train on universal parametric source + angular-spectrum propagator
- Verify that the agent learns to distinguish easy vs hard cases
- Verify that cost pressure reduces parameters for easy cases
- Compare against analytical estimator baseline

### Phase 3: SRW Integration (2–3 weeks)
- Wire to actual SRW propagation for both agent actions and references
- Pre-compute reference cache (parallelised across cluster)
- Train with curriculum (stages 1–4)
- Evaluate on held-out SRW configurations

### Phase 4: Evaluation and Deployment (1 week)
- Compare against: (a) analytical estimator, (b) supervised CNN, (c) expert humans
- Measure: accuracy at various cost budgets, failure rate, edge cases
- Export model, package, integrate with SRW Python interface

### Phase 5: Continuous Improvement
- Log deployed predictions and actual propagation outcomes
- Periodically retrain with new data
- Expand wavefront type coverage based on user feedback
