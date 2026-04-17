# World Model — Spike Prediction + Planning

Three components: (1) online-learnable spike predictor for codec integration, (2) stochastic state-transition model, (3) greedy action planner.

## SpikePredictor — Online Autoregressive Codec

The core workhorse. Predicts multi-channel spike patterns from recent history using a linear autoregressive model trained online via LMS (Least Mean Squares). No backprop, no batches — updates one sample at a time.

**Codec integration:** Encoder and decoder both maintain identical SpikePredictor instances. Both see the same history. Prediction error (XOR of actual vs predicted) is what gets transmitted. At the decoder, XOR recovers the original. Deterministic: same history → same prediction → lossless roundtrip.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_channels` | (required) | Number of spike channels |
| `history_len` | 8 | Context window (K past timesteps) |
| `lr` | 0.01 | LMS learning rate |
| `threshold` | 0.5 | Binary prediction threshold |

**Codec functions:**

- `predict_and_xor_world_model(spikes, n_channels, ...)` → (errors, correct_count) — Encoder
- `xor_and_recover_world_model(errors, n_channels, ...)` → spikes — Decoder

## PredictiveWorldModel — Linear Gaussian State-Space

Probabilistic predictive model implemented as a **Linear
Gaussian State-Space Model (LGSSM)** with Kalman filter
(forward), RTS smoother (backward), and EM parameter learner.
References: Kalman 1960, Rauch-Tung-Striebel 1965, Shumway &
Stoffer 1982, Bishop 2006 §13.3.

Provides `predict_next_state()` (deterministic mean),
`predict_next_state_with_cov()` (mean + covariance),
`forecast()` / `forecast_with_cov()` for multi-step rollouts.

The previous "linear transition matrix + clip-to-[0,1]"
implementation was a placeholder masquerading as a world
model and was replaced 2026-04-17 per
`feedback_sophisticated_from_start.md`. See [Predictive Model
detailed page](world_model/predictive_model.md) for the full
LGSSM + Kalman + RTS + EM derivations, performance numbers,
and the multi-language backend status.

## SCPlanner — Greedy Action Selection

Uses `PredictiveWorldModel` for random-shooting planning: sample N candidate actions, predict outcomes, pick the one closest to the goal state.

- `propose_action(current, goal, n_candidates)` — Best single action
- `plan_sequence(current, goal, horizon)` — Greedy multi-step plan

## Usage

```python
from sc_neurocore.world_model import SpikePredictor
from sc_neurocore.world_model.spike_predictor import (
    predict_and_xor_world_model,
    xor_and_recover_world_model,
)
import numpy as np

# Lossless codec roundtrip
spikes = (np.random.rand(100, 32) < 0.3).astype(np.int8)
errors, correct = predict_and_xor_world_model(spikes, n_channels=32)
recovered = xor_and_recover_world_model(errors, n_channels=32)
assert np.array_equal(spikes, recovered)  # Always true
print(f"Prediction accuracy: {correct / (100 * 32):.1%}")

# Planning
from sc_neurocore.world_model import PredictiveWorldModel, SCPlanner
model = PredictiveWorldModel(state_dim=4, action_dim=2)
planner = SCPlanner(world_model=model)
plan = planner.plan_sequence(
    current_state=np.array([0.1, 0.2, 0.3, 0.4]),
    goal_state=np.array([0.9, 0.8, 0.7, 0.6]),
    horizon=5,
)
```

::: sc_neurocore.world_model
    options:
      show_root_heading: true
