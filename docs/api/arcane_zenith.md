# ArcaneZenith Cognitive Core

A self-improving cognitive primitive. Couples the five-compartment
self-referential [`ArcaneNeuron`](neuron_models.md) to four reward-modulated
STDP plasticity rules so that the neuron's own meta-parameters (`tau_deep`,
`surprise_baseline`, `delta_conf`, `lr_base`) are themselves tracked by
plasticity, not held static.

```python
from sc_neurocore.arcane_zenith import create_arcane_neuron_with_zenith_plasticity

core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
for t in range(1000):
    spike = core.step(current=stimulus[t])
print(f"identity_drift = {core.neuron.identity_drift:.4f}")
```

## What the module actually does

`ArcaneNeuron` is a cognition-oriented neuron whose "identity" lives in a
slow deep compartment `v_deep` that moves only when the neuron sees genuine
prediction-error novelty. Four of its scalar parameters control that
identity loop:

| Parameter            | Role in the neuron                                                      |
| -------------------- | ----------------------------------------------------------------------- |
| `tau_deep` (ms)      | time constant of the deep (identity) compartment                        |
| `surprise_baseline`  | sigmoid centre of the novelty detector                                   |
| `delta_conf`         | confidence-dependent threshold relaxation                               |
| `lr_base`            | base learning rate of the predictor self-model                          |

`ArcaneZenithCognitiveCore` allocates **one reward-modulated STDP layer per
parameter** (four layers, each of length 1). On every tick the layer weights
`w ∈ [0, 1]` are pushed through a sharpened sigmoid and mapped into each
parameter's biological range. The reward signal driving all four layers is
the neuron's current novelty, so the whole system is a closed self-tuning
loop:

```
  neuron.step(I)  ─┬─►  spike
                   │
                   ├─►  novelty  ─►  reward for 4 plasticity rules
                   │
                   └─►  pre-activity proxy  ─►  pre_spike for 4 rules
                                              ▼
                          rules step, weights w drift
                                              ▼
                          sigmoid-map w into biological ranges
                                              ▼
                          write tau_deep, surprise_baseline,
                                delta_conf, lr_base  back onto neuron
```

## Sigmoid interpolator

The core maps each plasticity weight into a bounded physical range via

$$
\mathrm{map}(w, w_{\min}, w_{\max}) \;=\; w_{\min} + \sigma\big(10\,(w - 0.5)\big)\,(w_{\max} - w_{\min}),
\qquad \sigma(x) = \frac{1}{1+e^{-x}}.
$$

Key properties of the mapping, all verified in the test suite:

- **Endpoint at w = 0** — $\sigma(-5) \approx 0.0067$: result sits just above
  `min` and cannot undershoot it.
- **Endpoint at w = 1** — $\sigma(+5) \approx 0.9933$: result sits just below
  `max`.
- **Midpoint** — $w=0.5$ is the exact arithmetic mean of `min` and `max`.
- **Strict monotonicity** in `w` on `[0, 1]`.
- **Clamping** — weights outside `[0, 1]` (which can occur transiently
  inside the plasticity layer) still saturate cleanly at `min` / `max`.

The gain of `10` inside the sigmoid narrows the transition band to roughly
`w ∈ [0.3, 0.7]`, so the four meta-parameters don't oscillate across the
whole biological range from noise; they move only when the plasticity
layers accumulate directional evidence.

## Biological ranges

The four ranges baked into `step` are chosen to keep the neuron inside
its validated dynamical regime (the reference behaviour is covered by
`tests/test_model_arcane_neuron.py` in the repository).

| Parameter            | Range              | Reason the bounds matter                                          |
| -------------------- | ------------------ | ------------------------------------------------------------------ |
| `tau_deep`           | `[1000, 50000]` ms | at `<1000` ms the deep compartment collapses into working memory; at `>50000` ms it is effectively frozen |
| `surprise_baseline`  | `[0.01, 0.5]`      | sigmoid centre of the novelty detector — outside this band novelty either saturates or never fires       |
| `delta_conf`         | `[0, 1]`           | confidence threshold multiplier (negative values would flip threshold polarity)                          |
| `lr_base`            | `[0.001, 0.1]`     | predictor learning rate — below 0.001 the self-model never updates; above 0.1 it oscillates              |

The `TestStep` suite asserts each of these invariants across 200 steps of
driven input, and `TestIntegration.test_long_run_keeps_all_meta_parameters_bounded`
re-asserts them across 1000 steps of uniformly random input — any escape
signals a regression in either `_map_to_range` or the underlying plasticity
rule.

## API

### `ArcaneZenithCognitiveCore`

```python
class ArcaneZenithCognitiveCore:
    neuron: ArcaneNeuron
    tau_rule:  TorchRuleLayer | RustRuleLayer | RustWgpuRuleLayer
    nov_rule:  TorchRuleLayer | RustRuleLayer | RustWgpuRuleLayer
    conf_rule: TorchRuleLayer | RustRuleLayer | RustWgpuRuleLayer
    lr_rule:   TorchRuleLayer | RustRuleLayer | RustWgpuRuleLayer

    def __init__(self, backend: str = "torch", **kwargs) -> None: ...
    def step(self, current: float) -> int: ...
    def step_from_bio_rates(self, rates: dict[int, float]) -> None: ...
    def reset(self) -> None: ...
    def get_state(self) -> dict[str, Any]: ...
    def get_state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...
```

#### `step(current)`

Advances the simulation by one tick. Returns `1` if the neuron fired, else
`0`. Side-effects (in order):

1. `neuron.step(current)` — fast/working/deep compartments updated.
2. The four plasticity layers each take one step with `pre_spike`,
   `post_spike` and `reward = novelty`.
3. New layer weights are read and sigmoid-mapped onto
   `neuron.tau_deep`, `neuron.surprise_baseline`, `neuron.delta_conf`,
   `neuron.lr_base`.

#### `step_from_bio_rates(rates)`

Convenience adaptor for multi-channel biological rate inputs
(e.g. MEA channels). Takes a `{channel_id: rate_hz}` dictionary, computes
the arithmetic mean, and forwards to `step(mean)`. An empty dict is treated
as zero current.

#### `reset()`

Soft reset: clears the neuron's fast + working compartments and the
plasticity traces of all four layers. The deep compartment `v_deep` and
the plasticity weights themselves are preserved — they collectively
*are* the neuron's learned identity. Mirrors `ArcaneNeuron.reset()`
semantics.

#### `get_state()` / `get_state_dict()` / `load_state_dict()`

- `get_state()` returns a flat dict of human-readable scalars:
  `v_fast`, `v_work`, `v_deep`, `confidence`, `novelty`, `surprise`,
  `prediction`, `identity_drift`, `meta_lr`, `total_steps` from the
  neuron, plus `w_tau`, `w_nov`, `w_conf`, `w_lr` from the four
  plasticity layers.
- `get_state_dict()` / `load_state_dict()` round-trip the four plasticity
  layers' full internal state (weights + traces) through the backend's
  own serialiser. Tested in `TestSerialization.test_state_dict_roundtrip_restores_four_weights`.

### Factory

```python
def create_arcane_neuron_with_zenith_plasticity(
    backend: str = "torch", **kwargs
) -> ArcaneZenithCognitiveCore
```

Thin wrapper around `ArcaneZenithCognitiveCore(...)`. `backend` is one of
`"torch"` (default, pure PyTorch, no native deps), `"rust"` (Rayon CPU via
`libautonomous_learning`) or `"rust-wgpu"` (WGSL GPU via the same crate).
Extra `kwargs` are passed through to each plasticity layer's constructor.

## Integration with `BioHybridSession`

`BioHybridSession` (see [Bioware Interface](bioware.md)) accepts an optional
`zenith_core: ArcaneZenithCognitiveCore`. When present, each closed-loop
frame forwards the detected biological rates to
`zenith_core.step_from_bio_rates(...)`, so the neuron's meta-parameters
track the actual firing statistics of the wet culture in real time.

A full end-to-end demo lives at `examples/14_bioware_closed_loop_demo.py`
in the repository.

## Limitations and caveats

- The four plasticity layers all have **count = 1**. The mapping is scalar
  per meta-parameter; there is no per-synapse control.
- `reset()` clears traces but not plasticity weights. To wipe the learned
  identity (rarely what you want), reconstruct the core.
- The `"torch"` and `"rust"` backends are numerically close but not
  bit-identical — floating-point path lengths differ and the sigmoid
  mapping amplifies small weight drifts into small range drifts. Neither
  is the reference; both are covered by the same behavioural tests.
- `step_from_bio_rates` reduces rates to their arithmetic mean. Richer
  reductions (e.g. population vector, spectral features) should be
  computed by the caller and forwarded to `step(current)` directly.

## Reference

Module: `src/sc_neurocore/arcane_zenith.py`.
Tests: `tests/test_arcane_zenith/test_arcane_zenith.py` (28 tests).

::: sc_neurocore.arcane_zenith
