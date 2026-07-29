# KilincBhattMapNeuron

`KilincBhattMapNeuron` is an experimental two-state discrete map derived from
the Nagumo-Sato/Aihara sigmoid-map lineage. It is not a publication-exact model
and must not be attributed to a “Kilinc & Bhatt 2023” paper.

## Scientific custody

The implemented recurrence combines a sigmoid fast map with an
SC-NeuroCore-specific adaptive-threshold state. The relevant primary lineage
is:

- J. Nagumo and S. Sato, “On a response characteristic of a mathematical
  neuron model,” *Kybernetik* 10(3), 155–164 (1972).
- K. Aihara, T. Takabe, and M. Toyoda, “Chaotic neural networks,” *Physics
  Letters A* 144, 333–340 (1990),
  [doi:10.1016/0375-9601(90)90136-C](https://doi.org/10.1016/0375-9601(90)90136-C).

Neither paper defines this exact adaptive-threshold hybrid. The model therefore
remains explicitly experimental and derived.

## Canonical recurrence

For pre-update state `(x_n, theta_n)` and finite drive `I_n`:

\[
s_n = \frac{1}{1 + \exp[-4(x_n - \theta_n)]},
\]

\[
x_{n+1} = \operatorname{clamp}(-x_n + k s_n + I_n, -5, 5),
\]

\[
\theta_{n+1} = \operatorname{clamp}(\beta\theta_n
  + \gamma H(x_n-\theta_{spike}), -5, 5).
\]

The emitted event is an upward crossing by the fast state:

\[
e_n = [x_{n+1} \geq x_{threshold}]\,[x_n < x_{threshold}].
\]

The threshold update uses the pre-update level `x_n`; it is not driven by the
upward-crossing event. This distinction is part of the tested recurrence.

## Parameters and validity

| Field | Default | Valid range | Meaning |
|---|---:|---:|---|
| `x` | 0.0 | [-5, 5] | fast map state |
| `theta` | 0.0 | [-5, 5] | adaptive threshold state |
| `k` | 1.5 | [0, 5] | sigmoid gain |
| `beta` | 0.95 | [0, 1] | threshold retention per map step |
| `gamma` | 0.3 | [0, 2] | threshold increment above `theta_spike` |
| `theta_spike` | 0.8 | [0, 2] | level activating threshold adaptation |
| `x_threshold` | 0.8 | [0, 2] | upward-crossing event threshold |

All state, parameters, and inputs must be finite. Invalid configuration or
drive is rejected before state mutation. A non-finite candidate is likewise
rejected atomically. Finite candidates retain the model's explicit `[-5, 5]`
state clamp.

`reset()` clears only `x` and `theta`; it preserves configured parameters.

## Backend custody

| Surface | Source | Status | Executed evidence |
|---|---|---|---|
| Python reference | `src/sc_neurocore/neurons/models/kilinc_bhatt_map_neuron.py` | implemented | exact-file branch coverage |
| Production Rust/PyO3 | `engine/src/neurons/kilinc_bhatt_map.rs` | implemented | complete-state parity and Rust unit tests |
| Standalone safety Rust | `src/sc_neurocore/accel/rust/safety/kilinc_bhatt_map_neuron.rs` | implemented | compiled complete-state parity |
| Go | `src/sc_neurocore/accel/go/services/kilinc_bhatt_map_neuron.go` | implemented | compiled complete-state parity and Go unit tests |
| Julia | `src/sc_neurocore/accel/julia/neurons/kilinc_bhatt_map_neuron.jl` | implemented | executed complete-state parity |
| Mojo | — | **not implemented** | none |
| Generated RTL / silicon | — | **not implemented** | none |

Backend presence is not treated as proof. The parity suite executes the same 64
non-constant inputs through Python, production Rust, standalone safety Rust,
Go, and Julia and compares `(x, theta, event)` after every step with absolute
tolerance `2e-15`. Invalid-input atomicity is tested separately.

## Reproducibility anchor

At defaults, 1,000 steps with constant `I=0.6` produce:

- event count: 68
- final state: `x=0.4038531661502207`,
  `theta=0.8877927528402929`
- SHA-256 of the float64 `(x, theta, event)` trace:
  `e979a6cff86be008d8cff1ffdc9f333007c7826a045cf811cb522f36fa021e0e`

The descriptor at
`src/sc_neurocore/neurons/model_descriptors/KilincBhattMapNeuron.toml` stores
the same anchor.

## Usage

```python
from sc_neurocore.neurons.models import KilincBhattMapNeuron

neuron = KilincBhattMapNeuron()
events = [neuron.step(0.6) for _ in range(1_000)]
print(sum(events), neuron.x, neuron.theta)
```

## Known boundaries

- State variables and time are dimensionless; one call is one map iteration.
- The adaptive-threshold term is a repository-specific derivation, not an
  equation copied from the cited primary papers.
- No model-specific Mojo kernel, schema-to-RTL implementation, synthesis run,
  timing result, resource estimate, or silicon co-simulation is currently
  claimed.
- No throughput number is published without a committed benchmark artefact and
  exact host/toolchain custody.
