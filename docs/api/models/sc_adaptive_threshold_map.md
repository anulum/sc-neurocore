<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# SCAdaptiveThresholdMapNeuron

`SCAdaptiveThresholdMapNeuron` preserves SC-NeuroCore's original bounded
two-state sigmoid/adaptive-threshold recurrence. It is a project model: it is
not attributed to Nagumo–Sato, Aihara, or a “Kilinc & Bhatt” publication.

## Project specification

For pre-update state `(x_t, theta_t)`:

\[
s_t=(1+\exp[-4(x_t-\theta_t)])^{-1},
\]
\[
x_{t+1}=\operatorname{clamp}(-x_t+k s_t+I_t,-5,5),
\]
\[
\theta_{t+1}=\operatorname{clamp}(\beta\theta_t+
\gamma H(x_t-\theta_{spike}),-5,5).
\]

The event is the upward crossing
`x_t < x_threshold <= x_(t+1)`. Threshold adaptation uses the pre-update level,
not that crossing event. Updates are simultaneous and invalid input or a
non-finite candidate leaves both states unchanged.

## API and complete receipts

```python
import numpy as np
from sc_neurocore.neurons.models import SCAdaptiveThresholdMapNeuron

unit = SCAdaptiveThresholdMapNeuron()
receipt = unit.simulate(0.6 + 0.25 * np.sin(np.arange(1_000) * 0.017))
print(receipt["x_final"], receipt["theta_final"], receipt["spike_count"])
```

Python, Rust/PyO3, Julia, Go, and Mojo all return complete `x`, `theta`, and
event traces with final-state/count receipts. Events are exact; compiled state
traces remain within the declared `1e-10` envelope. The deprecated
`KilincBhattMapNeuron` name resolves to this class for compatibility only.

## Schema and hardware evidence

Paired TOML/JSON schemas and the committed project-spec trace preserve the
simultaneous recurrence. Q8.24 RTL uses a 256-entry sigmoid LUT. Its committed
32-cycle `I=0.4` co-simulation has exact event decisions, `x` error below
`0.07`, and `theta` error below `2e-6`. Yosys synthesis and a depth-12
SymbiYosys/Z3 induction proof cover bounded reset/state/crossing properties.
This is H1 evidence, not formal real-number equivalence or device timing/PPA.

## Evidence

- `tests/test_model_sc_adaptive_threshold_map_neuron.py`: scalar and atomicity contracts.
- `tests/test_sc_adaptive_threshold_map_backends.py`: complete five-runtime parity.
- `tests/test_cosim_sc_adaptive_threshold_map.py`: paired schemas and Q8.24 RTL.
- `benchmarks/results/bench_nagumo_sato_and_sc_adaptive_map.json`: pinned,
  source-bound local throughput and parity evidence.
