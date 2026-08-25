<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# NagumoSatoMapNeuron

`NagumoSatoMapNeuron` is the source-faithful one-state refractory map of
Nagumo and Sato. It is distinct from the retained two-state SC project map.

## Source custody and equations

Nagumo and Sato introduced the response model in 1972
([DOI 10.1007/BF00290514](https://doi.org/10.1007/BF00290514)). Aihara's
accessible 1989 primary-author treatment reproduces the reduced equations as

\[
y_{t+1}=k y_t-\alpha H(y_t)+a+I_t,\qquad x_{t+1}=H(y_{t+1}),
\]

where `H(z)=1` for `z>=0`, including `H(0)=1`. `y` is the independent state;
`x` and the returned event are the same source-defined level output. The
implementation does not add a second adaptive state or reinterpret the level
as an upward crossing.

## API and complete receipts

```python
import numpy as np
from sc_neurocore.neurons.models import NagumoSatoMapNeuron

unit = NagumoSatoMapNeuron()
receipt = unit.simulate(0.05 * np.sin(np.arange(1_000) * 0.037), backend="auto")
print(receipt["y_final"], receipt["spike_count"])
```

`step()` validates before mutation. `simulate()` returns full `y`, derived `x`,
and `spikes` arrays plus final-state and count receipts. Python is the golden
lane; Rust/PyO3, Julia, Go, and Mojo execute the same batch contract. The first
four are binary64-exact over the enrolled trace, while Mojo remains within
`5e-15`; events are exact in every lane.

## Schema and hardware evidence

Paired TOML/JSON schemas preserve the same one-state recurrence. The committed
primary-equation trace independently iterates it for 1,000 nonconstant steps.
The Q16.16 RTL implements the piecewise-linear map, matches all enrolled event
decisions, and stays within the declared quantisation envelope. Icarus
co-simulation, Yosys synthesis, and a depth-12 SymbiYosys/Z3 induction proof
cover the maintained H1 surface; they are not formal real-number equivalence or
device timing/PPA claims.

## Evidence

- `tests/test_reference_nagumo_sato_map.py`: independent primary equation.
- `tests/test_nagumo_sato_map_backends.py`: complete five-runtime parity.
- `tests/test_cosim_nagumo_sato_map.py`: schema and Q16.16 RTL co-simulation.
- `benchmarks/results/bench_nagumo_sato_and_sc_adaptive_map.json`: pinned,
  source-bound local throughput and parity evidence.
