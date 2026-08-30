# PerfectIntegratorNeuron

**Module:** `sc_neurocore.neurons.models.perfect_integrator`

**Source:** Naud and Gerstner (2012), section 1.1, [doi:10.1007/978-94-007-3858-4_6](https://doi.org/10.1007/978-94-007-3858-4_6)
**Family:** non-leaky integrate-and-fire

## Source equation and boundary

Naud and Gerstner define

$$\frac{dV}{dt}=\frac{I(t)}{C}$$

and reset when $V(t)>V_T$. For SC-NeuroCore's piecewise-constant input sample,
the implemented update is the exact integral

$$V_{n+1}=V_n+\frac{I_n\,\Delta t}{C},$$

not an Euler approximation. Equality with the threshold does not emit a source
event: a candidate must be strictly greater than $V_T$.

`PerfectIntegratorNeuron.naud_gerstner_2012()` constructs this count-bearing
source profile. The normalized defaults are maintained reproducibility choices,
not measured source constants.

## Preserved SC compatibility

`PerfectIntegratorNeuron()` remains backward compatible with the historical
inclusive `candidate >= v_threshold` recurrence. The same recurrence is exposed
unambiguously as `SCInclusivePerfectIntegratorNeuron`. Its paired schema and
formal RTL retain the `sc_perfect_integrator` name. No SC variant was removed or
silently relabelled.

The compatibility identity has its own
[descriptor and public model page](sc_inclusive_perfect_integrator.md).

At the exact-boundary protocol `I=5`, `dt=0.1`, `C=1`, `V_T=1`, the distinction
is executable:

| Profile | First three events | First three post-step voltages |
| --- | --- | --- |
| Naud-Gerstner source (`>`) | `0, 0, 1` | `0.5, 1.0, 0.0` |
| Preserved SC (`>=`) | `0, 1, 0` | `0.5, 0.0, 0.5` |

## Public API

```python
from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron

neuron = PerfectIntegratorNeuron.naud_gerstner_2012()
voltage, events = neuron.simulate_complete(1_000, 5.0, backend="auto")
trace, event_count = neuron.simulate(1_000, 5.0, backend="python")
```

`simulate_complete` returns aligned post-step `float64` voltage and `uint8`
event vectors. Python, Rust/PyO3, Julia, Go, and Mojo transport the complete
numeric contract, source selector, event vector, and final state. A backend
packet is validated before the Python instance commits its final voltage.
Go and Mojo also validate a complete dry run before writing caller buffers.

## Validation and evidence

- The DOI-bound source receipt pins the exact-integral voltage and event-vector
  SHA-256 digests for 1,000 equality-sensitive steps.
- The five public backends preserve the complete source packet, including the
  strict equality sequence and non-default state/parameter contracts.
- `perfect_integrator` TOML/JSON schemas encode the strict source boundary;
  `sc_perfect_integrator` TOML/JSON schemas preserve the inclusive SC boundary.
- Canonical NetworkRunner aliases select the source profile, while
  `SCInclusivePerfectIntegratorNeuron` selects the retained SC profile.
- Curated Q8.8 source RTL keeps `dt=1/10` as an exact rational so fixed-point
  rounding cannot erase the equality test. The original inclusive SC RTL remains
  tracked separately.
- The source core has tracked depth-20 formal safety and Yosys synthesis
  evidence (H2). Timing, PPA, target-device, board, and physical-silicon claims
  remain outside the evidence envelope.

Focused evidence lives in
`tests/test_reference_perfect_integrator_source_receipt.py`,
`tests/test_perfect_integrator_backend_parity.py`,
`tests/test_cosim_perfect_integrator.py`, and
`tests/test_bench_perfect_integrator.py`. The split Python behavioural cohort is
`tests/test_model_perfect_integrator_perfect_integrator_*.py`; the former
monolithic `tests/test_model_perfect_integrator.py` path no longer exists.

## Parameters and failure contract

| Parameter | Default | Contract |
| --- | ---: | --- |
| `v` | `0.0` | finite; source permits `v == v_threshold`, SC requires `v < v_threshold` |
| `c_m` | `1.0` | finite and positive |
| `v_threshold` | `1.0` | finite and greater than `v_reset` |
| `v_reset` | `0.0` | finite |
| `dt` | `0.1` | finite and positive |

Non-finite input, invalid runtime state, overflow, malformed output shape,
non-binary events, or trace/final-state disagreement fail before instance-state
commit. Explicit unavailable backend requests never fall back silently.

## Controlled local performance

The committed 2026-08-30 evidence ran 100,000 source-profile steps at `I=5`
for seven repeats, pinned to logical CPU 10 on an i5-11600K. The CPU was not
kernel-isolated and used the powersave governor, so these are reproducible local
regression timings rather than production-throughput claims. Every lane returned
33,333 identical events and bit-exact voltage traces.

| Backend | Median ms/call | Speedup vs Python | Maximum voltage difference |
| --- | ---: | ---: | ---: |
| Rust | 0.447 | 267.45x | `0` |
| Mojo | 1.232 | 96.98x | `0` |
| Julia | 1.371 | 87.18x | `0` |
| Go | 3.005 | 39.76x | `0` |
| Python | 119.498 | 1.00x | `0` |

The source-hashed artefact is
`benchmarks/results/bench_perfect_integrator.json`.
