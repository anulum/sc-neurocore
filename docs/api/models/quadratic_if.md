# QuadraticIFNeuron

**Module:** `sc_neurocore.neurons.models.quadratic_if`

**Source:** Latham et al. (2000), equations (1), (2), and (5a),
[doi:10.1152/jn.2000.83.2.808](https://doi.org/10.1152/jn.2000.83.2.808)

**Family:** quadratic integrate-and-fire / Type-I excitability

## Source equation and normalisation

After removing the paper's AHP and synaptic terms, the isolated voltage equation
is

$$
\frac{dV}{dt}=\frac{1}{\tau_{cell}}
\left[\frac{(V-V_r)(V-V_t)}{\Delta V}+\hat I_a\right],
\qquad \Delta V=V_t-V_r.
$$

With

$$
M=\frac{V_r+V_t}{2},\quad
x=\frac{2(V-M)}{\Delta V},\quad
s=\frac{t}{2\tau_{cell}},\quad
\eta=\frac{4\hat I_a}{\Delta V}-1,
$$

this becomes exactly

$$\frac{dx}{ds}=x^2+\eta.$$

The source's numerical values are `V_r=-65 mV`, `V_t=-50 mV`,
`V_apex=+20 mV`, `V_repol=-80 mV`, `tau_cell=10 ms`, and a `1 ms`
timestep. They map to:

| Meaning | Source value | Normalized value |
| --- | ---: | ---: |
| initial/rest | `-65 mV` | `-1` |
| unstable threshold | `-50 mV` | `+1` |
| event apex | `+20 mV` | `31/3` |
| reset/repolarization | `-80 mV` | `-3` |
| timestep | `1 ms` | `0.05` |

The important boundary is that `+1` is the unstable equilibrium, not the spike
apex. A source event occurs when the flow reaches the finite normalized apex
`31/3`, after which the state resets to `-3`.

## Public API and preserved compatibility

```python
from sc_neurocore.neurons.models.quadratic_if import (
    QuadraticIFNeuron,
    SCSymmetricQuadraticIFNeuron,
)

source = QuadraticIFNeuron.latham_2000()
voltage, events = source.simulate_complete(240, 4.0, backend="auto")

# Historical -1/+1/.01 SC behavior remains available.
retained = SCSymmetricQuadraticIFNeuron()
```

`QuadraticIFNeuron()` remains backward compatible with the historical
`v_reset=-1`, `v_peak=+1`, `dt=0.01` recurrence. The same recurrence has the
explicit count-neutral `SCSymmetricQuadraticIFNeuron` identity, paired
`sc_symmetric_quadratic_if` schemas, its original RTL/formal lane, and a
[separate public page](sc_symmetric_quadratic_if.md). No SC variant was removed
or silently relabelled.

Canonical NetworkRunner aliases construct `QuadraticIFNeuron.latham_2000()`;
`SCSymmetricQuadraticIFNeuron` selects the retained SC profile.

## Integration and failure contract

All five production runtimes use the exact analytic Riccati flow for a current
held constant over one step. This is a maintained higher-grade numerical
specialisation of the paper's ODE; the paper is not claimed to have used that
algorithm. The schema/RTL lane separately uses explicit Euler at the source
timestep and declares its fixed-point approximation envelope.

`simulate_complete()` returns aligned post-step `float64` voltage and `uint8`
binary event arrays. Python, production Rust/PyO3, Julia, Go, and Mojo carry the
same arbitrary finite state/parameter contract. Invalid step counts, parameters,
currents, intermediate values, packet shapes, non-binary events, or inconsistent
final state fail before the Python instance commits any mutation. The Go and
Mojo C ABIs validate a complete dry run before writing caller buffers.

## Independent source custody

The DOI-bound receipt
`src/sc_neurocore/neurons/reference_receipts/quadratic_if_latham_2000.json`
records the inspected PDF SHA-256, exact normalisation, numerical source values,
input digest, complete voltage digest, and event-vector digest. At `eta=4` for
240 exact held-current steps it records ten events at indices
`18, 42, ..., 234` and final state `-1.0483338948753533`.

The source `quadratic_if` TOML/JSON schemas and the preserved
`sc_symmetric_quadratic_if` pair are independently exercised. The source and SC
zero-current traces use the analytic solution `x(s)=-1/(1+s)` at their
respective timesteps.

## Polyglot and silicon evidence

The five runtime complete packets have exact event-vector parity and a maximum
enrolled cross-libm voltage tolerance of `2e-12`. Rust safety is exercised as a
separate executable lane.

The original `sc_quadratic_if` RTL remains assigned to the preserved symmetric
SC identity. A dedicated source-profile fixed-point core carries the Latham
apex/reset/timestep constants, co-simulation, tracked Yosys synthesis evidence,
and depth-20 reset/event safety. Yosys 0.33 reports 9,379 coarse cells. This
establishes H2 only inside the declared
fixed-point envelope; timing, PPA, target-device, board, physical-silicon, and
universal real-valued equivalence claims remain open.

## Controlled local benchmark

The source-hashed 100,000-step, seven-repeat run at `eta=5` produced 4,762
events in every lane. It was pinned to one logical CPU on a loaded, non-isolated
workstation, so these values are regression evidence rather than production
throughput claims.

| Backend | Median call ms | Speedup vs Python | Maximum voltage difference |
| --- | ---: | ---: | ---: |
| Rust | 4.141 | 27.47x | `0` |
| Julia | 6.011 | 18.92x | `0` |
| Go | 6.932 | 16.41x | `9.77e-15` |
| Mojo | 10.444 | 10.89x | `0` |
| Python | 113.738 | 1.00x | `0` |

The committed artefact is
`benchmarks/results/local_python_2026-06-16_quadratic_if_exact_flow.json`.

Focused executable evidence is in:

- `tests/test_reference_quadratic_if.py`
- `tests/test_quadratic_if_backends_rejects_and_flow.py`
- `tests/test_quadratic_if_backends_backend_parity.py`
- `tests/test_quadratic_if_backends_c_abi.py`
- `tests/test_cosim_quadratic_if.py`
- `tests/test_bench_quadratic_if.py`
