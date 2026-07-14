# COBALIFNeuron

`COBALIFNeuron` implements the conductance-based leaky integrate-and-fire cell
from Appendix 2, Benchmark 1 of Brette et al. (2007). The primary source is
*Simulation of networks of spiking neurons: a review of tools and strategies*,
[doi:10.1007/s10827-007-0038-6](https://doi.org/10.1007/s10827-007-0038-6).

The source specifies the continuous equations and benchmark constants. SC-NeuroCore
advances those equations with a maintained coupled classical RK4 step; RK4 is a
repository discretisation, not a numerical method prescribed by the paper.

## Continuous equations

The membrane and conductance dynamics are

\[
C_m\frac{dV}{dt} = -g_L(V-E_L)-g_e(V-E_e)-g_i(V-E_i)+I,
\]

\[
\frac{dg_e}{dt}=-\frac{g_e}{\tau_e},\qquad
\frac{dg_i}{dt}=-\frac{g_i}{\tau_i}.
\]

Excitatory and inhibitory boundary events are instantaneous non-negative
increments applied before integration:

\[
g_e\leftarrow g_e+\Delta g_e,\qquad
g_i\leftarrow g_i+\Delta g_i.
\]

The biological state is `(v, g_e, g_i, refractory_time)`. A threshold crossing
loads the absolute refractory timer and resets only voltage; both conductances
continue to decay while voltage is held at reset.

## Factory contract

| Field | Default | Unit | Meaning |
|---|---:|---|---|
| `v` | -60.0 | mV | membrane voltage |
| `g_e` | 0.0 | nS | excitatory conductance |
| `g_i` | 0.0 | nS | inhibitory conductance |
| `refractory_time` | 0.0 | ms | remaining absolute refractory interval |
| `c_m` | 200.0 | pF | membrane capacitance |
| `g_l` | 10.0 | nS | leak conductance |
| `e_l` | -60.0 | mV | leak reversal potential |
| `e_e` | 0.0 | mV | excitatory reversal potential |
| `e_i` | -80.0 | mV | inhibitory reversal potential |
| `tau_e` | 5.0 | ms | excitatory decay constant |
| `tau_i` | 10.0 | ms | inhibitory decay constant |
| `v_threshold` | -50.0 | mV | spike threshold |
| `v_reset` | -60.0 | mV | reset and refractory-hold voltage |
| `refractory_period` | 5.0 | ms | absolute refractory interval |
| `dt` | 0.1 | ms | RK4 macro timestep |

All maintained runtimes implement the same update order:

1. Validate the stored state, parameters, current, and conductance increments.
2. Apply `delta_ge` and `delta_gi` at the macro-step boundary.
3. If refractory, hold `v_reset`, RK4-decay both conductances, and decrement the
   timer. The round-off guard clamps the final `0.1 ms` interval to zero, giving
   exactly 50 held steps for the default `5 ms` period.
4. Otherwise, compute one coupled four-stage RK4 candidate for `(v, g_e, g_i)`.
5. Validate the raw candidate before threshold/reset, preserving failure
   atomicity.
6. Compare the raw voltage candidate with threshold. On a spike, retain the
   conductance candidates, reset voltage, and load `refractory_period`.
7. Commit all four state values together.

## Public simulation API

The scalar `step()` method accepts arbitrary event schedules:

```python
from sc_neurocore.neurons.models import COBALIFNeuron

neuron = COBALIFNeuron()
spike = neuron.step(current=650.0, delta_ge=0.15, delta_gi=0.07)
```

The public batch dispatcher preserves the complete numeric contract across the
Python, Rust engine/PyO3, Julia, Go, and Mojo implementations:

```python
trace, spike_count = neuron.simulate(
    n_steps=400,
    current=650.0,
    delta_ge=0.15,
    delta_gi=0.07,
    backend="auto",  # or python/rust/julia/go/mojo
)
```

`simulate()` applies constant boundary increments on every macro step. Use
`step()` when increments vary over time. A native call receives all four state
values and all eleven configurable model/timestep parameters; it commits the
returned state only after successful completion.

The `auto` order is Julia, Rust, Mojo, Go, then Python. That order is derived
from the committed warmed local regression measurement, not a general
production-speed claim. An explicitly requested unavailable native runtime
raises `RuntimeError` instead of silently changing backends.

## Schema, RTL, and formal boundary

The paired `src/sc_neurocore/neurons/model_schemas/coba_lif.toml` and
`src/sc_neurocore/neurons/model_schemas/coba_lif.json` schemas are structurally
identical. The equation compiler represents classical RK4 as a four-phase map
with explicit `base_*`, `last_k_*`, and `weighted_*` stage registers. These are
deterministic lowering registers, not additional biological state.

The generated 48-bit Q24.24 RTL retains the complete voltage, conductance,
refractory, and event datapath. Over the enrolled 400-step conductance-driven
protocol it preserves all six event indices exactly. The measured error bounds
are `1e-5 mV` for voltage, `5e-6 nS` for excitatory conductance, `3e-6 nS` for
inhibitory conductance, and `2e-6 ms` for the refractory timer.

The generated SymbiYosys depth-4 job passes with Z3. Its formal assertion is a
minimal reset-safety property; it is not presented as a proof of fixed-point
equivalence. Float64 schema parity and Q24.24 trajectory/event parity are
separate executed tests.

## Reference and acceleration evidence

The committed DOI trace uses `I=650`, `delta_ge=0.15`, and `delta_gi=0.07` for
400 steps. An independent test re-derives the coupled RK4 equations without
calling the production model, then checks every committed feature to `1e-12`.
The public schema runner produces six spikes at zero-based indices
`29, 103, 177, 251, 325, 399`.

The controlled 200,000-step benchmark exercises the complete non-default ABI
on one pinned, non-exclusive logical CPU. Every backend produces 3,077 events;
Rust, Julia, and Go are bit-identical to Python for the enrolled trace, while
Mojo differs by at most `7.11e-15`. The recorded local median order is Julia,
Rust, Mojo, Go, Python. The artifact explicitly sets
`production_speed_claim=false` and `hardware_measurement_claimed=false`.

The Rust safety module is also compiled and tested independently: five tests
cover coupled conductance injection, raw-candidate validation, failure
atomicity, refractory clamping, and threshold/reset behaviour.

See [COBA-LIF validation evidence](../../validation/coba_lif_cosim_fidelity.md)
for the exact evidence surfaces and focused commands, and
[Model Fidelity & Polyglot-Completion Status](../model_fidelity_status.md) for
the catalogue-wide graduation bar.

## Maintained limitations

- The standard scalar network pipeline supplies `current`; arbitrary synaptic
  conductance-event schedules currently use direct `step()` calls or a custom
  integration layer.
- The committed benchmark is a local regression record under non-exclusive CPU
  affinity. It does not establish deployment-wide throughput.
- The formal job proves reset safety at bounded depth. Numerical equivalence is
  established by the executed schema and RTL co-simulation tests, within their
  declared domains and tolerances.
