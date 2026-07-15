# GutkinErmentroutNeuron

**Module:** `sc_neurocore.neurons.models.gutkin_ermentrout`
**Rust:** `sc_neurocore_engine::neurons::simple_spiking::GutkinErmentroutNeuron`
**Reference:** Gutkin & Ermentrout 1998
**Family:** Conductance-based persistent-sodium model
**State variables:** `v` (voltage), `n` (delayed-rectifier K activation)

## Equations

The maintained SC-NeuroCore surface uses the reduced two-state
Gutkin-Ermentrout persistent-sodium model with implicit unit membrane
capacitance:

$$
\frac{dV}{dt} =
-g_{Na}m_\infty(V)(V-E_{Na}) - g_K n(V-E_K) - g_L(V-E_L) + I
$$

$$
\frac{dn}{dt} = n_\infty(V) - n
$$

The instantaneous gates are:

$$
m_\infty(V)=\frac{1}{1+\exp(-(V+20)/15)}
$$

$$
n_\infty(V)=\frac{1}{1+\exp(-(V+25)/5)}
$$

Spike output is the upward crossing event:

$$
V_t \geq V_\theta \land V_{t-\Delta t} < V_\theta
$$

Voltage is not reset by this model after the event marker.

## Integration Contract

Python, Rust engine, Rust safety, Go, Julia, and Mojo now advance the
same coupled ODE with a candidate-first fourth-order Runge-Kutta step.
The previous state is preserved unless the complete candidate is finite
and the potassium gate remains in `0 <= n <= 1`.

Runtime validation rejects:

- non-finite voltage, gate, current, reversal potential, timestep, or threshold;
- negative conductances;
- non-positive timestep;
- initial or candidate gate values outside `[0, 1]`;
- non-finite intermediate RK4 derivative stages.

Python raises `ValueError` for invalid constructor or runtime input.
Rust safety, Go, Julia, and Mojo fail closed through sentinel return
values or `NaN` candidates appropriate to their low-level surface.

## Parameters

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `v` | -65.0 | Voltage state |
| `n` | 0.1 | Potassium activation gate |
| `g_na` | 20.0 | Persistent sodium conductance |
| `g_k` | 10.0 | Potassium conductance |
| `g_l` | 8.0 | Leak conductance |
| `e_na` | 60.0 | Sodium reversal potential |
| `e_k` | -90.0 | Potassium reversal potential |
| `e_l` | -80.0 | Leak reversal potential |
| `dt` | 0.05 | RK4 integration step |
| `v_threshold` | -20.0 | Upward crossing spike threshold |

## Pipeline Coverage

```
GutkinErmentroutNeuron
├── Python: sc_neurocore.neurons.models.gutkin_ermentrout
├── Rust engine: engine/src/neurons/simple_spiking/gutkin_ermentrout.rs
├── Rust safety: src/sc_neurocore/accel/rust/safety/gutkin_ermentrout.rs
├── Go service: src/sc_neurocore/accel/go/services/gutkin_ermentrout.go
├── Julia mirror: src/sc_neurocore/accel/julia/neurons/gutkin_ermentrout.jl
├── Mojo kernel: src/sc_neurocore/accel/mojo/kernels/gutkin_ermentrout.mojo
├── Population / Network Python integration
└── Analysis: spike_count(), firing_rate(), isi()
```

## Test Coverage

| Surface | Verification |
|---------|--------------|
| Python model | Defaults, RK4 current balance, Euler separation, invalid constructor/runtime rejection, state preservation, reset, deterministic trace, dynamics, parameter sweeps, population/network/analysis integration |
| Rust engine | RK4 current-balance parity, invalid-candidate preservation, reset, finite long-run, NaN no-panic |
| Rust safety | Valid-state contract, RK4 step, invalid-current state preservation |
| Go service | RK4 current-balance parity, invalid-state and invalid-current fail-closed preservation, benchmark hook |
| Julia mirror | RK4 step smoke check and benchmark execution |
| Mojo kernel | `next_v`, `next_n`, and spike event kernel smoke check and benchmark execution |

## Measured Local Benchmark

Benchmark artefact:
`benchmarks/results/local_python_2026-06-18_gutkin_ermentrout_rk4.json`

Command:
`PYTHONPATH=src .venv/bin/python benchmarks/bench_model_gutkin_ermentrout.py`

Evidence class:
`local_regression_non_isolated`; no production speed or hardware
measurement claim is made. The Rust row recorded high concurrent host
load (`load_average_before`: `4.31 31.41 63.88`) and no runtime
cpuset shield claim, so the timing medians below are regression context
only.

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 3804.339620 | 3756.070690 | 3935.855725 | 662 |
| Rust | 168.049885 | 163.083295 | 202.456505 | 662 |
| Go | 172.600000 | 169.700000 | 174.400000 | 662 |
| Julia | 117.993155 | 116.740550 | 120.401650 | 662 |
| Mojo | 135.060575 | 134.558150 | 136.721020 | 662 |

All measured backends used 200,000 steps, five repeats, and `current = 5.0`.
Spike-count parity is exact across Python, Rust, Go, Julia, and Mojo for
this local regression workload.
