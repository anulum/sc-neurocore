# GLIFNeuron

**Module:** `sc_neurocore.neurons.models.glif`

**Reference:** Teeter et al., *Nature Communications* 9, 709 (2018), [doi:10.1038/s41467-017-02717-4](https://doi.org/10.1038/s41467-017-02717-4)

**States:** `v`, `theta_spike`, `i_asc1`, `i_asc2`, `theta_voltage`; runtime also exposes `refractory_remaining`

`GLIFNeuron` is the five-state GLIF5 point model. The former four-state project
recurrence remains available without publication attribution as
[`SCFourStateGLIFNeuron`](sc_four_state_glif.md).

## Equations

Between events,

$$
\dot V = \frac{I_e + I_1 + I_2 - (V-E_L)/R}{C},
\qquad \dot\Theta_s=-b_s\Theta_s,
$$

$$
\dot I_j=-k_jI_j,
\qquad
\dot\Theta_v=a_v(V-E_L)-b_v\Theta_v.
$$

The fixed-grid implementation integrates the linear interval exactly. It holds
the pre-step after-spike currents while evaluating the membrane and
voltage-threshold flow, matching the official AllenSDK exact-dynamics update
order. An event uses the strict source condition

$$V > \Theta_\infty + \Theta_s + \Theta_v.$$

The finite candidate then receives the fitted affine cuts

$$
V^+=E_L+f_v(V^- - E_L)-\delta_V,\quad
\Theta_s^+=\Theta_s^-+\delta_{\Theta_s},
$$

$$
I_j^+=f_jI_j^-+\delta_{I_j},\quad
\Theta_v^+=\Theta_v^-.
$$

The public state exposes the post-cut values immediately and holds them during
the configured refractory input-suppression interval. This sampled-state choice
is an explicit implementation specialization; sub-step spike interpolation is
not claimed.

## Parameters and defaults

The defaults are a source-consistent repository operating profile, not the fit
of a particular Allen Cell Types specimen. State, current, and voltage fields
use one internally consistent unit system.

| Field | Default | Contract |
|---|---:|---|
| `v` / `e_l` | -70 / -70 | initial voltage / leak reversal |
| `theta_spike` / `theta_voltage` | 0 / 0 | initial threshold components |
| `i_asc1` / `i_asc2` | 0 / 0 | initial after-spike currents |
| `refractory_remaining` | 0 | initial input-suppression time |
| `capacitance` / `resistance` | 10 / 1 | finite and strictly positive |
| `theta_inf` | -50 | baseline threshold |
| `b_spike` / `b_voltage` | 0.01 / 0.01 | strictly positive decay rates |
| `a_voltage` | 0.0001 | finite voltage-threshold coupling |
| `k_asc1` / `k_asc2` | 0.1 / 0.005 | strictly positive current-decay rates |
| `f_v` / `delta_v` | 0 / 0 | finite affine voltage-reset coefficients |
| `delta_theta_spike` | 2 | finite spike-threshold increment |
| `f_asc1` / `f_asc2` | 1 / 1 | finite current-retention factors |
| `delta_i_asc1` / `delta_i_asc2` | 1 / 0.5 | finite current increments |
| `refractory_period` | 2 | finite and non-negative |
| `dt` | 1 | finite and strictly positive |

## Runtime surfaces

The same complete state transition is implemented by Python, the production
Rust engine/PyO3 class and `NetworkRunner`, the standalone Rust safety mirror,
Julia, Go, and Mojo. `simulate(n_steps, current, backend=...)` returns the
post-step voltage trace and event count while committing all six runtime-state
fields. Invalid input and non-finite candidates fail before the Python object is
mutated.

```python
from sc_neurocore.neurons.models.glif import GLIFNeuron

neuron = GLIFNeuron()
trace, events = neuron.simulate(1000, current=30.0, backend="auto")
assert events == 49
assert neuron.refractory_remaining == 1.0
```

The 2,000,000-step loaded-host benchmark records 98,255 events in every lane.
Rust, Julia, and Go reproduce the Python voltage trace and final state exactly
on the measured host; Mojo's maximum complete-state/trace error is below
`1.5e-14`. These timings are local regression evidence, not isolated-core or
hardware performance claims. See `benchmarks/results/bench_glif_simulate.json`.

## Independent evidence

- `glif5_teeter_2018.json` independently re-derives a 512-step mixed-drive
  receipt from the five source equations. It records 15 strict events and hashes
  every complete state plus event byte.
- The paired TOML/JSON schemas reproduce the public five-state trajectory within
  `1e-12` and all event decisions exactly.
- The committed default-profile `sc_glif` RTL uses signed Q32.32 coefficients.
  Its integer trajectory is bit-exact to an independent integer oracle, its
  complete 512-step event vector matches the source model, and all five exposed
  source states remain within `2e-7`.
- Q32.32 RTL preserves the 1,000-step event counts `0/0/22/49/74/80` at
  `I=0/15/22/30/45/50`.
- Yosys `synth_xilinx` passes for `sc_glif`; the committed report records 2,129
  LUTs, 323 flip-flops, and 64 DSP48E1 cells in the coarse default mapping.
- The depth-6 SymbiYosys/Z3 job proves bounded reset, refractory-range, and
  event-reset safety after an observed reset. It does not prove real-number
  equation equivalence.

Timing, PPA, board, device, and physical-silicon validation remain open.

## Verification

```bash
PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_model_glif.py tests/test_glif_backends.py \
  tests/test_glif_engine_binding.py tests/test_reference_glif.py \
  tests/test_cosim_glif_q3232.py
cargo test --manifest-path engine/Cargo.toml glif --release
cargo test --manifest-path src/sc_neurocore/accel/rust/safety/Cargo.toml glif --release
(cd hdl/formal/catalogue && sby -f sc_glif.sby)
```
