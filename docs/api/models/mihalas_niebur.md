# MihalasNieburNeuron

**Module:** `sc_neurocore.neurons.models.mihalas_niebur`
**Reference:** Mihalaş & Niebur (2009), DOI `10.1162/neco.2008.12-07-680`, equations 2.1–2.2 and Table 1
**State:** `v`, `theta`, `i1`, `i2`

## Mathematical contract

Rates are represented per millisecond, voltage in volts, and the input/current states after division by capacitance. The fixed-grid classical RK4 integrator and sampled candidate event are declared numerical specialisations.

$$
\dot I_j=-k_jI_j,\qquad
\dot V=I+I_1+I_2-\frac{G}{C}(V-E_L),\qquad
\dot\Theta=a(V-E_L)-b(\Theta-\Theta_\infty).
$$

When the finite RK4 candidate satisfies $V\geq\Theta$, equation 2.2 is applied directly:

$$
I_j\leftarrow R_jI_j+A_j,\qquad
V\leftarrow V_r,\qquad
\Theta\leftarrow\max(\Theta_r,\Theta).
$$

`theta_reset > v_reset` is enforced as required below equation 2.2. The previous candidate-proportional voltage reset is not attributed to this paper; it remains available count-neutrally as `SCScaledResetAdaptiveIFNeuron`.

## Parameters and defaults

The default state is `(-0.07, -0.05, 0, 0)` for `(v, theta, i1, i2)`. Common
Table-1 defaults are `E_L=V_r=-0.07 V`, `Theta_r=-0.06 V`,
`Theta_inf=-0.05 V`, `G/C=0.05 ms^-1`, `b=0.01 ms^-1`,
`k1=0.2 ms^-1`, `k2=0.02 ms^-1`, `R1=0`, and `R2=1`. The default
`a=0.005 ms^-1` selects panel C; event jumps default to zero and panel M sets
`A1/C=0.01 V/ms`, `A2/C=-0.0006 V/ms`.

## Runtime and safety contract

Python, production Rust/PyO3, Rust safety, Julia, Go, and Mojo implement the same four-state update. Constructor and step validation reject non-finite values, non-positive rates/timestep, and invalid reset ordering. A non-finite candidate raises `FloatingPointError`; one-step and batch failures preserve the caller-visible state.

The independent panel-M receipt uses `I=0.002`, `A1/C=0.01`, and `A2/C=-0.0006` for 2,000 intervals. It records 14 events, first at zero-based index 146, and trace SHA-256 `fa3871a…d8d5`.

## Hardware evidence

The paired TOML/JSON schemas match the public class state and complete event sequence. Committed Q32.32 RTL preserves the complete 2,000-step, 14-event panel-M vector and keeps all four states within `1.3e-6` of binary64. The RTL compiles in Icarus, synthesises for Xilinx 7-series in Yosys, and its source-profile depth-2 SymbiYosys/Z3 job proves the public reset-spike safety property. This is H2 evidence, not timing closure, formal numerical equivalence, PPA, board, device, or physical-silicon evidence.

## Usage

```python
from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron

neuron = MihalasNieburNeuron(current_jump_1=0.01, current_jump_2=-0.0006)
trace, events = neuron.simulate(2_000, current=0.002, backend="auto")
```

`auto` selects the production Rust batch kernel when installed. The focused non-isolated 200,000-step panel-M benchmark measured exact Python/Rust/Julia/Go traces; Mojo differed by `2.78e-17` with identical events. See `benchmarks/results/bench_mihalas_niebur_simulate.json`.

## Focused verification

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/test_model_mihalas_niebur.py tests/test_mihalas_niebur_backends.py tests/test_mihalas_niebur_engine_binding.py tests/test_reference_mihalas_niebur.py tests/test_cosim_mihalas_niebur.py
cargo test --manifest-path src/sc_neurocore/accel/rust/Cargo.toml mihalas_niebur
(cd hdl/formal/catalogue && sby -f sc_mihalasnieburneuron.sby)
```
