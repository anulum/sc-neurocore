# Nonlinear LIF neuron

`NonlinearLIFNeuron` implements a nonlinear leaky integrate-and-fire neuron with slow adaptation.

The membrane equation is

```text
c_m dV/dt = a(V - v_rest)(V - v_crit) - w + I
```

and the adaptation equation is

```text
tau_w dw/dt = b(V - v_rest) - w
```

## Contract

The implementation advances the two-state nonlinear ODE with a candidate-first
fourth-order Runge-Kutta step. The candidate is computed before mutation and is
committed only when both `v` and `w` remain finite. A spike is evaluated against
the RK4 membrane candidate, then the membrane voltage is reset while the
adaptation candidate is retained.

The implementation rejects invalid state before mutation:

- all state, parameter, and current values must be finite;
- voltage geometry must satisfy `v_rest < v_crit < v_threshold`;
- reset voltage must remain below threshold;
- nonlinear and adaptation gains are non-negative;
- `tau_w`, `c_m`, and `dt` are positive;
- `dt` must not exceed `tau_w`.

A spike resets only the dynamic membrane voltage to `v_reset`. `reset()` restores `v` to `v_rest` and `w` to `0.0` without changing model parameters.

## Polyglot surfaces

The Python, Rust safety, Go service, Julia, and Mojo acceleration surfaces carry
the same validation geometry and RK4 spike contract. Invalid accelerator inputs
fail closed and do not advance state.

## Local measured performance

Measured on `aaarthuus` on 2026-06-18 with
`benchmarks/results/local_python_2026-06-18_nlif_rk4.json`. This is a local,
non-isolated regression artefact and is not a production speed claim.

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 2902.702155 | 2664.224690 | 2938.295890 | 4563 |
| Rust safety | 26.980975 | 26.779290 | 27.431215 | 4563 |
| Go service | 66.090000 | 62.370000 | 67.460000 | 4563 |
| Julia kernel | 45.844030 | 45.595800 | 46.380100 | 4563 |
| Mojo kernel | 23.544875 | 21.593910 | 24.248970 | 4563 |

All measured mirrors emitted exactly 4,563 spikes over 200,000 steps at
`current=20.0`, giving zero-tolerance spike parity across the maintained
polyglot surfaces.

## Verification

The dedicated NLIF test module asserts configuration rejection, non-finite
current rejection before state mutation, reset parameter preservation,
candidate-first RK4 dynamics, and spike reset behavior. Current module test
count: 65.
