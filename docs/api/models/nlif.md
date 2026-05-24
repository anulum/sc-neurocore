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

The implementation rejects invalid state before mutation:

- all state, parameter, and current values must be finite;
- voltage geometry must satisfy `v_rest < v_crit < v_threshold`;
- reset voltage must remain below threshold;
- nonlinear and adaptation gains are non-negative;
- `tau_w`, `c_m`, and `dt` are positive;
- `dt` must not exceed `tau_w`.

A spike resets only the dynamic membrane voltage to `v_reset`. `reset()` restores `v` to `v_rest` and `w` to `0.0` without changing model parameters.

## Polyglot surfaces

The Rust, Go, Julia, and Mojo acceleration surfaces carry the same validation geometry and Euler-step spike contract. Invalid accelerator inputs fail closed and do not advance state.

## Verification

The dedicated NLIF test module asserts configuration rejection, non-finite current rejection before state mutation, reset parameter preservation, deterministic Euler dynamics, and spike reset behavior. Current module test count: 65.
