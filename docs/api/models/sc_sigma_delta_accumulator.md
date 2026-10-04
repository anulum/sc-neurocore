# SCSigmaDeltaAccumulatorNeuron

`SCSigmaDeltaAccumulatorNeuron` preserves the historical SC-NeuroCore bipolar
accumulator that was formerly exposed under the source-facing sigma-delta name.
It is project-defined and has no external paper attribution.

```text
candidate = sigma + I
if candidate >= v_threshold:  event = +1; candidate -= v_threshold
elif candidate <= -v_threshold: event = -1; candidate += v_threshold
else: event = 0
sigma = candidate
```

At most one signed event is emitted per sample. Threshold excess remains in
the state, so sustained `|I| > v_threshold` can grow the residual. This is a
frozen compatibility behavior, not an implementation of the source APSDM
feedback system.

## Native Python contract

The native `sc_neurocore_engine.SCSigmaDeltaAccumulatorNeuron` accepts
`sigma=0.0` and `v_threshold=1.0`, positionally or by keyword. Residual state
must be finite and the threshold finite and positive. Finite residuals have
no additional magnitude cap. `get_state()` returns a detached `sigma`
dictionary; `reset()` clears the residual while retaining the threshold.
Nonfinite current or candidate overflow raises `ValueError` without committing
state. Class pickle identity is retained; native instances refuse pickle/copy.

`py_sc_sigma_delta_accumulator_simulate` takes the two configuration fields
and an aligned, contiguous, native-endian one-dimensional `float64` NumPy
input. Readonly input is accepted. It returns independent writable NumPy
owners: a `float64` residual trace, `int32` signed events and `sigma_final`.
Empty input retains the initial residual, including its zero sign.
Configuration and layout are checked before output allocation. Allocation
failure raises `MemoryError`, releases partial output and allows retry;
the input remains unchanged. Real Linux address-space-pressure contracts
exercise both output allocations and layout refusal under the same limit.
This evidence is bounded to the named installed Linux wheel profile.

The independent 256-step receipt retains SHA-256
`8cb57c49…3ae25`, 54 positive events, zero negative events, and final
`sigma=0.40000000000000857`. Python, Rust, Julia, Go, and Mojo reproduce the
200,000-step constant-drive event vector exactly. The Q32.32 RTL matches its
integer oracle, synthesizes in Yosys, and passes depth-12 CVC5 bounded safety.

This SC compatibility identity is separate from the literature models in the
source catalogue. Use [`SigmaDeltaNeuron`](sigma_delta.md) for the
source-bound sampled APSDM model.
