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

The independent 256-step receipt retains SHA-256
`8cb57c49…3ae25`, 54 positive events, zero negative events, and final
`sigma=0.40000000000000857`. Python, Rust, Julia, Go, and Mojo reproduce the
200,000-step constant-drive event vector exactly. The Q32.32 RTL matches its
integer oracle, synthesizes in Yosys, and passes depth-12 CVC5 bounded safety.

This SC compatibility identity does not add a literature model to the
155-model catalogue. Use [`SigmaDeltaNeuron`](sigma_delta.md) for the
source-bound sampled APSDM model.
