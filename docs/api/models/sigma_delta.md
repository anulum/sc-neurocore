# SigmaDeltaNeuron

`SigmaDeltaNeuron` is the source-catalogue implementation of Young C. Yoon's
asynchronous pulse sigma-delta interpretation of spiking neurons. The public
implementation is an explicit clocked specialization of the encoder equations
disclosed in WO2016022241A1, not a claim of exact continuous-time event timing.

## Sampled equations

For input `I`, sample interval `dt`, reconstruction quantum `delta`, and
reconstruction time constant `tau_reconstruction`:

```text
sigma'         = sigma + dt * I
reconstruction_decay = reconstruction * exp(-dt / tau_reconstruction)
event          = 1 if sigma' - reconstruction_decay >= delta / 2 else 0
reconstruction'= reconstruction_decay + event * delta
```

The integrating prefilter corresponds to disclosed equation 20, the local
difference and reconstruction feedback to equations 21-26, the unipolar upper
quantizer to equation 27, and the exponential reconstruction to equation 40.
Paragraph 102 explicitly permits a discrete-time version.

Defaults are `sigma=0`, `reconstruction=0`, `delta=1`,
`tau_reconstruction=10`, and `dt=0.1`. Both state fields and every candidate
transition are validated before mutation. `reset()` clears the two states while
preserving configuration.

## Identity boundary

The former signed, one-quantum-per-sample accumulator is preserved exactly as
[`SCSigmaDeltaAccumulatorNeuron`](sc_sigma_delta_accumulator.md). It is a
count-neutral SC project model and is not an alias of this source identity.

## Evidence

- Python, modular Rust, Rust safety, Julia, Go, and Mojo implement the same
  complete state transition.
- The independent 512-step receipt records 276 unipolar events and trace
  SHA-256 `4c22a86d…b5b8d99`.
- Paired TOML/JSON schemas agree.
- The 200,000-step five-runtime benchmark records 199,839 exact events and
  zero observed state difference on the measured host.
- The signed-Q32.32 specialization matches its independent integer oracle,
  synthesizes in Yosys, and passes depth-12 CVC5 bounded safety.

The fixed-point and benchmark evidence is bounded local regression evidence;
it is not timing/PPA, board/device, biological-fit, or universal-equivalence
evidence.

Primary sources: [publisher record](https://doi.org/10.1109/TNNLS.2016.2526029)
and [inventors' equation disclosure](https://patents.google.com/patent/WO2016022241A1/en).
