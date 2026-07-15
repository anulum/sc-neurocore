# Wong-Wang two-choice decision circuit

- **Class:** `WongWangUnit`
- **Module:** `sc_neurocore.neurons.models.wong_wang`
- **Source:** Wong and Wang (2006), Appendix reduced decision circuit
- **DOI:** [10.1523/JNEUROSCI.3733-05.2006](https://doi.org/10.1523/JNEUROSCI.3733-05.2006)

`WongWangUnit` advances two selective populations with NMDA gating and
Ornstein-Uhlenbeck AMPA input-current noise. It is a continuous reduced
mean-field model: the returned values are firing rates in hertz, not binary
spikes.

## Equations

For populations \(i\in\{1,2\}\), with \(j\ne i\),

\[
I_i=J_N S_i-J_{cross}S_j+I_0+I_{stim,i}+I_{noise,i},
\]

\[
r_i=\phi(I_i)=\max\left(0,
\frac{aI_i-b}{1-\exp[-d(aI_i-b)]}\right),
\]

\[
\frac{dS_i}{dt}=-\frac{S_i}{\tau_s}+(1-S_i)\gamma r_i,
\]

and the discrete AMPA noise update is

\[
I_{noise,i}^{n+1}=I_{noise,i}^{n}
-\frac{dt}{\tau_{AMPA}}I_{noise,i}^{n}
+\sqrt{\frac{dt}{\tau_{AMPA}}}\sigma\xi_i^n,
\]

where each \(\xi_i^n\) is a supplied standard-normal sample. The transfer
constants are \(a=270\,\mathrm{Hz/nA}\), \(b=108\,\mathrm{Hz}\), and
\(d=0.154\,\mathrm{s}\). The implementation evaluates the removable
singularity at \(aI-b=0\) as \(1/d\) and uses a stable `expm1` form elsewhere.

Rates use the complete pre-update state. The two gating candidates and two
noise candidates are then evaluated by explicit Euler, validated together,
and committed simultaneously. Invalid candidates fail without clipping or a
partial state write.

## Scientific boundary

The maintained equations are the reduced two-choice Appendix model without
recurrent AMPA. Gaussian generation stays outside the deterministic batch and
fixed-point boundaries: callers supply the two samples consumed by each
physical step.

The paper Methods specify a `0.1 ms` integration step. The pinned author-lab
trial script uses `0.5 ms`; SC-NeuroCore follows the paper value by default and
records this discrepancy in the source-reference artefact. The implementation
does not reproduce the script's one-index initial rate lag: each returned rate
is the direct algebraic response of the current pre-update state.

## Parameters and state

| Name | Default | Constraint | Meaning |
|---|---:|---|---|
| `s1` | `0.1` | finite in `[0, 1]` | population-one NMDA gating fraction |
| `s2` | `0.1` | finite in `[0, 1]` | population-two NMDA gating fraction |
| `noise1` | `0.0 nA` | finite | population-one OU current state |
| `noise2` | `0.0 nA` | finite | population-two OU current state |
| `tau_s` | `0.1 s` | finite, `> 0` | NMDA time constant |
| `tau_ampa` | `0.002 s` | finite, `> 0` | AMPA-noise time constant |
| `gamma` | `0.641` | finite, `> 0` | NMDA kinetic conversion factor |
| `j_n` | `0.2609 nA` | finite, `>= 0` | self-coupling strength |
| `j_cross` | `0.0497 nA` | finite, `>= 0` | cross-inhibition magnitude |
| `i_0` | `0.3255 nA` | finite | constant background current |
| `sigma` | `0.02 nA` | finite, `>= 0` | stationary OU amplitude |
| `dt` | `0.0001 s` | finite, `> 0` | explicit-Euler step |

## Scalar and deterministic-sample use

```python
import numpy as np

from sc_neurocore.neurons.models.wong_wang import WongWangUnit

unit = WongWangUnit()

# Stochastic scalar path: consumes two NumPy standard-normal samples.
rate1, rate2 = unit.step(stim1=0.02, stim2=-0.01)

# Deterministic scalar path for replay and parity.
rate1, rate2 = unit.step_with_gaussian_samples(
    stim1=0.02,
    stim2=-0.01,
    xi1=0.25,
    xi2=-0.5,
)

# Atomic batch: xi is interleaved [xi1_0, xi2_0, xi1_1, xi2_1, ...].
n_steps = 128
index = np.arange(n_steps, dtype=np.float64)
stim1 = 0.02 + 0.01 * np.sin(index * 0.07)
stim2 = -0.01 + 0.008 * np.cos(index * 0.11)
xi = np.sin(np.arange(2 * n_steps, dtype=np.float64) * 0.17)
result = unit.simulate(stim1, stim2, xi, backend="auto")

assert result["s1"].shape == result["r1"].shape == (n_steps,)
assert unit.s1 == result["s1_final"]
```

The batch result contains post-update traces `s1`, `s2`, `noise1`, and
`noise2`; pre-update rate traces `r1` and `r2`; and all four final states.
Explicitly requesting an unavailable compiled runtime raises rather than
substituting Python. The model instance changes only after the complete result
passes shared shape, finiteness, range, and trace/final consistency checks.

## Reset

`reset()` restores only `s1`, `s2`, `noise1`, and `noise2`. It preserves every
time constant, coupling, noise amplitude, and integration parameter.

## Executable runtimes

| Runtime | Maintained surface | Enrolled contract |
|---|---|---|
| Python | scalar model and public atomic batch | source reference |
| Rust engine | modular PyO3 batch | six traces and four final states within `1e-12` |
| Rust safety | independently compiled scalar module | Euler/OU state, transfer, validation, and reset |
| Julia | JuliaCall batch | six traces and four final states within `1e-12` |
| Go | C-shared ABI | six traces and four final states within `1e-12` |
| Mojo | exported shared-library C ABI | six traces and four final states within `1e-9` |

Native scalar contracts validate all inputs before writing caller-owned output
buffers. Zero-step batches preserve the four dynamic states.

## Reference and validation evidence

The committed 256-step source trace independently re-derives the Appendix
Euler/OU equations using varied stimuli and explicit samples. Its canonical
interleaved little-endian float64 SHA-256 is
`d39f219d3cd21d505c71749a1d9547d4cef550299f8e829bb2aa2a30d66daf44`.
The artefact pins the DOI, the author-lab repository, and source commit
`c39c6742329f89f1b5137f32910d55ad52d4bc24`.

The configured cross-runtime contract starts from non-default physical states,
uses non-default parameters, and compares every trace and final state over 128
steps. See [Wong-Wang source fidelity](../../validation/wong_wang_source_fidelity.md)
for the evidence matrix and reproduction commands.

## Fixed-point and hardware boundary

The paired declarative schemas serialise one physical update across six input
edges: two stimuli, two explicit samples, transfer evaluation, and simultaneous
state commit. Generated Q32.32 Verilog compiles and preserves a varied 32-step
trace within the declared state and rate envelopes. This is H1 co-simulation
evidence only. No synthesis, timing, formal-equivalence, device, or PPA result
is claimed.

## Reference

K.-F. Wong and X.-J. Wang, “A Recurrent Network Mechanism of Time Integration
in Perceptual Decisions,” *Journal of Neuroscience*, vol. 26, pp. 1314–1328,
2006. [Author-lab source](https://github.com/xjwanglab/wong-wang-2006/tree/c39c6742329f89f1b5137f32910d55ad52d4bc24).
