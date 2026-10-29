<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SNN Studio user manual -->

# Experiments and analysis

## Choose your source

**Models** runs a catalogue class with its declared constructor inputs and
observable state. Read its citation, numerical profile, readiness and capability
messages. A declared readiness tier describes the descriptor; a verified tier
requires matching evidence. An installed build can show evidence sealed at
build time. [Model readiness](../model-readiness.md) explains the difference.

Search and filter the catalogue by model identity, category, behaviour and
readiness as offered by the build. Behaviour labels describe a measured protocol,
not every possible firing regime. Check aliases and metadata warnings in
[Catalogue health](../catalogue-health.md).

**ODE** runs equations you enter. Begin with a built-in template and change one
term at a time. The editor accepts one derivative per line, plus threshold and
reset directives, for example:

```text
dv/dt = -v / tau + I
# threshold: v > 1
# reset: v = 0
```

Supply `tau` in the parameter controls and `v` in the initial-state controls.
Keep `tau` positive and choose a suitable timestep. The editor's colouring is
not equation validation; simulation admission checks names, syntax and allowed
operations. Omitted equation variables start at zero. An additional initial
value declares a held state variable: it keeps its value between steps unless
an authored reset changes it. The result records both equation and held states.
[ODE editor](../ode-editor.md) describes the syntax.

Changing to ODE mode does not make every catalogue analysis available.
Simulation, precision, compilation and hardware each have their own supported
operations. The [hardware chapter](hardware.md) describes the compiler boundary.

## Set the numerical experiment

Record the selected source and revision, parameters, initial state, method,
timestep `dt`, duration, drive protocol and randomness. The server resolves
these into the effective experiment returned with the result. Use that resolved
record when a UI default and a model default might differ.

- `dt` is the integration step. A smaller step increases work and can change
  numerical behaviour; compare convergence rather than assuming it is exact.
- Duration determines the observed interval. A short run can miss adaptation,
  bursting, slow transients or rare stochastic events.
- Initial state affects the trajectory. Resetting the UI to defaults is not a
  replay of an arbitrary previously exported state.
- Units are model-defined where the contract does not declare physical units.
  Inspect them before combining traces from different models.

Large synchronous requests are refused or directed to a job workflow rather
than silently shortened. Read the returned limit and reduce the requested
work deliberately. An element budget can also prevent complete raw export
without making a bounded display plot disappear.

## Choose a drive

| Protocol | How the current is applied | Useful first question |
|---|---|---|
| Constant | Fixed amplitude throughout | Does this neuron spike under steady drive? |
| Step | Active from 20% to 80% of the run | How does onset and recovery behave? |
| Ramp | Increases from zero to the selected amplitude | Where does firing begin under this ramp? |
| Pulse train | Five bounded pulses over the run | Does each pulse elicit a response? |
| Sine | Sinusoidal modulation at the chosen frequency in Hz | How does the response vary with drive frequency? |

A rheobase inferred from one protocol and duration is not a protocol-independent
constant. Frequency-response analysis also has a bias/modulation contract;
inspect its settings and metric validity.

## Control randomness

For stochastic models or noisy equations, distinguish **replay** from **fresh**
trials. A replay uses the effective seed. A fresh trial draws a new seed and
reports it in the result. Export the original JSON before another run.
The **Replay pack** action executes an export request from the current controls;
with **fresh** still selected it can draw another trial. To preserve a displayed
trial, take its effective seed from the JSON result, set that exact seed and
**replay**, then export the pack and compare its recorded expectation with the
original. Saving a project set to fresh does not repeat the previous draw.

A seed on a deterministic model is refused. Model-specific generators can have
smaller seed ranges than the UI's exact-integer representation. Graph seeds
have a separate limit; do not reuse a large single-neuron seed blindly.
Compare several seeds when making a statistical claim. One repeated trajectory
demonstrates replay of that run, not a distributional result.

## Choose an analysis

| View | Use it to ask | Check before interpreting |
|---|---|---|
| **Trace** | What happened over time? | Current run status, units and raw-data availability |
| **Phase plane** | How do two state variables evolve together? | At least two variables; nullcline domain and current experiment |
| **ISI** | How are inter-spike intervals distributed? | Enough spikes and the observation interval |
| **f-I curve** | How does firing rate vary with current? | Sweep range, duration, transient and rate definition |
| **Bifurcation** | How does the observed attractor change with a parameter? | Parameter, sweep range, finite observation; this is not a mathematical proof |
| **2-D sweep** | How does rate vary over two parameters? | Both axes, units, grid resolution and current result identity |
| **Sensitivity** | Which parameters affect the chosen metric locally? | Perturbation, baseline and metric validity |
| **STA** | What waveform accompanies spikes? | Enough events, window and full-resolution samples |
| **Frequency response** | How does firing respond to modulated drive? | Bias, frequencies, amplitude, gain/phase contract |
| **Characterization** | What does a bundled set of measurements report? | Catalogue-model support and each component's validity |
| **Multi-model** | How do selected models compare under the submitted inputs? | Per-model units, timestep, inputs and failures |
| **A/B compare** | What differs between two configurations? | Which inputs actually differ and which result is current |
| **Q8.8 precision** | How does the selected fixed-point kernel differ from float? | Supported equations, selected format, rounding, overflow and compiler tools |

Use **Run async analysis** for offered job-backed sweeps. Watch the job's status;
changing tabs does not change what it submitted. Changing the experiment makes
an old completion historical. A failed, cancelled or disconnected operation
has not completed merely because an old plot remains visible.

A heatmap cell can apply its two values together only while that heatmap
belongs to the current experiment. A historical heatmap remains a reference,
not an authority to alter a different experiment.

## Read validity, state and error reports

An analysis can return an undefined or invalid metric with a reason. Preserve
that reason. Do not turn an undefined rate, singular fit or insufficient spike
count into zero in a paper or comparison.

Simulation metadata distinguishes initial state at time zero from post-step
samples. Raw sample `i` is after step `i`, at `(i + 1) * dt`; the corresponding
drive acts over `[i * dt, (i + 1) * dt)`. Display decimation does not change the
raw spike indices or the underlying samples.

Inspect `state_layout.complete`, `incomplete_reasons`, `undeclared_mutable` and
raw-data omission fields in the JSON export. A complete-looking voltage plot
can coexist with unrecorded state. Keep these limitations with any result you
share. The [technical guide](../../guides/studio.md#complete-state-and-raw-result-custody)
defines the full response contract.

Next: [Networks and training](networks-and-training.md), or
[save and export your experiment](projects-and-evidence.md).
