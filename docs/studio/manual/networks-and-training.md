<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SNN Studio user manual -->

# Networks, training and research workflows

## Build a network

The **Network canvas** is a graph of populations and projections. A population
contains neurons of one model; a projection connects a source population to a
target. It is separate from the single-neuron experiment in the sidebar.

1. Open **Network canvas** in the **Network** view group.
2. Add **+ Exc** and **+ Inh** populations.
3. Select each population to inspect its model, count, parameters and external
   drive. Start with small counts. A population with no drive or incoming
   activity can remain silent.
4. Connect a source handle to a target. Select the projection and set weight,
   connection rule, probability where applicable, delay, seed and autapses.
5. Set graph duration, timestep and seed, then **Simulate**.
6. Read validation messages by field before interpreting population rates or
   spike counts. Save the graph as part of a project when it is useful.

An excitatory source requires the corresponding positive weight sign; an
inhibitory source requires the negative sign. Delays must be a whole number of
graph timesteps. A `0.05 ms` delay with `dt = 0.1 ms` is refused, not rounded.
A spike at step `t` reaches its targets at `t + 1 + delay_steps` in the public
reference loop. Account for this latency when interpreting network activity.

Only graph-admissible models appear in its model picker. A neuron available for
single-model simulation is not automatically a supported population model.
The graph uses its declared timestep, drives, constructor inputs, connectivity
and seeds; moving a node changes layout, not the simulated specification.

## Edit without losing the graph

- Select and edit populations/projections through their property panels.
- Use **Undo** / **Redo**, or **Ctrl+Z** / **Ctrl+Shift+Z**, outside text fields.
  The edit history is session-local; saving the graph does not save undo history.
- Deleting a population also deletes its incoming and outgoing projections.
  Deleting an edge leaves both populations.
- **Duplicate** copies selected populations and projections whose two endpoints
  are inside the selection. Crossing projections are reported rather than
  silently redirected.
- Use the graph's **table view** for keyboard and screen-reader access to
  adding, connecting, editing and deleting the same objects.

The [network reference](../network-canvas.md) describes exact fields, budgets,
NIR transfer, reference execution and API refusals. An imported NIR graph can
contain operations this canvas does not admit; read the import result before
assuming framework equivalence.

## E-I and delay views

The **E-I network** panel is a parameterised excitatory/inhibitory experiment.
Inspect population sizes, drive and weights before reading its raster and
rates. Its optional compiled backend is distinct from the canvas reference
runner; availability in one panel does not establish parity in another.

The **Synaptic delays** view is a separate learning experiment. Keep its own
configuration, seed and result when comparing runs. A learning demo does not
add arbitrary plasticity or per-synapse delay arrays to the network canvas.
See the [delay-training tutorial](../../tutorials/39_learnable_delays.md) for the
underlying method.

## Train a selected architecture

The **Training** panel trains the architecture configured in that panel. It
does not automatically train the graph you drew or replace a selected catalogue
neuron with the trained network.

1. Open **Training** in the **Research** group.
2. Start with the **Synthetic** dataset for a small functional exercise.
3. Select architecture/hidden widths, surrogate, epochs, batch size, learning
   rate and timesteps. Keep the displayed settings with your result.
4. Choose **Train** and observe the recorded configuration and job ID.
5. Read loss, training/validation accuracy and per-layer activity together.
6. Use **Stop** if needed. Wait for the returned terminal state; cancellation
   is cooperative and an active tensor operation can finish first.
7. Export the checkpoint when available and save the workspace configuration.

Training requires the optional PyTorch backend. MNIST additionally needs its
data path/download arrangement. A synthetic accuracy result is a demonstration
of that task; it is not an MNIST or event-dataset benchmark.

A retained run's configuration is shown separately from editable project
settings. **Refresh runs** reconnects status and metrics after interruption.
An `unknown` run has uncertain liveness; refresh before starting or stopping
another run under that selection. Reattachment does not qualify new training
for an experiment edited since the run was submitted.

### Checkpoints and continuation

A portable checkpoint can restore recorded configuration. Full weight restore
also depends on the declared weight artifacts, digests and architecture.
Distinguish:

| Action | What continues |
|---|---|
| Load checkpoint configuration | Settings for a later request |
| Warm start | Compatible trained weights in a new training run |
| Exact resume | The qualified state needed to continue the interrupted trajectory |
| Live attach | A separately admitted change to a running worker |

Do not call a configuration import an exact weight/optimiser resume. Follow
[Training Monitor](../training-monitor.md#portable-checkpoints) for artifact
requirements, replayable seeds, warm start, exact resume and attach semantics.
A completed job does not by itself meet an accuracy criterion; use the recorded
validation result and any criterion declared before the run.

### Local event datasets

N-MNIST, SHD and DVS-CIFAR10 use operator-provided local recordings, an explicit
manifest, group split and event encoder. Starting such a run does not download
the data. Preserve file digests, labels, groups, encoder and split with the
run; keep acquisition groups out of both training and held-out evaluation
where the protocol requires independence. Follow
[Event dataset training](../event-training.md), including worker input budgets.
A verified manifest identifies bytes; it does not establish training accuracy
or hardware power and latency.

## Candidate models and parameter fitting

Use **Candidate model** to propose a model without changing the catalogue:

1. Import a candidate package or start from an available canonical schema.
2. State the source, assumptions, authors, units and parent when applicable.
3. Validate and correct every located diagnostic.
4. Diff against the parent and inspect both symbolic and textual differences.
5. Run the proposed reference tests and export the candidate and review packet.

A candidate remains a proposal. Its author's tests do not replace independent
source validation or admit it to the catalogue. The draft is kept with the
workspace, including an unfinished draft. See [Candidates](../candidate-models.md).

Use **Fitting** to estimate parameters of a catalogue schema or candidate:

1. Select the target and observable.
2. Declare parameter names, bounds, scales and fixed values using schema names.
3. Import current/observed-value recordings and assign training/hold-out roles.
4. Run the bounded fit with its seed and optimisation settings.
5. Read hold-out errors, uncertainty and unconstrained parameter combinations.
6. Export the result and replay information before applying a fitted value.

A small fit error does not establish unique parameters, independent source
fidelity or hardware validity. For cohorts, acquisition groups, constraints and
measurement custody, read [Parameter fitting](../parameter-fitting.md).

Next: [Projects, recovery and evidence](projects-and-evidence.md).
