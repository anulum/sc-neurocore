<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SNN Studio user manual -->

# SNN Studio user manual

SNN Studio is the Visual SNN Design Studio in SC-NeuroCore. Use it to explore
spiking neurons, build small networks, run experiments, train selected network
architectures, inspect generated hardware and keep the evidence of your work.
This manual follows those tasks in the order you are likely to need them.
Names in **bold** are controls or sections in the interface.

Studio is a **development preview**. Available actions depend on the model,
source mode, installed dependencies and deployment policy. Read the reason
beside an unavailable action; selecting a model does not qualify every backend
or hardware path for it.

## Getting started

1. [Install and launch](installation.md), then check which deployment you opened.
2. [Run your first experiment](first-experiment.md). Save it and export its result.
3. [Choose experiments and analyses](experiments.md) for your research question.
4. [Build a network or train](networks-and-training.md) when one neuron is insufficient.
5. [Save, reproduce and share](projects-and-evidence.md) before changing environments.
6. [Follow the hardware workflow](hardware.md) when you need RTL or FPGA evidence.

If something fails, start with [Troubleshooting](troubleshooting.md).
The [glossary](glossary.md) explains the terminology used throughout the manual.

## Find the right part of the interface

| Area | Use it for | What to read first |
|---|---|---|
| Left sidebar: workbench and readiness | See the current project, experiment, run and missing prerequisites | [First experiment](first-experiment.md#read-the-workbench) |
| **Models / ODE** | Choose a catalogue neuron or enter equations | [Experiment inputs](experiments.md#choose-your-source) |
| **Neuron** views | Traces, firing statistics, sweeps and comparisons | [Analysis chooser](experiments.md#choose-an-analysis) |
| **Network** views | E-I demonstration, Synaptic delays and network canvas | [Networks and training](networks-and-training.md) |
| **Code and hardware** views | Code export, precision, compiler and synthesis | [Hardware](hardware.md) |
| **Research** views | Candidate models, fitting, training and revision review | [Research workflows](networks-and-training.md#candidate-models-and-parameter-fitting) |
| **Projects / Sessions** | Keep a server workspace or a browser snapshot | [Storage choices](projects-and-evidence.md#choose-what-to-save) |
| Toolbar and evidence cart | Download results, figures, replay packs and evidence | [Exports](projects-and-evidence.md#choose-an-export) |
| **Operator → Admin** | Identity, audit, retained jobs and deployment status | [Operator use](installation.md#operator-and-lab-use) |

Some tabs appear only when they can show a result. A phase plane needs at least
two state variables; an ISI histogram needs interval data. Characterisation is
for catalogue models. The precision and IR-inspector tabs are offered in ODE
mode. A hidden or disabled tab is not evidence that its analysis has run.

## Read workflow status correctly

The guided workflow follows **design → simulate → analyse → train (optional)
→ compile → co-simulate (where applicable) → synthesise → export**.
You can also work directly in a panel.

| Status | What it means for you |
|---|---|
| `done` | A successful result exists for the current inputs |
| `next` / `ready` | This is an actionable step; check its prerequisites |
| `blocked` | A prerequisite is missing; read the stated reason |
| `unsupported` | This deployment cannot perform the action |
| `failed` | The current attempt failed; inspect its message before retrying |
| `skipped` | You deliberately skipped optional training; no training result is implied |
| `not applicable` | The step does not apply to this source |

Changing an input or starting a rerun withdraws the affected completion state.
An older plot may remain visible for comparison. It does not establish success
for the current experiment. A successful download means the browser accepted
bytes; check your Downloads directory to confirm that you saved the file.

## What counts as evidence

Keep the question you tested, the exact inputs, the output and the environment
together. Simulation, a good fit, emitted RTL, bit-exact co-simulation, synthesis,
routing and a physical measurement answer different questions. The manual
explains each boundary where you encounter it; the [hardware guide](hardware.md)
puts them side by side.

A readiness label reports a declared or verified scope. It does not certify the
whole application or all catalogue models. For the detailed rules, read
[Model readiness](../model-readiness.md) and
[the hardware numeric contract](../numeric-contract.md).

## Detailed references

Use these pages when you need the full contract behind a manual procedure.

| Task | Detailed guide |
|---|---|
| Numerical inputs, raw state and metric validity | [Studio technical guide](../../guides/studio.md) |
| Model search, aliases and metadata health | [Catalogue health](../catalogue-health.md), [measured behaviour](../behaviour-facet.md) |
| Equations | [ODE editor](../ode-editor.md) |
| Graph editing, supported models, NIR and execution | [Network canvas](../network-canvas.md) |
| Learning and event recordings | [Training monitor](../training-monitor.md), [event datasets](../event-training.md) |
| Proposing and fitting a model | [Candidates](../candidate-models.md), [parameter fitting](../parameter-fitting.md) |
| Compiler and hardware arithmetic | [Compiler inspector](../compiler-inspector.md), [numeric contract](../numeric-contract.md) |
| Target tools, resources and routing | [Synthesis dashboard](../synthesis-dashboard.md) |
| Figures and reproducibility | [SVG](../svg-export.md), [replay and code](../replay-and-export.md) |
| Workspace history and peer review | [Workspaces](../workspaces.md), [collaboration](../collaboration.md) |
| Checking exported evidence | [Evidence verification](../evidence-verification.md) |
| Jobs, cancellation, restart and backup | [Job ledger](../job-ledger.md), [WebSocket progress](../websocket-progress.md) |
| Login and operation | [Admin identity and audit](../admin-identity.md) |
| API automation and host integration | [OpenAPI contract](../../api/studio_openapi.md), [federation](../../api/federation.md), [integration](../integration.md) |

Next: [Install and launch](installation.md).
