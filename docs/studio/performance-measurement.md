<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Measuring How Long the Studio Takes to Show a Result

The Studio puts a named mark on the browser's performance timeline each time
it draws a measured result. A browser test reads those marks and records how
long each result took to appear. This page says what is measured, how to run
the measurement, and how to read what it writes. It gives no number:
a number belongs with the artefact it came from.

## What is marked

The marks are always on. They change nothing on screen and carry only their
name and their time.

| Mark | Placed when |
|---|---|
| `studio:catalogue-drawn` | the model catalogue's rows are drawn for the first time |
| `studio:trace-drawn` | the plot draws the trace of a new simulation result (not when it redraws the same trace) |
| `studio:view-drawn` | a view chosen in the view switcher has been drawn |
| `studio:network-drawn` | the network view, as canvas or as table, has drawn the current populations and projections |
| `studio:training-event-received` | a training run's epoch arrives from its stream |
| `studio:training-chart-drawn` | the training charts have drawn a new epoch's row |

A long session keeps at most 1 000 marks of one name; older ones are cleared.

## What is measured

**How a sample is taken.** The test puts a start mark on the timeline just
before a user action. It then waits for the first mark after it. The sample is
the time between the two. Each measure takes ten samples; a measure with fewer
is refused.

| Measure | Workload | From | To |
|---|---|---|---|
| catalogue first draw | the shipped catalogue | page navigation | `catalogue-drawn` |
| simulation request to trace draw | AdExNeuron, 1 000 ms at dt 1, 0.1 and 0.01 ms, about 1 000, 10 000 and 100 000 steps | a change of the duration | `trace-drawn` |
| model switch to trace draw | AdExNeuron and Izhikevich2007Neuron | a click on the other model | `trace-drawn` |
| view switch to draw | AdExNeuron, Trace and Phase plane | a click on the other view | `view-drawn` |
| network canvas drag frame to redraw | networks of 2, 10 and 50 populations | one step of a node drag | `network-drawn` |
| network canvas and table switch to redraw | the same networks | a click on Table view | `network-drawn` |
| training epoch event to chart draw | 50 epochs on the synthetic dataset | `training-event-received` | the next `training-chart-drawn` |

**Two details of the workloads:**

- **Simulation durations.** The duration alternates between 990 and 1 000 ms, so every
  sample is a new request.
- **Training samples.** Each of the 50 epochs is a sample.

## How to run it

From `studio/frontend`, with the Studio backend's Python environment active.
That environment needs PyTorch for the training workload;
`requirements/ci-torch-cpu.txt` holds the CPU build the hosted run installs.

```bash
npm run test:e2e:timing
```

**What the command does.** It builds the bundle, serves it with `vite preview`
against a real Studio backend, and runs `e2e/render-timing-live.spec.ts` in
Chromium. A development server is not used, because it would time React's
development build.

**Where the artefact goes.** The run writes the artefact to the path in
`SC_NEUROCORE_RENDER_TIMING_OUT`, or to the test's output folder.

**Ports.** `SC_NEUROCORE_STUDIO_TIMING_API_PORT` and
`SC_NEUROCORE_STUDIO_TIMING_UI_PORT` move the two servers (defaults 18006 and
15180).

**On a hosted runner.** Start the main CI workflow by hand
(`workflow_dispatch`). Its frontend job then runs the measurement after its
browser suites and keeps the artefact as `studio-render-timing`. Ordinary
pushes do not run it.

## How to read the artefact

The artefact is JSON with the schema `studio.render-timing.v1`:

| Field | Meaning |
|---|---|
| `recordedAt` | when the run ended, in UTC |
| `environment.host`, `environment.cpuModel` | where it ran |
| `environment.loadBefore`, `environment.loadAfter` | the 1, 5 and 15 minute load averages before the first and after the last workload |
| `environment.browser` | the browser and its version |
| `environment.commit` | the commit of the measured bundle, followed by `with uncommitted changes` when a tracked file of the checkout differed from it |
| `environment.isolation` | `isolated-reserved-cores` when the run was given reserved cores through `SC_NEUROCORE_TIMING_RESERVED_CORES`, otherwise `functional-non-isolated` |
| `measures[]` | per measure and workload: every sample in milliseconds, in the order taken, their count, and p50 and p95 |

**How the percentiles are taken.** They use the nearest-rank method: the p-th
percentile of n sorted samples is the one at rank `ceil(p / 100 × n)`.

**What a `functional-non-isolated` run tells you.** It describes the code on
that host at that load, not an isolated benchmark. Compare it only with runs
of the same label, host and workload.

**Budgets.** No budget is checked by this measurement; budgets are proposed
from its artefacts and decided separately.
