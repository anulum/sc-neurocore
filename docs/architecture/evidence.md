<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Evidence: what each kind establishes -->

# Evidence: what each kind establishes

SC-NeuroCore produces several kinds of evidence: records of what ran,
comparisons between implementations, formal results, reports of design
tools and benchmark artefacts. They are not interchangeable. This page says
what each kind shows, what it does not show, and where to find it. It is the
companion of the [architecture overview](architecture.md).

## Two different facts

**Custody** means that a result is the one that was written, unaltered, and
consistent with the inputs it names. **Correctness** means that the result
is right. A record can establish custody. It cannot establish correctness;
that takes a comparison with something independent.

The documentation keeps the two apart. A receipt on a piece of Studio
evidence shows that a pack is internally consistent and unaltered since
export. It does not show who produced it, and it does not show that the
model is right ([checking an evidence pack](../studio/evidence-verification.md)).

## Kinds of evidence

| Kind | What it shows | What it does not show | Where |
| --- | --- | --- | --- |
| A record of a run (receipt, seal, digest) | Which inputs, model and settings produced these bytes, and that the bytes are unaltered | That the result is right | [Checking an evidence pack](../studio/evidence-verification.md) |
| Agreement between implementations | Two or more implementations of one model give the same result on the inputs that were run | Agreement with the source model, with generated hardware or with a device | [Per-model evidence table](../api/model_fidelity_status.md) |
| Co-simulation | Generated or hand-written hardware and its fixed-point reference agree on the stimuli that were run | Agreement on other stimuli; the scientific fidelity of the model | [Model readiness](../studio/model-readiness.md) |
| A formal result | A stated property holds under its stated assumptions and bounds | Anything outside those assumptions; a count of proof jobs is not a count of passing results | [External formal evidence](../formal/external_results.md) |
| A report of a design tool | What the tool reports for one design and one target: resources, timing estimates | Behaviour of a real device; power or energy, unless a matching report is committed | [Hardware guide](../hardware/HARDWARE_GUIDE.md) |
| A benchmark artefact | A measured figure with its host, date and committed record | The same figure on another machine or under another load | [Benchmark report](../benchmarks/BENCHMARKS.md), [evidence boundary](../benchmarks/cross_framework.md) |
| A measurement on a device | What an identified instrument observed on real hardware, with its uncertainty | Behaviour under other conditions | None yet: no board run has validated any RTL of this repository |

A design, a simulation, a synthesis report and a measurement on a real
device establish different facts, and none of them stands in for another.

## Rules the documentation follows

- **A number names its record.** A measured figure appears with the
  committed artefact it comes from. A figure without one is not evidence.
- **Missing is shown as missing.** A check that did not run is reported as
  not run, not as passed. A payload from a build that wrote no receipt reads
  as `unsealed`: a boundary the report names, not a check that passed.
- **A comparison names its two sides.** The hand-written RTL of the
  reference neuron is compared with the fixed-point Python model. The Studio
  compares generated RTL with the generated bit-true C kernel. These are
  different comparisons, and neither is a statement about a model's
  scientific fidelity.
- **A record says what it attests.** A Studio receipt is marked `produced`
  when the surface that ran the work wrote it, and `exported` when it was
  written while a pack was assembled from supplied payloads. The two are
  different claims.
- **Readiness is not approval.** Evidence categories and readiness
  assessments support a human decision. They are not certification.

## What is planned

The wider programme plans a family of measurement and evidence tools, so
that a claim about timing, data quality, numerical agreement or deployment
can be traced to its source. Their status is on the
[group and project map](../ecosystem.md). None is announced as available.
