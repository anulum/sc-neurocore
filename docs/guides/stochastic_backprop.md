<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Stochastic backpropagation guide -->

# Stochastic Backpropagation

SC-NeuroCore includes an experimental reference path for training through
stochastic-computing design variables, not only through a floating-point model
that is converted to bitstreams afterwards. The current workflow optimises
model weights together with bitstream length, encoding choice, and correlation
metadata, then writes deterministic benchmark, SC-NIR export, estimator
regression, and executable HDL parity artefacts.

The evidence boundary is `local_simulation_and_executable_hdl_parity`.

## What The Evidence Means

The current workflow can report:

- loss reduction for a deterministic stochastic-backprop reference task;
- trained bitstream-length, encoding, and correlation selections;
- SC-NIR stream metadata and correlation constraints used by the handoff;
- estimator-family variance evidence for `pathwise_relaxation`,
  `straight_through`, and `score_function`;
- executable local HDL parity for the trained design metadata via
  `iverilog`/`vvp`.

The current workflow does not provide:

- no physical hardware measurement;
- no Vivado timing closure claim;
- no PYNQ deployment claim;
- no board-level power, thermal, or timing evidence.

Treat these artefacts as local software and executable-HDL parity evidence.
Vivado implementation reports, PYNQ runs, lab measurements, and physical
deployment packets are separate evidence classes.

## Python API

Use `stochastic_backprop_joint_objective` when the training step must expose
gradients through both model parameters and stochastic-computing design
variables:

```python
import torch

from sc_neurocore.training.sc_estimators import DifferentiableSCConfig
from sc_neurocore.training.stochastic_backprop import (
    SCBackpropDesignSpace,
    SCTrainingObjectiveConfig,
    stochastic_backprop_joint_objective,
)

inputs = torch.tensor([[0.2, -0.4], [0.8, 0.1]], dtype=torch.float32)
targets = torch.tensor([[0.3], [0.6]], dtype=torch.float32)
weight = torch.tensor([[0.25, -0.35]], dtype=torch.float32, requires_grad=True)
bias = torch.tensor([0.05], dtype=torch.float32, requires_grad=True)
length_logits = torch.tensor([0.6, -0.2, 0.1], dtype=torch.float32, requires_grad=True)
encoding_logits = torch.tensor([0.8, -0.3], dtype=torch.float32, requires_grad=True)
correlation_logit = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)

report = stochastic_backprop_joint_objective(
    inputs,
    targets,
    weight=weight,
    bias=bias,
    length_logits=length_logits,
    encoding_logits=encoding_logits,
    correlation_logit=correlation_logit,
    base_config=DifferentiableSCConfig(
        bitstream_length=128,
        encoding="bipolar",
        generator="sobol",
        estimator="pathwise_relaxation",
        input_seed=3,
        weight_seed=11,
        correlation=0.0,
    ),
    design_space=SCBackpropDesignSpace(
        bitstream_lengths=(64, 128, 256),
        encodings=("bipolar", "unipolar"),
        min_correlation=-0.25,
        max_correlation=0.25,
    ),
    objective_config=SCTrainingObjectiveConfig(
        length_cost_weight=1.5,
        correlation_cost_weight=0.4,
        encoding_cost_weight=0.3,
    ),
)

report.breakdown.total.backward()
```

The returned report contains the relaxed prediction, objective breakdown,
selected exportable `DifferentiableSCConfig`, length probabilities, encoding
probabilities, expected bitstream length, selected encoding, selected length,
and selected correlation.

## Reproducible Artifact Generation

Generate the benchmark report, SC-NIR export manifest, estimator-regression
manifest, and HDL handoff directory with one command:

```bash
PYTHONPATH=src python tools/stochastic_backprop_benchmark.py \
  --output benchmarks/results/stochastic_backprop_benchmark.json \
  --export-manifest benchmarks/results/stochastic_backprop_export_manifest.json \
  --estimator-regression-manifest benchmarks/results/stochastic_backprop_estimator_regression_manifest.json \
  --handoff-dir benchmarks/results/stochastic_backprop_handoff \
  --bitstream-length 256 \
  --steps 32 \
  --learning-rate 0.4
```

The generated evidence set is:

- `benchmarks/results/stochastic_backprop_benchmark.json`;
- `benchmarks/results/stochastic_backprop_export_manifest.json`;
- `benchmarks/results/stochastic_backprop_estimator_regression_manifest.json`;
- `benchmarks/results/stochastic_backprop_handoff/scnir_document.json`;
- `benchmarks/results/stochastic_backprop_handoff/scnir_source_manifest.json`;
- `benchmarks/results/stochastic_backprop_handoff/stochastic_backprop_trained_design.v`;
- `benchmarks/results/stochastic_backprop_handoff/stochastic_backprop_trained_design_parity.json`;
- `benchmarks/results/stochastic_backprop_handoff/stochastic_backprop_handoff_audit.json`.

## Estimator Regression

The standalone estimator-regression manifest compares the seeded variance of
three estimator assumptions over multiple bitstream lengths:

- `pathwise_relaxation`: differentiable relaxed Bernoulli expectation;
- `straight_through`: sampled forward pass with a relaxed backward proxy;
- `score_function`: seeded likelihood-ratio estimate without a baseline.

The manifest acceptance fields check that:

- all variances are finite and nonnegative;
- pathwise variance is zero for the deterministic relaxation;
- the longest configured score-function variance is below the shortest
  configured score-function variance.

This is a regression signal for estimator stability. It is not a statistical
certification of convergence for every network or dataset.

## HDL Parity

The handoff bundle emits `stochastic_backprop_trained_design.v`, a small
executable Verilog module that exposes the trained stochastic-design metadata as
constant outputs:

- selected bitstream length;
- expected bitstream length in Q16;
- trained correlation in Q16;
- selected encoding flag.

The test suite compiles this module with `iverilog` and executes it with `vvp`
against `stochastic_backprop_trained_design_parity.json`. This proves that the
trained design metadata survives into executable HDL. It does not prove FPGA
place-and-route, timing closure, or board deployment.

## Review Checklist

Before using stochastic-backprop evidence in a release packet, check that:

- `hardware_measurement_claimed` is `false` in every benchmark/export artifact;
- `joint_design.enabled` is `true` in the benchmark and export manifest;
- `selected_bitstream_length`, `selected_encoding`, and `correlation` match
  between the benchmark and export manifest;
- estimator-regression `status` is `pass`;
- `stochastic_backprop_handoff_audit.json` reports `status: valid`;
- the trained-design HDL parity test has been run on the checked-out artifact;
- no physical hardware claim is made unless separate raw hardware evidence is
  attached.
