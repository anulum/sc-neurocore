<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Optimiser API reference
-->

# Optimiser API

The `sc_neurocore.optimizer` package exposes two optimisation surfaces:

- `fit_to_target(...)` for deterministic resource fitting against FPGA target
  budgets.
- `SurrogateSCOptimizer` for ML-guided stochastic-computing compiler choices
  using generated analytical design points plus optional measured observations.

It also exposes strict evidence loaders so benchmark and synthesis reports can
feed the surrogate without fabricating missing LUT, power, latency, or accuracy
values.

## Deterministic Resource Fitting

Use `fit_to_target` when you have layer dimensions and weights and want the
legacy prune / quantise / bitstream-length loop to reduce estimated FPGA
resource use:

```python
import numpy as np

from sc_neurocore.optimizer import fit_to_target

weights = [np.ones((32, 16), dtype=np.float32)]
result = fit_to_target(
    layer_sizes=[(32, 16)],
    weights=weights,
    target="ice40",
    initial_bitstream_length=256,
)

print(result.summary())
```

The result reports whether the design fits, final estimated LUT use, selected
bitstream length, sparsity, and each optimisation step.

## Surrogate-Guided SC Optimisation

Use `SurrogateSCOptimizer` when the compiler must choose per-layer stochastic
computing settings under LUT, power, and latency pressure:

```python
from sc_neurocore.optimizer import SurrogateSCOptimizer, TargetHardwareProfile
from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile

target = TargetHardwareProfile(
    name="ice40-low-power",
    budget=HardwareBudget(max_luts=7680, max_power_mw=250.0, max_latency_cycles=4096),
)
network = [
    LayerProfile(id="encoder", mac_count=256, is_critical_path=True),
    LayerProfile(id="decoder", mac_count=128),
]

report = SurrogateSCOptimizer(target).optimise(network)
```

The report contains selected bitstream length, decorrelator, precision, LFSR
polynomial, estimated LUTs, power, latency, utility score, and any rejected
layers.

## Measured Evidence

Measured observations are optional but preferred. The loader accepts JSON
payloads with `observations`, `benchmark_observations`, `layers`, `runs`, or
`results` records and raises `ObservationLoadError` when required fields are
missing.

```python
from sc_neurocore.optimizer import load_observations

observations = load_observations("benchmarks/results/fpga_power_observations.json")
```

For raw Vivado or Quartus reports, use the package helper:

```python
from sc_neurocore.optimizer import build_payload_from_reports, write_payload

payload = build_payload_from_reports(
    design_path="build/network_design.json",
    utilisation_path="build/vivado_utilisation.rpt",
    power_path="build/vivado_power.rpt",
    timing_path="build/vivado_timing.rpt",
    accuracy_score=0.991,
    clock_mhz=100.0,
    inferences_per_run=1,
)
write_payload(payload, "build/synthesis_observations.json")
```

The same flow is available from the CLI:

```bash
sc-neurocore collect-synthesis \
  --design build/network_design.json \
  --utilisation build/vivado_utilisation.rpt \
  --power build/vivado_power.rpt \
  --timing build/vivado_timing.rpt \
  --accuracy-score 0.991 \
  --clock-mhz 100 \
  --inferences-per-run 1 \
  --out build/synthesis_observations.json
```

Energy fields are computed only when both `clock_mhz` and
`inferences_per_run` are provided. Vendor reports remain the evidence source;
the helper does not run synthesis or invent measurements.

::: sc_neurocore.optimizer
    options:
      show_root_heading: true
