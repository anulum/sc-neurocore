<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Vivado CI gate contract
-->

# Vivado CI Gates

The default local and GitHub-hosted test lanes do not require Vivado. Vivado
tests are opt-in because they need a vendor installation, target-device support,
and enough disk and memory for synthesis or implementation runs.

## Runner contract

Set `MIF_VIVADO_CI=1` only on a runner that provides:

- Vivado 2024.2 on `PATH` as `vivado`;
- Xilinx Zynq UltraScale+ device support for `xczu3eg-sbva484-1-e`;
- enough scratch space for Vivado project directories and report output;
- no public hardware-measurement claim unless a matching board, pin map, timing
  report, utilisation report, and power report are committed as evidence.

Without `MIF_VIVADO_CI=1`, these tests skip intentionally. With the variable set
but without `vivado` on `PATH`, they also skip rather than pretending the runner
validated synthesis.

## Test selector

```bash
MIF_VIVADO_CI=1 PYTHONPATH=src:. python -m pytest \
  tests/test_ultrascale_plus_flow.py \
  tests/test_adc_to_spike_quantiser_synth.py \
  tests/test_dcls_synth_zu3eg.py \
  -q
```

| Test file | Gate | What it proves |
| --- | --- | --- |
| `tests/test_ultrascale_plus_flow.py` | `NEU-C.1` | Generates a ZU3EG Vivado Tcl project from a manifest and runs the UltraScale+ batch flow. |
| `tests/test_adc_to_spike_quantiser_synth.py` | `NEU-C.5` | Elaborates the ADC-to-spike quantiser RTL through Vivado. |
| `tests/test_dcls_synth_zu3eg.py` | `NEU-C.6` | Synthesises the DCLS layer core out of context for the ZU3EG target. |

These tests are synthesis-flow gates. They do not by themselves establish board
timing closure, routed power, or hardware energy numbers. Publish those claims
only with committed Vivado reports and the exact runner, device, constraint, and
workload context.
