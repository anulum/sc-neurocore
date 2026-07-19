<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — DM-03 MNIST → quantise → SC → Verilog path pointer
-->

# DM-03 — MNIST → Q8.8 → SC → Verilog (path pointer)

This is **documentation only**. It does not retrain inventively or invent
synthesis numbers. The runnable demo already lives in-tree.

## Honesty box

| | |
|---|---|
| **Proves** | How to invoke the existing end-to-end demo and where outputs land. |
| **Does not prove** | Production MNIST accuracy on silicon, Vivado timing closure, or power. |
| **Artefacts** | Whatever `examples/mnist_fpga/demo.py` writes under your run directory / `hdl/generated/`. |

## One-command path

From the SC-NeuroCore repository root (with package or `PYTHONPATH=src`):

```bash
# sklearn digits (8×8), no external MNIST download
python examples/mnist_fpga/demo.py

# optional Verilog weight export
python examples/mnist_fpga/demo.py --export-verilog hdl/generated/mnist_weights.vh
```

Requires: `numpy`, `scikit-learn` (see `examples/README.md`).

## Related notebooks

| Notebook | Role |
|----------|------|
| `13_quantisation_pipeline.ipynb` | QAT / quant pedagogy |
| `08_equation_to_verilog.ipynb` | ODE → Verilog for **catalogue** neurons |
| `27_python_to_proven_silicon.ipynb` | Broader silicon path |
| `29_golden_path_evidence.ipynb` | Evidence-bound golden path |

## High-fidelity neurons

MNIST demo is a **classifier stack**, not a polyglot-complete biophysical neuron.
For high-fidelity neuron demos use NB-41 (`HodgkinHuxleyNeuron`,
`MorrisLecarNeuron`, `AdExNeuron`) and
`docs/api/model_fidelity_status.md`.

## Next hop after export

1. Inspect generated `.vh` / RTL under `hdl/generated/` (if produced).
2. Read committed reports only via `examples/dm04_synthesis_report_reader.py`.
3. Do not promote a local run to package benchmarks without committing JSON under
   `benchmarks/results/` or reports under `hdl/reports/`.
