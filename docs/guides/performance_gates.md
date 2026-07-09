<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Performance-gated pytest contract
-->

# Performance-Gated Tests

SC-NeuroCore keeps timing sanity checks out of the default pytest path. Tests
that assert a wall-clock threshold are marked with `SC_NEUROCORE_PERF=1` so
regular local and pull-request runs stay deterministic and cheap. The scheduled
`Performance Benchmarks` workflow enables the same variable and runs the
current perf-gated selector nightly.

Run the same selector locally when changing a perf-gated file:

```bash
SC_NEUROCORE_PERF=1 PYTHONPATH=src:. python -m pytest \
  tests/test_export/test_onnx_exporter.py \
  tests/test_learning/test_lifelong.py \
  tests/test_learning/test_federated.py \
  tests/test_transformers/test_block.py \
  tests/test_layers/test_recurrent.py \
  tests/test_layers/test_vectorized_layer.py \
  tests/test_layers/test_sc_conv_layer.py \
  tests/test_layers/test_sc_dense_layer.py \
  tests/test_layers/test_memristive.py \
  tests/test_layers/test_sc_learning_layer.py \
  tests/test_layers/test_fusion.py \
  tests/test_hdc/test_base.py \
  tests/test_accel/test_vector_ops.py \
  tests/test_solvers/test_ising.py \
  tests/test_optics/test_photonic_layer.py \
  tests/test_quantum/test_hybrid.py \
  tests/test_interfaces/test_dvs_input.py \
  tests/test_sources/test_bitstream_current_source.py \
  tests/test_hdl_gen/test_verilog_generator.py \
  tests/test_hdl_gen/test_spice_generator.py \
  tests/test_bio/test_dna_storage.py \
  tests/test_bio/test_grn.py \
  tests/test_graphs/test_gnn.py
```

## Current Selector

| Surface | Test file |
| --- | --- |
| ONNX export | `tests/test_export/test_onnx_exporter.py` |
| Lifelong learning | `tests/test_learning/test_lifelong.py` |
| Federated learning | `tests/test_learning/test_federated.py` |
| Transformer block | `tests/test_transformers/test_block.py` |
| Recurrent layer | `tests/test_layers/test_recurrent.py` |
| Vectorized layer | `tests/test_layers/test_vectorized_layer.py` |
| Convolution layer | `tests/test_layers/test_sc_conv_layer.py` |
| Dense layer | `tests/test_layers/test_sc_dense_layer.py` |
| Memristive layer | `tests/test_layers/test_memristive.py` |
| Learning layer | `tests/test_layers/test_sc_learning_layer.py` |
| Fusion layer | `tests/test_layers/test_fusion.py` |
| HDC encoder | `tests/test_hdc/test_base.py` |
| Vector operations | `tests/test_accel/test_vector_ops.py` |
| Ising solver | `tests/test_solvers/test_ising.py` |
| Photonic layer | `tests/test_optics/test_photonic_layer.py` |
| Quantum hybrid layer | `tests/test_quantum/test_hybrid.py` |
| DVS input layer | `tests/test_interfaces/test_dvs_input.py` |
| Bitstream current source | `tests/test_sources/test_bitstream_current_source.py` |
| Verilog generator | `tests/test_hdl_gen/test_verilog_generator.py` |
| SPICE generator | `tests/test_hdl_gen/test_spice_generator.py` |
| DNA storage | `tests/test_bio/test_dna_storage.py` |
| Gene-regulatory network | `tests/test_bio/test_grn.py` |
| Graph neural layer | `tests/test_graphs/test_gnn.py` |

The selector is intentionally pytest-based. It is regression evidence for
small timing thresholds inside existing behaviour tests, not a replacement for
isolated benchmark artefacts under `benchmarks/results/`. Public benchmark
claims still require the benchmark-evidence rules described in
[Benchmarks](../benchmarks/BENCHMARKS.md).
