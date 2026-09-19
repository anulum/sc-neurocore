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
regular local and pull-request runs stay deterministic and cheap. The
`Performance Benchmarks` workflow runs only by manual dispatch with a required
run reason and change reference. It runs the same perf-gated selector alongside
the Rust and Python benchmark suites. Run it for a new catalogue model after
fidelity gates, a material numerical or compute-path change, a confirmed
regression, a planned stable baseline, or an explicitly scoped comparison. Documentation changes,
an unrelated source hash, and routine CI repetition do not justify a new
timing record.

The Rust Criterion comparison still posts an alert for a large timing change,
but that alert is advisory because runner hardware and host state affect the
measurement. Build, execution, and artifact-upload errors still fail the
workflow. The dispatch reference and source SHA are retained with the raw
benchmark output; historical timing records are not overwritten to match new
code. The planned shared benchmark utility will add sealed plans, before/after
hardware-state capture, immutable result lineage, and the same trigger
decision for Catalogue and Studio before SC-NEUROCORE STABLE.

Run the same selector locally when changing a perf-gated file:

```bash
SC_NEUROCORE_PERF=1 PYTHONPATH=src:. python -m pytest \
  tests/test_export/test_onnx_exporter_performance.py \
  tests/test_learning/test_lifelong.py \
  tests/test_learning/test_federated.py \
  tests/test_transformers/test_block.py \
  tests/test_layers/test_recurrent.py \
  tests/test_layers/test_vectorized_layer_packed_forward.py \
  tests/layers/test_sc_conv_layer_configuration.py \
  tests/test_layers/test_sc_dense_layer_performance.py \
  tests/test_layers/test_memristive.py \
  tests/test_layers/test_sc_learning_layer.py \
  tests/layers/test_fusion_performance.py \
  tests/test_hdc/test_base.py \
  tests/accel/test_vector_ops_validation_performance.py \
  tests/test_solvers/test_ising.py \
  tests/test_optics/test_photonic_layer.py \
  tests/test_quantum/test_hybrid.py \
  tests/interfaces/test_dvs_input_performance.py \
  tests/sources/test_bitstream_current_source_performance.py \
  tests/hdl_gen/test_verilog_generator_io_perf.py \
  tests/test_hdl_gen/test_spice_generator.py \
  tests/test_bio/test_dna_storage.py \
  tests/test_bio/test_grn.py \
  tests/test_graphs/test_gnn.py
```

## Current Selector

| Surface | Test file |
| --- | --- |
| ONNX export | `tests/test_export/test_onnx_exporter_performance.py` |
| Lifelong learning | `tests/test_learning/test_lifelong.py` |
| Federated learning | `tests/test_learning/test_federated.py` |
| Transformer block | `tests/test_transformers/test_block.py` |
| Recurrent layer | `tests/test_layers/test_recurrent.py` |
| Vectorized layer | `tests/test_layers/test_vectorized_layer_packed_forward.py` |
| Convolution layer | `tests/layers/test_sc_conv_layer_configuration.py` |
| Dense layer | `tests/test_layers/test_sc_dense_layer_performance.py` |
| Memristive layer | `tests/test_layers/test_memristive.py` |
| Learning layer | `tests/test_layers/test_sc_learning_layer.py` |
| Fusion layer | `tests/layers/test_fusion_performance.py` |
| HDC encoder | `tests/test_hdc/test_base.py` |
| Vector operations | `tests/accel/test_vector_ops_validation_performance.py` |
| Ising solver | `tests/test_solvers/test_ising.py` |
| Photonic layer | `tests/test_optics/test_photonic_layer.py` |
| Quantum hybrid layer | `tests/test_quantum/test_hybrid.py` |
| DVS input layer | `tests/interfaces/test_dvs_input_performance.py` |
| Bitstream current source | `tests/sources/test_bitstream_current_source_performance.py` |
| Verilog generator | `tests/hdl_gen/test_verilog_generator_io_perf.py` |
| SPICE generator | `tests/test_hdl_gen/test_spice_generator.py` |
| DNA storage | `tests/test_bio/test_dna_storage.py` |
| Gene-regulatory network | `tests/test_bio/test_grn.py` |
| Graph neural layer | `tests/test_graphs/test_gnn.py` |

The selector is intentionally pytest-based. It is regression evidence for
small timing thresholds inside existing behaviour tests, not a replacement for
isolated benchmark artefacts under `benchmarks/results/`. Public benchmark
claims still require the benchmark-evidence rules described in
[Benchmarks](../benchmarks/BENCHMARKS.md).
