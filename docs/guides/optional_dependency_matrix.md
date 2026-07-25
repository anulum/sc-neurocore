<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Optional dependency matrix
-->

# Optional Dependency Matrix

The base package must stay usable without heavyweight accelerator, vendor, or
interop stacks. Optional dependencies are grouped by the workflow that needs
them. This matrix records the declared install profile, the import checked by
tests, and the current CI/dev boundary.

| Dependency | Import surface | Install profile | CI/dev boundary | Evidence |
| --- | --- | --- | --- | --- |
| `gdsfactory>=9.0` | `gdsfactory` | `pip install "sc-neurocore[optics]"` | Dedicated CI job `test-optics-extra` installs `.[dev,optics]` and runs `tests/test_optics -q -rs` so GDSII round-trip tests execute with `gdsfactory`; default CPU jobs still skip GDSII tests when the extra is absent. | `tests/test_optics/test_gdsii.py`, `tests/test_optics/test_photonic_emitter_branches.py`, `.github/workflows/ci.yml` |
| `dwave-neal` + `dimod` | `neal`, `dimod` | `pip install "sc-neurocore[annealing]"` | Dedicated CI matrix lane `test-optional-extras (annealing)` installs `.[dev,annealing]` and runs `tests/test_bridges/test_quantum_annealing_neal_parity.py -q -rs`; default CPU jobs still skip the parity selector when either package is absent. | `tests/test_bridges/test_quantum_annealing_neal_parity.py`, `.github/workflows/ci.yml` |
| `onnx` | `onnx` | `dev`, `full`, `research` | Dedicated CI matrix lane `test-optional-extras (onnx-protobuf)` installs `.[dev]` and runs `tests/test_export/test_onnx_exporter_protobuf_graph.py -q -rs`; JSON export tests run without ONNX, while protobuf export tests skip when ONNX is absent outside that lane. | `tests/test_export/test_onnx_exporter_protobuf_graph.py`, `tests/test_export/test_onnx_export.py`, `.github/workflows/ci.yml` |
| `lava-nc` | `lava` on Python `<3.11` | `pip install "sc-neurocore[lava]"` on supported Python | Loihi/Lava adapter packages are handoff artefacts; the dependency is recorded in generated adapter manifests rather than imported by default CI. | `tests/test_nir_neuromorphic_adapters.py` |
| `snntorch` | `snntorch` | Not declared in `pyproject.toml`; install `snntorch==0.9.4` manually for the reproduced NIR interop profile. | Verified with `nir==1.0.8` and `nirtorch==2.6`. snnTorch 1.0.0 currently fails inside its upstream NIR exporter before SC-NeuroCore receives a graph, so it is not a supported interop profile. The exporter test skips when snnTorch is absent outside an explicit interop run. | `tests/test_nir_bridge/test_scalar_broadcast.py`, `docs/guides/nir_integration.md` |
| `spikingjelly` | `spikingjelly.activation_based` | Not declared in `pyproject.toml`; install upstream commit `2797c5575515b59d7f09a3d5b732d6d25e148d13` manually for NIR export support. | Verified with `nir==1.0.7`. NIR 1.0.8 removed `input_type`/`output_type` from `nir.LIF` while this upstream exporter still supplies them, so the latest/latest pairing fails before SC-NeuroCore receives a graph. The test skips when SpikingJelly is absent outside an explicit interop run. | `tests/test_nir_bridge/test_spikingjelly_interop.py`, `docs/guides/nir_integration.md` |
| `cupy-cuda12x>=12.0` | `cupy`, `cupyx` | `pip install "sc-neurocore[gpu]"` | GPU path depends on local CUDA compatibility; default and CI CPU paths must not import CuPy. PyTorch training uses its separate `training` extra, and `auto_device()` falls back when the installed PyTorch build does not support the local GPU compute capability. | `pyproject.toml`, `docs/guides/faq.md`, `docs/api/training.md` |
| `mpi4py>=3.0` | `mpi4py` | `pip install "sc-neurocore[mpi]"` | Dedicated CI matrix lane `test-optional-extras (mpi)` installs Open MPI plus `.[dev,mpi]` and runs `tests/test_mpi_runner_real.py -q -rs`; distributed execution outside CI requires a local `mpirun` runtime. | `tests/test_mpi_runner_real.py`, `tests/_mpi_helpers/mpi_runner_worker.py`, `.github/workflows/ci.yml` |

## Rules

- Do not add a package to the base dependency set just to satisfy an optional
  test.
- If a test uses `pytest.importorskip`, the matrix must name the import and the
  install profile or state that the dependency is manual.
- If an optional dependency becomes part of a declared profile, update
  `pyproject.toml`, this matrix, and the focused matrix contract test together.
- CI jobs that install optional profiles must name the profile explicitly and
  run the matching focused test selector.
