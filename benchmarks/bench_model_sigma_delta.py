#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Source/binary-bound five-runtime sampled APSDM benchmark."""

from benchmarks._sigma_delta_benchmark import BenchmarkSpec, REPOSITORY, run
from sc_neurocore.accel import sigma_delta as backends

SOURCE_PATHS = (
    "benchmarks/_sigma_delta_benchmark.py",
    "benchmarks/bench_model_sigma_delta.py",
    "engine/src/bindings/trivial/sigma_delta.rs",
    "engine/src/neurons/trivial/sigma_delta.rs",
    "src/sc_neurocore/accel/sigma_delta.py",
    "src/sc_neurocore/accel/go/sigma_delta/sigma_delta.go",
    "src/sc_neurocore/accel/go/services/sigma_delta.go",
    "src/sc_neurocore/accel/julia/neurons/sigma_delta.jl",
    "src/sc_neurocore/accel/mojo/kernels/sigma_delta.mojo",
    "src/sc_neurocore/accel/mojo/sigma_delta/sigma_delta.mojo",
    "src/sc_neurocore/accel/rust/safety/sigma_delta.rs",
    "src/sc_neurocore/neurons/model_descriptors/SigmaDeltaNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/sigma_delta.json",
    "src/sc_neurocore/neurons/model_schemas/sigma_delta.toml",
    "src/sc_neurocore/neurons/models/sigma_delta.py",
    "src/sc_neurocore/neurons/reference_trace_data/sigma_delta_apsdm.json",
)
SPEC = BenchmarkSpec(
    benchmark="Yoon sampled asynchronous pulse sigma-delta source recurrence",
    model="SigmaDeltaNeuron",
    output=REPOSITORY / "benchmarks/results/bench_sigma_delta.json",
    current=2.0,
    steps=200_000,
    repeats=5,
    simulate=backends.simulate_sigma_delta,
    backend_available=backends.backend_available,
    parity_atol=backends.PARITY_ATOL,
    state_keys=("sigma", "reconstruction"),
    final_keys=("sigma_final", "reconstruction_final"),
    source_paths=SOURCE_PATHS,
    go_library="src/sc_neurocore/accel/go/sigma_delta/libsigma_delta.so",
    mojo_library="src/sc_neurocore/accel/mojo/sigma_delta/libsigma_delta.so",
)


def main(argv: list[str] | None = None) -> int:
    """Write parity-clean local sampled APSDM evidence."""
    return int(run(SPEC, argv))


if __name__ == "__main__":
    raise SystemExit(main())
