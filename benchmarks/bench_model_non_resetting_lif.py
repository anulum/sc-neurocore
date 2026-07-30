#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Source-bound five-runtime Kobayashi MAT(1) benchmark."""

from __future__ import annotations

from benchmarks._non_resetting_lif_benchmark import BenchmarkSpec, REPOSITORY, run
from sc_neurocore.accel import non_resetting_lif as backends

SOURCE_PATHS = (
    "benchmarks/_non_resetting_lif_benchmark.py",
    "benchmarks/bench_model_non_resetting_lif.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/trivial/non_resetting_lif.rs",
    "engine/src/neurons/trivial/non_resetting_lif.rs",
    "src/sc_neurocore/accel/non_resetting_lif.py",
    "src/sc_neurocore/accel/go/non_resetting_lif/non_resetting_lif.go",
    "src/sc_neurocore/accel/go/services/non_resetting_lif.go",
    "src/sc_neurocore/accel/go/services/non_resetting_lif_test.go",
    "src/sc_neurocore/accel/julia/neurons/non_resetting_lif.jl",
    "src/sc_neurocore/accel/mojo/kernels/non_resetting_lif.mojo",
    "src/sc_neurocore/accel/mojo/non_resetting_lif/non_resetting_lif.mojo",
    "src/sc_neurocore/accel/rust/safety/non_resetting_lif.rs",
    "src/sc_neurocore/neurons/model_descriptors/NonResettingLIFNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/non_resetting_lif.json",
    "src/sc_neurocore/neurons/model_schemas/non_resetting_lif.toml",
    "src/sc_neurocore/neurons/models/non_resetting_lif.py",
    "src/sc_neurocore/neurons/reference_trace_data/non_resetting_lif_mat1.json",
)
SPEC = BenchmarkSpec(
    benchmark="Kobayashi MAT(1) non-resetting source recurrence",
    model="NonResettingLIFNeuron",
    output=REPOSITORY / "benchmarks/results/bench_non_resetting_lif.json",
    current=0.7,
    steps=200_000,
    repeats=5,
    simulate=backends.simulate_non_resetting_lif,
    backend_available=backends.backend_available,
    parity_atol=backends.PARITY_ATOL,
    state_keys=("voltages", "theta", "refractory"),
    final_keys=("v_final", "theta_final", "refractory_final"),
    source_paths=SOURCE_PATHS,
    go_library="src/sc_neurocore/accel/go/non_resetting_lif/libnon_resetting_lif.so",
    mojo_library="src/sc_neurocore/accel/mojo/non_resetting_lif/libnon_resetting_lif.so",
)


def main(argv: list[str] | None = None) -> int:
    """Write parity-clean local MAT(1) evidence."""
    return int(run(SPEC, argv))


if __name__ == "__main__":
    raise SystemExit(main())
