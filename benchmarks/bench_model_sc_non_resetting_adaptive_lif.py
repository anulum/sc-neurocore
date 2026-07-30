#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Source-bound five-runtime retained SC adaptive-LIF benchmark."""

from __future__ import annotations

from benchmarks._non_resetting_lif_benchmark import BenchmarkSpec, REPOSITORY, run
from sc_neurocore.accel import sc_non_resetting_adaptive_lif as backends

SOURCE_PATHS = (
    "benchmarks/_non_resetting_lif_benchmark.py",
    "benchmarks/bench_model_sc_non_resetting_adaptive_lif.py",
    "bridge/sc_neurocore_engine/__init__.py",
    "engine/src/bindings/trivial/sc_non_resetting_adaptive_lif.rs",
    "engine/src/neurons/trivial/sc_non_resetting_adaptive_lif.rs",
    "src/sc_neurocore/accel/sc_non_resetting_adaptive_lif.py",
    "src/sc_neurocore/accel/go/sc_non_resetting_adaptive_lif/sc_non_resetting_adaptive_lif.go",
    "src/sc_neurocore/accel/go/services/sc_non_resetting_adaptive_lif.go",
    "src/sc_neurocore/accel/go/services/sc_non_resetting_adaptive_lif_test.go",
    "src/sc_neurocore/accel/julia/neurons/sc_non_resetting_adaptive_lif.jl",
    "src/sc_neurocore/accel/mojo/kernels/sc_non_resetting_adaptive_lif.mojo",
    "src/sc_neurocore/accel/mojo/sc_non_resetting_adaptive_lif/sc_non_resetting_adaptive_lif.mojo",
    "src/sc_neurocore/accel/rust/safety/sc_non_resetting_adaptive_lif.rs",
    "src/sc_neurocore/neurons/model_descriptors/SCNonResettingAdaptiveLIFNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/sc_non_resetting_adaptive_lif.json",
    "src/sc_neurocore/neurons/model_schemas/sc_non_resetting_adaptive_lif.toml",
    "src/sc_neurocore/neurons/models/sc_non_resetting_adaptive_lif.py",
    "src/sc_neurocore/neurons/reference_trace_data/sc_non_resetting_adaptive_lif_project.json",
)
SPEC = BenchmarkSpec(
    benchmark="SC non-resetting adaptive LIF exact-relaxation recurrence",
    model="SCNonResettingAdaptiveLIFNeuron",
    output=REPOSITORY / "benchmarks/results/bench_sc_non_resetting_adaptive_lif.json",
    current=20.0,
    steps=200_000,
    repeats=5,
    simulate=backends.simulate_sc_non_resetting_adaptive_lif,
    backend_available=backends.backend_available,
    parity_atol=backends.PARITY_ATOL,
    state_keys=("voltages", "theta"),
    final_keys=("v_final", "theta_final"),
    source_paths=SOURCE_PATHS,
    go_library=(
        "src/sc_neurocore/accel/go/sc_non_resetting_adaptive_lif/"
        "libsc_non_resetting_adaptive_lif.so"
    ),
    mojo_library=(
        "src/sc_neurocore/accel/mojo/sc_non_resetting_adaptive_lif/"
        "libsc_non_resetting_adaptive_lif.so"
    ),
)


def main(argv: list[str] | None = None) -> int:
    """Write parity-clean local retained-project evidence."""
    return int(run(SPEC, argv))


if __name__ == "__main__":
    raise SystemExit(main())
