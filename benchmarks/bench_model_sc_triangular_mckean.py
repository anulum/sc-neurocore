#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Source/binary-bound five-runtime retained SC recurrence benchmark."""

from benchmarks._mckean_benchmark import BenchmarkSpec, REPOSITORY, run
from sc_neurocore.accel import sc_triangular_mckean as backends

SOURCE_PATHS = (
    "benchmarks/_mckean_benchmark.py",
    "benchmarks/_non_resetting_lif_benchmark.py",
    "benchmarks/bench_model_sc_triangular_mckean.py",
    "engine/src/bindings/sc_triangular_mckean.rs",
    "engine/src/neurons/simple_spiking/sc_triangular_mckean.rs",
    "src/sc_neurocore/accel/sc_triangular_mckean.py",
    "src/sc_neurocore/accel/go/neurons/mckean/mckean.go",
    "src/sc_neurocore/accel/julia/neurons/sc_triangular_mckean.jl",
    "src/sc_neurocore/accel/mojo/neurons/sc_triangular_mckean.mojo",
    "src/sc_neurocore/accel/rust/safety/sc_triangular_mckean.rs",
    "src/sc_neurocore/neurons/model_descriptors/SCTriangularMcKeanNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/sc_triangular_mckean.json",
    "src/sc_neurocore/neurons/model_schemas/sc_triangular_mckean.toml",
    "src/sc_neurocore/neurons/models/sc_triangular_mckean.py",
    "src/sc_neurocore/neurons/reference_trace_data/sc_triangular_mckean_project.json",
)
SPEC = BenchmarkSpec(
    benchmark="SC retained triangular piecewise-linear RK4 recurrence",
    model="SCTriangularMcKeanNeuron",
    output=REPOSITORY / "benchmarks/results/bench_sc_triangular_mckean.json",
    current=0.5,
    steps=512,
    repeats=5,
    simulate=backends.simulate_sc_triangular_mckean,
    backend_available=backends.backend_available,
    parity_atol=backends.PARITY_ATOL,
    state_keys=("voltages", "recovery"),
    final_keys=("v_final", "w_final"),
    source_paths=SOURCE_PATHS,
    go_library="src/sc_neurocore/accel/go/neurons/mckean/libsc_triangular_mckean.so",
    mojo_library="src/sc_neurocore/accel/mojo/neurons/libsc_triangular_mckean.so",
)


def main(argv: list[str] | None = None) -> int:
    return int(run(SPEC, argv))


if __name__ == "__main__":
    raise SystemExit(main())
