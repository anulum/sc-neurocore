#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Source/binary-bound five-runtime retained bipolar benchmark."""

from benchmarks._sigma_delta_benchmark import BenchmarkSpec, REPOSITORY, run
from sc_neurocore.accel import sc_sigma_delta_accumulator as backends

SOURCE_PATHS = (
    "benchmarks/_sigma_delta_benchmark.py",
    "benchmarks/bench_model_sc_sigma_delta_accumulator.py",
    "engine/src/bindings/trivial/sc_sigma_delta_accumulator.rs",
    "engine/src/neurons/trivial/sc_sigma_delta_accumulator.rs",
    "src/sc_neurocore/accel/sc_sigma_delta_accumulator.py",
    "src/sc_neurocore/accel/go/sc_sigma_delta_accumulator/sc_sigma_delta_accumulator.go",
    "src/sc_neurocore/accel/go/services/sc_sigma_delta_accumulator.go",
    "src/sc_neurocore/accel/julia/neurons/sc_sigma_delta_accumulator.jl",
    "src/sc_neurocore/accel/mojo/kernels/sc_sigma_delta_accumulator.mojo",
    "src/sc_neurocore/accel/mojo/sc_sigma_delta_accumulator/sc_sigma_delta_accumulator.mojo",
    "src/sc_neurocore/accel/rust/safety/sc_sigma_delta_accumulator.rs",
    "src/sc_neurocore/neurons/model_descriptors/SCSigmaDeltaAccumulatorNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/sc_sigma_delta_accumulator.json",
    "src/sc_neurocore/neurons/model_schemas/sc_sigma_delta_accumulator.toml",
    "src/sc_neurocore/neurons/models/sc_sigma_delta_accumulator.py",
    "src/sc_neurocore/neurons/reference_trace_data/sc_sigma_delta_accumulator_project.json",
)
SPEC = BenchmarkSpec(
    benchmark="SC retained bipolar sigma-delta accumulator recurrence",
    model="SCSigmaDeltaAccumulatorNeuron",
    output=REPOSITORY / "benchmarks/results/bench_sc_sigma_delta_accumulator.json",
    current=0.3,
    steps=200_000,
    repeats=5,
    simulate=backends.simulate_sc_sigma_delta_accumulator,
    backend_available=backends.backend_available,
    parity_atol=backends.PARITY_ATOL,
    state_keys=("sigma",),
    final_keys=("sigma_final",),
    source_paths=SOURCE_PATHS,
    go_library="src/sc_neurocore/accel/go/sc_sigma_delta_accumulator/libsc_sigma_delta_accumulator.so",
    mojo_library="src/sc_neurocore/accel/mojo/sc_sigma_delta_accumulator/libsc_sigma_delta_accumulator.so",
)


def main(argv: list[str] | None = None) -> int:
    """Write parity-clean local retained-project evidence."""
    return int(run(SPEC, argv))


if __name__ == "__main__":
    raise SystemExit(main())
