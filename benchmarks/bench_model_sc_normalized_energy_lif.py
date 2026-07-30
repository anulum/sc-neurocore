#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Source/binary-bound five-runtime retained SC EnergyLIF benchmark."""

from benchmarks._energy_lif_benchmark import BenchmarkSpec, REPOSITORY, run
from sc_neurocore.accel import sc_normalized_energy_lif as backends

SOURCE_PATHS = (
    "benchmarks/_energy_lif_benchmark.py",
    "benchmarks/_non_resetting_lif_benchmark.py",
    "benchmarks/bench_model_sc_normalized_energy_lif.py",
    "engine/src/bindings/trivial/sc_normalized_energy_lif.rs",
    "engine/src/neurons/trivial/sc_normalized_energy_lif.rs",
    "src/sc_neurocore/accel/sc_normalized_energy_lif.py",
    "src/sc_neurocore/accel/go/sc_normalized_energy_lif/sc_normalized_energy_lif.go",
    "src/sc_neurocore/accel/go/services/sc_normalized_energy_lif.go",
    "src/sc_neurocore/accel/julia/neurons/sc_normalized_energy_lif.jl",
    "src/sc_neurocore/accel/mojo/kernels/sc_normalized_energy_lif.mojo",
    "src/sc_neurocore/accel/mojo/sc_normalized_energy_lif/sc_normalized_energy_lif_abi.mojo",
    "src/sc_neurocore/accel/rust/safety/sc_normalized_energy_lif.rs",
    "src/sc_neurocore/neurons/model_descriptors/SCNormalizedEnergyLIFNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/sc_normalized_energy_lif.json",
    "src/sc_neurocore/neurons/model_schemas/sc_normalized_energy_lif.toml",
    "src/sc_neurocore/neurons/models/sc_normalized_energy_lif.py",
    "src/sc_neurocore/neurons/reference_trace_data/sc_normalized_energy_lif_project.json",
)
SPEC = BenchmarkSpec(
    benchmark="SC retained normalized energy-gated exact flow",
    model="SCNormalizedEnergyLIFNeuron",
    output=REPOSITORY / "benchmarks/results/bench_sc_normalized_energy_lif.json",
    current=30.0,
    steps=200_000,
    repeats=5,
    simulate=backends.simulate_sc_normalized_energy_lif,
    backend_available=backends.backend_available,
    parity_atol=backends.PARITY_ATOL,
    state_keys=("voltages", "epsilon"),
    final_keys=("v_final", "epsilon_final"),
    source_paths=SOURCE_PATHS,
    go_library="src/sc_neurocore/accel/go/sc_normalized_energy_lif/libsc_normalized_energy_lif.so",
    mojo_library="src/sc_neurocore/accel/mojo/sc_normalized_energy_lif/libsc_normalized_energy_lif.so",
)


def main(argv: list[str] | None = None) -> int:
    return int(run(SPEC, argv))


if __name__ == "__main__":
    raise SystemExit(main())
