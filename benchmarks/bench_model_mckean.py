#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Source/binary-bound five-runtime McKean/Tonnelier benchmark."""

from benchmarks._mckean_benchmark import BenchmarkSpec, REPOSITORY, run
from sc_neurocore.accel import mckean as backends

SOURCE_PATHS = (
    "benchmarks/_mckean_benchmark.py",
    "benchmarks/_non_resetting_lif_benchmark.py",
    "benchmarks/bench_model_mckean.py",
    "engine/src/bindings/mckean.rs",
    "engine/src/neurons/simple_spiking/mckean.rs",
    "src/sc_neurocore/accel/mckean.py",
    "src/sc_neurocore/accel/go/mckean/mckean.go",
    "src/sc_neurocore/accel/go/services/mckean.go",
    "src/sc_neurocore/accel/julia/neurons/mckean.jl",
    "src/sc_neurocore/accel/mojo/kernels/mckean.mojo",
    "src/sc_neurocore/accel/mojo/mckean/mckean_abi.mojo",
    "src/sc_neurocore/accel/rust/safety/mckean.rs",
    "src/sc_neurocore/neurons/model_descriptors/McKeanNeuron.toml",
    "src/sc_neurocore/neurons/model_schemas/mckean.json",
    "src/sc_neurocore/neurons/model_schemas/mckean.toml",
    "src/sc_neurocore/neurons/models/mckean.py",
    "src/sc_neurocore/neurons/reference_receipts/mckean_tonnelier.json",
)
SPEC = BenchmarkSpec(
    benchmark="McKean/Tonnelier source Heaviside coupled RK4",
    model="McKeanNeuron",
    output=REPOSITORY / "benchmarks/results/bench_mckean.json",
    current=3.0,
    steps=20_000,
    repeats=5,
    simulate=backends.simulate_mckean,
    backend_available=backends.backend_available,
    parity_atol=backends.PARITY_ATOL,
    state_keys=("voltages", "recovery"),
    final_keys=("v_final", "w_final"),
    source_paths=SOURCE_PATHS,
    go_library="src/sc_neurocore/accel/go/mckean/libmckean.so",
    mojo_library="src/sc_neurocore/accel/mojo/mckean/libmckean.so",
)


def main(argv: list[str] | None = None) -> int:
    return int(run(SPEC, argv))


if __name__ == "__main__":
    raise SystemExit(main())
