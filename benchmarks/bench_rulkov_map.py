#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov 2002 source-event benchmark

"""Measure all five runtimes of the source-faithful Rulkov 2002 map."""

from __future__ import annotations

import sys

from rulkov_map_benchmark_support import BenchmarkSpec, run_benchmark
from sc_neurocore.neurons.models import rulkov_map
from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron


SPEC = BenchmarkSpec(
    benchmark="rulkov_map_simulate",
    model="RulkovMapNeuron",
    title="Rulkov 2002 source reset-branch event",
    event_semantics="pre_update_rightmost_reset_branch",
    model_factory=RulkovMapNeuron,
    backend_probes={
        "python": lambda: True,
        "rust": lambda: rulkov_map._HAS_RUST,
        "julia": rulkov_map._ensure_julia_loaded,
        "go": rulkov_map._ensure_go_loaded,
        "mojo": rulkov_map._ensure_mojo_loaded,
    },
    unavailable_reasons={
        "rust": "engine wheel lacks the source-event symbol",
        "julia": "juliacall or the source-event Julia module is unavailable",
        "go": "the source-event Go shared library is unavailable",
        "mojo": "the source-event Mojo shared library is unavailable",
    },
    source_hash_paths=(
        "benchmarks/bench_rulkov_map.py",
        "benchmarks/rulkov_map_benchmark_support.py",
        "src/sc_neurocore/neurons/models/rulkov_map.py",
        "engine/src/neurons/rulkov_map.rs",
        "engine/src/bindings/rulkov_map.rs",
        "src/sc_neurocore/accel/go/neurons/rulkov_map/rulkov_map.go",
        "src/sc_neurocore/accel/julia/neurons/rulkov_map.jl",
        "src/sc_neurocore/accel/mojo/neurons/rulkov_map.mojo",
    ),
)


def main(argv: list[str]) -> int:
    """Run the source-event benchmark and optionally write its JSON receipt."""
    return run_benchmark(argv, SPEC)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
