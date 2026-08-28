#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained upward-crossing Rulkov benchmark

"""Measure all five runtimes of the retained SC Rulkov event identity."""

from __future__ import annotations

import sys

from rulkov_map_benchmark_support import BenchmarkSpec, run_benchmark
from sc_neurocore.neurons.models import sc_upward_crossing_rulkov_map
from sc_neurocore.neurons.models.sc_upward_crossing_rulkov_map import (
    SCUpwardCrossingRulkovMapNeuron,
)


SPEC = BenchmarkSpec(
    benchmark="sc_upward_crossing_rulkov_map_simulate",
    model="SCUpwardCrossingRulkovMapNeuron",
    title="retained SC configurable upward-crossing Rulkov event",
    event_semantics="post_update_configurable_upward_x_crossing",
    model_factory=SCUpwardCrossingRulkovMapNeuron,
    backend_probes={
        "python": lambda: True,
        "rust": lambda: sc_upward_crossing_rulkov_map._HAS_RUST,
        "julia": sc_upward_crossing_rulkov_map._ensure_julia_loaded,
        "go": sc_upward_crossing_rulkov_map._ensure_go_loaded,
        "mojo": sc_upward_crossing_rulkov_map._ensure_mojo_loaded,
    },
    unavailable_reasons={
        "rust": "engine wheel lacks the retained-event symbol",
        "julia": "juliacall or the retained-event Julia module is unavailable",
        "go": "the retained-event Go shared library is unavailable",
        "mojo": "the retained-event Mojo shared library is unavailable",
    },
    source_hash_paths=(
        "benchmarks/bench_sc_upward_crossing_rulkov_map.py",
        "benchmarks/rulkov_map_benchmark_support.py",
        "src/sc_neurocore/neurons/models/sc_upward_crossing_rulkov_map.py",
        "engine/src/neurons/sc_upward_crossing_rulkov_map.rs",
        "engine/src/bindings/sc_upward_crossing_rulkov_map.rs",
        "src/sc_neurocore/accel/go/neurons/sc_upward_crossing_rulkov_map/"
        "sc_upward_crossing_rulkov_map.go",
        "src/sc_neurocore/accel/julia/neurons/sc_upward_crossing_rulkov_map.jl",
        "src/sc_neurocore/accel/mojo/neurons/sc_upward_crossing_rulkov_map.mojo",
    ),
)


def main(argv: list[str]) -> int:
    """Run the retained-event benchmark and optionally write its JSON receipt."""
    return run_benchmark(argv, SPEC)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
