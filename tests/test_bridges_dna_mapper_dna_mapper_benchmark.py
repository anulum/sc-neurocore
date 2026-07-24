# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDNAMapperBenchmark from former test_bridges_dna_mapper.py

"""Focused suite: TestDNAMapperBenchmark from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from tests.bridges_dna_mapper_support import *  # noqa: F403


class TestDNAMapperBenchmark:
    def test_sequence_generation_throughput(self) -> None:
        """Generate 500 random 20nt sequences."""
        designer = SequenceDesigner(seed=42)
        t0 = time.perf_counter()
        for _ in range(500):
            designer.generate(length=20)
        elapsed = time.perf_counter() - t0
        max_elapsed = 10.0 if os.environ.get("CI") else 5.0
        assert elapsed < max_elapsed, f"500 sequences took {elapsed:.2f}s"

    def test_gate_compilation_throughput(self) -> None:
        """Compile 100 AND gates."""
        compiler = StrandDisplacementCompiler()
        t0 = time.perf_counter()
        for i in range(100):
            compiler.compile_and(f"a_{i}", f"b_{i}", f"out_{i}")
        elapsed = time.perf_counter() - t0
        max_elapsed = 20.0 if os.environ.get("CI") else 10.0
        assert elapsed < max_elapsed, f"100 AND gates took {elapsed:.2f}s"
