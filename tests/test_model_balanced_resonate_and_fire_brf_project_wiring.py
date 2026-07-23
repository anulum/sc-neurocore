# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBRFProjectWiring from former test_model_balanced_resonate_and_fire.py

"""Focused suite: TestBRFProjectWiring from former test_model_balanced_resonate_and_fire.py."""

from __future__ import annotations

from tests.model_balanced_resonate_and_fire_support import *  # noqa: F403

class TestBRFProjectWiring:
    def test_polyglot_mirror_files_exist_with_brf_equations(self) -> None:
        equation_paths = [
            "src/sc_neurocore/accel/rust/safety/balanced_resonate_and_fire.rs",
            "engine/src/neurons/simple_spiking/balanced_resonate_and_fire.rs",
            "src/sc_neurocore/accel/go/services/balanced_resonate_and_fire.go",
            "src/sc_neurocore/accel/julia/neurons/balanced_resonate_and_fire.jl",
            "src/sc_neurocore/accel/mojo/kernels/balanced_resonate_and_fire.mojo",
        ]
        for relative_path in equation_paths:
            path = REPO_ROOT / relative_path
            body = path.read_text(encoding="utf-8")
            assert "sustain" in body.lower()
            assert "b_offset" in body.lower() or "boffset" in body.lower()
            assert "gamma" in body.lower()
        binding_source = (
            REPO_ROOT / "engine/src/bindings/simple_spiking/balanced_resonate_and_fire.rs"
        )
        assert "PyBalancedResonateAndFireNeuron" in binding_source.read_text(encoding="utf-8")
        variant_source = REPO_ROOT / "engine/src/network_runner/neuron_variant.rs"
        assert "BalancedResonateAndFire" in variant_source.read_text(encoding="utf-8")

    def test_benchmark_and_documentation_are_wired(self) -> None:
        assert (REPO_ROOT / "benchmarks/bench_balanced_resonate_and_fire.py").exists()
        assert (REPO_ROOT / "benchmarks/results/bench_balanced_resonate_and_fire.json").exists()
        benchmark = (
            REPO_ROOT / "benchmarks/results/bench_balanced_resonate_and_fire.json"
        ).read_text(encoding="utf-8")
        assert '"python_step_ns"' in benchmark
        assert '"rust_pyo3_step_ns"' in benchmark
        assert '"go_step_ns"' in benchmark
        assert '"julia_step_ns"' in benchmark
        assert '"mojo_step_ns"' in benchmark
        doc = (REPO_ROOT / "docs/api/models/balanced_resonate_and_fire.md").read_text(
            encoding="utf-8"
        )
        assert "Algorithm 1" in doc
        assert "bench_balanced_resonate_and_fire.py" in doc
