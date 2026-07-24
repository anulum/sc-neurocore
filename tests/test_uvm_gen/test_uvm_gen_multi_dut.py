# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiDUT from former test_uvm_gen.py

"""Focused suite: TestMultiDUT from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403


class TestMultiDUT:
    def test_generate_multi(self):
        gen = UVMGenerator()
        benchmarks = gen.generate_multi([lif_module(), dense_module()])
        assert len(benchmarks) == 2
        assert benchmarks[0].module_name == "sc_lif_neuron"
        assert benchmarks[1].module_name == "sc_dense_layer_core"

    def test_multi_independent(self):
        gen = UVMGenerator()
        benchmarks = gen.generate_multi([lif_module(), dense_module()])
        assert benchmarks[0].top_sv != benchmarks[1].top_sv
