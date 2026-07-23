# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompileForChip from former test_chip_compiler.py

"""Focused suite: TestCompileForChip from former test_chip_compiler.py."""

from __future__ import annotations

from tests.chip_compiler_support import *  # noqa: F403

class TestCompileForChip:
    def test_small_network_loihi2(self):
        r = compile_for_chip([(64, 32), (32, 10)], target="loihi2")
        assert r.success
        assert r.total_cores_used >= 1
        assert r.total_neurons_mapped == 42

    def test_network_too_large(self):
        # xylo has 1000 neurons max
        r = compile_for_chip([(1000, 2000)], target="xylo")
        assert not r.success
        assert any("neurons" in v.lower() for v in r.violations)

    def test_unsupported_neuron_type(self):
        r = compile_for_chip(
            [(64, 32)],
            neuron_types=["HodgkinHuxley"],
            target="speck",
        )
        assert not r.success
        assert any("not supported" in v for v in r.violations)

    def test_weight_quantization(self):
        weights = [np.random.randn(32, 64)]
        r = compile_for_chip([(64, 32)], weights=weights, target="loihi2")
        assert r.success
        assert len(r.quantized_weights) == 1
        assert r.weight_bits == 8

    def test_core_partitioning(self):
        # 500 neurons, loihi2 cores hold 128 each → 4 cores
        r = compile_for_chip([(100, 500)], target="loihi2")
        assert r.success
        assert r.total_cores_used == 4  # ceil(500/128)

    def test_unknown_target(self):
        r = compile_for_chip([(10, 5)], target="nonexistent_chip")
        assert not r.success
        assert any("Unknown" in v for v in r.violations)

    def test_analog_noise_warning(self):
        r = compile_for_chip([(64, 32)], target="brainscales2")
        assert any("noise" in w.lower() for w in r.warnings)

    def test_summary(self):
        r = compile_for_chip([(64, 32)], target="loihi2")
        s = r.summary()
        assert "loihi2" in s
        assert "SUCCESS" in s

    def test_custom_chip_spec(self):
        custom = ChipSpec(
            name="custom_fpga",
            vendor="Custom",
            total_cores=4,
            core=CoreSpec(
                max_neurons=64,
                max_synapses_per_neuron=256,
                weight_bits=8,
                supported_neuron_types=["LIF"],
            ),
        )
        r = compile_for_chip([(32, 16)], target=custom)
        assert r.success
        assert r.chip == "custom_fpga"

    def test_fan_out_violation(self):
        r = compile_for_chip([(10, 10000)], target="xylo")
        assert not r.success

    def test_all_builtin_chips_compile(self):
        for name in BUILTIN_CHIPS:
            r = compile_for_chip([(16, 8)], target=name)
            assert isinstance(r, CompilationResult)
