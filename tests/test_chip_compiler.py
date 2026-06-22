# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Tests for sc_neurocore.chip_compiler (multi-chip compiler)

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.chip_compiler import (
    ChipSpec,
    CoreSpec,
    BUILTIN_CHIPS,
    compile_for_chip,
    CompilationResult,
)


class TestChipSpec:
    def test_loihi2(self):
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.total_neurons == 128 * 128
        assert chip.core.has_on_chip_learning
        assert "LIF" in chip.core.supported_neuron_types

    def test_xylo(self):
        chip = BUILTIN_CHIPS["xylo"]
        assert chip.total_neurons == 1000
        assert not chip.core.has_on_chip_learning

    def test_speck(self):
        chip = BUILTIN_CHIPS["speck"]
        assert chip.core.weight_bits == 4

    def test_akida(self):
        chip = BUILTIN_CHIPS["akida"]
        assert chip.vendor == "BrainChip"

    def test_spinnaker2(self):
        chip = BUILTIN_CHIPS["spinnaker2"]
        assert chip.core.max_delay_steps == 256

    def test_brainscales2(self):
        chip = BUILTIN_CHIPS["brainscales2"]
        assert chip.analog_noise_cv == 0.20

    def test_fits(self):
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.fits(1000)
        assert not chip.fits(100000)

    def test_cores_needed(self):
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.cores_needed(100) == 1
        assert chip.cores_needed(200) == 2
        assert chip.cores_needed(128) == 1

    def test_total_power(self):
        chip = BUILTIN_CHIPS["loihi2"]
        assert chip.total_power_mw == 128 * 0.5

    def test_builtin_count(self):
        assert len(BUILTIN_CHIPS) >= 6


class TestChipSpecLoadingGuards:
    def test_load_chip_spec_rejects_invalid_json(self, tmp_path):
        from sc_neurocore.chip_compiler.chip_spec import load_chip_spec

        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid chip spec JSON"):
            load_chip_spec(bad)

    def test_validate_core_payload_rejects_non_object(self):
        from sc_neurocore.chip_compiler.chip_spec import _validate_core_payload

        with pytest.raises(ValueError, match="core must be an object"):
            _validate_core_payload([1, 2], source="spec")

    def test_required_float_rejects_non_numeric(self):
        from sc_neurocore.chip_compiler.chip_spec import _required_float

        with pytest.raises(ValueError, match="must be numeric"):
            _required_float({"freq": "fast"}, "freq", "spec")


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
