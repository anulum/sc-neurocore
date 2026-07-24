# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (seed_and_rejects) from former test_stochastic_source_emitters.py

from __future__ import annotations

from tests.test_hdl_gen.stochastic_source_emitters_support import *  # noqa: F403

def test_stochastic_source_emitters_reject_invalid_module_names():
    with pytest.raises(ValueError, match="Invalid module name"):
        Lfsr16Emitter(module_name="lfsr-16 source")

    with pytest.raises(ValueError, match="Invalid module name"):
        Sobol16Emitter(module_name="sobol-16 source")


def test_lfsr16_emitter_zero_seed_falls_back_to_default():
    """Seed 0 is an absorbing state for the LFSR; emitter must reject it."""
    emitter = Lfsr16Emitter(module_name="lfsr16_zero", seed=0x0000)
    assert emitter.seed == 0xACE1
    verilog = emitter.generate()
    assert "16'hACE1" in verilog


def test_lfsr16_emitter_masks_seed_to_16_bits():
    """Seeds wider than 16 bits are silently masked to preserve module contract."""
    emitter = Lfsr16Emitter(module_name="lfsr16_mask", seed=0x1234_BEEF)
    assert emitter.seed == 0xBEEF


def test_sobol16_emitter_masks_seed_to_16_bits():
    """Same 16-bit mask guarantee for the Sobol emitter."""
    emitter = Sobol16Emitter(module_name="sobol16_mask", seed=0x00FF_0042)
    assert emitter.seed == 0x0042
    verilog = emitter.generate()
    assert "16'h0042" in verilog


def test_require_positive_int_rejects_non_positive_value():
    with pytest.raises(ValueError, match="must be a positive integer"):
        VerilogGenerator._require_positive_int(0, "width")


def test_emit_async_aer_wraps_declared_dense_layers():
    generator = VerilogGenerator(module_name="async_route")
    generator.add_layer("Dense", "dense0", {"n_neurons": 4})
    verilog = generator.emit_async_aer()
    assert "module async_route" in verilog


def test_emit_quasirandom_source_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be 'sobol' or 'halton'"):
        VerilogGenerator().emit_quasirandom_source(method="mt19937")


def test_emit_sources_from_ir_rejects_non_collection_payload():
    with pytest.raises(TypeError, match="mapping or sequence of nodes"):
        emit_sources_from_ir(42)


def test_emit_sources_from_ir_rejects_source_without_generator():
    with pytest.raises(ValueError, match="missing source_type/decorrelator"):
        emit_sources_from_ir([{"type": "stochastic_source"}])


def test_emit_sources_from_ir_defaults_unnamed_source_module() -> None:
    verilog = emit_sources_from_ir([{"type": "stochastic_source", "source_type": "sobol"}])
    assert "sc_stochastic_source_0" in verilog
