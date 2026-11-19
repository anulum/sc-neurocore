# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog generator validation tests

"""Width, layer topology, neuron count, dense-core parameter, and source-seed validation."""

from __future__ import annotations

import pytest

from sc_neurocore.hdl_gen.aer_emitter import (
    render_dense_layer_instance,
    validate_dense_layer_params,
)
from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator, _source_seed


def test_verilog_generator_rejects_mismatched_dense_widths() -> None:
    """Adjacent Dense layers must agree on the inter-layer bus width."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {"n_neurons": 5, "output_width": 5})
    gen.add_layer("Dense", "dense1", {"n_neurons": 3, "input_width": 7})

    with pytest.raises(ValueError, match="dense0 -> dense1 width mismatch"):
        gen.generate()


def test_verilog_generator_requires_dense_neuron_count() -> None:
    """Dense sync generation must not invent omitted neuron counts."""
    gen = VerilogGenerator()
    gen.add_layer("Dense", "dense0", {})
    with pytest.raises(ValueError, match="Dense layer 'dense0' requires n_neurons"):
        gen.generate()


def test_verilog_generator_rejects_unsupported_sync_layer() -> None:
    """Unsupported sync layers must fail closed instead of being silently dropped."""
    gen = VerilogGenerator()
    gen.add_layer("Custom", "custom0", {})
    with pytest.raises(ValueError, match="unsupported sync layer type 'Custom'"):
        gen.generate()


def test_verilog_generator_rejects_invalid_dense_core_parameters() -> None:
    """Reject duplicate layer names, mismatched output_width, and invalid dense parameters."""
    dup = VerilogGenerator()
    dup.add_layer("Dense", "dense0", {"n_neurons": 4})
    dup.add_layer("Dense", "dense0", {"n_neurons": 4})
    with pytest.raises(ValueError, match="duplicate layer name"):
        dup.generate()

    bad_out = VerilogGenerator()
    bad_out.add_layer("Dense", "dense0", {"n_neurons": 4, "output_width": 2})
    with pytest.raises(ValueError, match="output_width"):
        bad_out.generate()

    bad_dw = VerilogGenerator()
    bad_dw.add_layer("Dense", "dense0", {"n_neurons": 4, "data_width": 8})
    with pytest.raises(ValueError, match="data_width"):
        bad_dw.generate()

    bad_slen = VerilogGenerator()
    bad_slen.add_layer("Dense", "dense0", {"n_neurons": 4, "stream_len": 1 << 32})
    with pytest.raises(ValueError, match="stream_len"):
        bad_slen.generate()

    bad_wfp = VerilogGenerator()
    bad_wfp.add_layer("Dense", "dense0", {"n_neurons": 4, "weight_fp": -1})
    with pytest.raises(ValueError, match="weight_fp"):
        bad_wfp.generate()

    bad_wfp_bool = VerilogGenerator()
    bad_wfp_bool.add_layer("Dense", "dense0", {"n_neurons": 4, "weight_fp": True})
    with pytest.raises(ValueError, match="weight_fp"):
        bad_wfp_bool.generate()

    both_weights = VerilogGenerator()
    both_weights.add_layer("Dense", "dense0", {"n_neurons": 4, "weight_fp": 1, "weights": [1] * 8})
    with pytest.raises(ValueError, match="both weights and weight_fp"):
        both_weights.generate()

    bad_weights_type = VerilogGenerator()
    bad_weights_type.add_layer("Dense", "dense0", {"n_neurons": 4, "weights": "bad"})
    with pytest.raises(ValueError, match="weights"):
        bad_weights_type.generate()

    bad_weights_len = VerilogGenerator()
    bad_weights_len.add_layer("Dense", "dense0", {"n_neurons": 4, "weights": [1, 2]})
    with pytest.raises(ValueError, match="weights"):
        bad_weights_len.generate()

    bad_signed = VerilogGenerator()
    bad_signed.add_layer("Dense", "dense0", {"n_neurons": 4, "y_min_fp": 40000})
    with pytest.raises(ValueError, match="y_min_fp"):
        bad_signed.generate()

    bad_signed_bool = VerilogGenerator()
    bad_signed_bool.add_layer("Dense", "dense0", {"n_neurons": 4, "cfg_leak": False})
    with pytest.raises(ValueError, match="cfg_leak"):
        bad_signed_bool.generate()

    with pytest.raises(TypeError, match="params must be a mapping"):
        validate_dense_layer_params("dense0", "not-a-mapping")

    with pytest.raises(ValueError, match="weights must be a list or tuple"):
        render_dense_layer_instance(
            comment_label="Layer 0",
            layer_name="dense0",
            params={"n_neurons": 2, "weights": 99},
            input_bus="input_bus",
            output_bus="output_bus",
            input_width=2,
        )


def test_source_seed_rejects_non_integer_seed() -> None:
    """A stochastic-source seed must be a real integer, not a bool or string."""
    with pytest.raises(ValueError, match="seed must be an integer"):
        _source_seed({"seed": True}, default=0)
    with pytest.raises(ValueError, match="seed must be an integer"):
        _source_seed({"seed": "not-an-int"}, default=0)
