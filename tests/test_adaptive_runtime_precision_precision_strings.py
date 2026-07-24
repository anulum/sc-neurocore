# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrecisionStrings from former test_adaptive_runtime_precision.py

"""Focused suite: TestPrecisionStrings from former test_adaptive_runtime_precision.py."""

from __future__ import annotations

from tests.adaptive_runtime_precision_support import *  # noqa: F403


class TestPrecisionStrings:
    """Verify new precision-string support and manifest emission."""

    def test_precision_string_api(self, lif_neuron):
        """Q-format strings should be resolved and emitted deterministically."""
        v = compile_adaptive_precision(
            lif_neuron,
            module_name="sc_lif_adapt_precision_strings",
            lp_precision="Q8.8",
            hp_precision="Q16.16",
        )
        manifest = _extract_manifest(v)
        assert manifest["lp_precision"]["kind"] == "fixed"
        assert manifest["hp_precision"]["kind"] == "fixed"
        assert manifest["lp_precision"]["label"] == "Q8.8"
        assert manifest["hp_precision"]["label"] == "Q16.16"

    def test_block_floating_precision_metadata(self, lif_neuron):
        """Block-floating precision should emit block metadata and deterministic label."""
        v = compile_adaptive_precision(
            lif_neuron,
            module_name="sc_lif_adapt_bfp",
            lp_precision="BFP16E3X32",
            hp_precision="Q16.16",
        )
        manifest = _extract_manifest(v)
        assert manifest["lp_precision"]["kind"] == "block_floating"
        assert manifest["lp_precision"]["mantissa_bits"] == 16
        assert manifest["lp_precision"]["exponent_bits"] == 3
        assert manifest["lp_precision"]["block_size"] == 32
        assert manifest["lp_precision"]["label"].startswith("BFP16E3")
        assert manifest["lp_precision"]["exponent_bias"] == 3
        assert manifest["lp_precision"]["exponent_code_range"] == [0, 7]
        assert manifest["lp_precision"]["exponent_min"] == -3
        assert manifest["lp_precision"]["exponent_max"] == 4
        assert manifest["emitter_contract_version"] == "adaptive_precision_emitter.v1"
        assert manifest["lp_precision"]["emitted_datapath_width"] == 16
        assert manifest["lp_precision"]["emitted_datapath_fraction"] == 15
        assert manifest["lp_precision"]["exponent_stream_width"] == 3
        assert manifest["lp_precision"]["exponent_vector_width"] == (
            "exponent_bits * ceil(parameter_count / block_size)"
        )
        assert manifest["lp_precision"]["emitted_datapath_contract"] == (
            "mantissa_width_fixed_datapath_with_detached_shared_exponent_stream"
        )
        assert manifest["lp_precision"]["mantissa_abs_max"] == 32_767
        assert manifest["lp_precision"]["minimum_quantum"] == pytest.approx(0.125)
        assert manifest["lp_precision"]["max_abs_value"] == pytest.approx(524_272.0)
        assert manifest["lp_precision"]["block_exponent_alignment"] == "contiguous_flattened_block"
        assert manifest["lp_precision"]["block_exponent_count"] == (
            "ceil(parameter_count / block_size)"
        )
        assert manifest["lp_precision"]["datapath_contract"] == (
            "fixed_mantissa_with_explicit_shared_exponent_metadata"
        )

    def test_block_floating_precision_metadata_carries_concrete_layout(self, lif_neuron):
        """BFP manifests must carry exact block layout when parameter count is known."""
        v = compile_adaptive_precision(
            lif_neuron,
            module_name="sc_lif_adapt_bfp_layout",
            lp_precision="BFP16E3X32",
            hp_precision="Q16.16",
            lp_parameter_count=65,
        )
        manifest = _extract_manifest(v)
        lp = manifest["lp_precision"]

        assert lp["parameter_count"] == 65
        assert lp["block_exponent_count"] == 3
        assert lp["exponent_vector_width"] == 9
        assert lp["block_exponent_layout"] == {
            "alignment": "contiguous_flattened_block",
            "flattened_order": "row_major",
            "parameter_count": 65,
            "block_size": 32,
            "exponent_count": 3,
            "last_block_size": 1,
            "exponent_index_formula": "parameter_index // block_size",
        }

    def test_block_floating_precision_metadata_rejects_bad_parameter_count(self, lif_neuron):
        """Invalid BFP parameter-count metadata must fail before RTL emission."""
        with pytest.raises(ValueError, match="parameter_count"):
            compile_adaptive_precision(
                lif_neuron,
                lp_precision="BFP16E3X32",
                hp_precision="Q16.16",
                lp_parameter_count=-1,
            )

    def test_fixed_precision_rejects_block_parameter_count(self, lif_neuron):
        """Fixed Q-format manifests must not silently accept BFP layout metadata."""
        with pytest.raises(ValueError, match="block-floating precision"):
            compile_adaptive_precision(
                lif_neuron,
                lp_precision="Q8.8",
                hp_precision="Q16.16",
                lp_parameter_count=65,
            )

    def test_invalid_precision_string(self, lif_neuron):
        """Invalid precision strings must fail with ValueError."""
        with pytest.raises(ValueError, match="precision"):
            compile_adaptive_precision(
                lif_neuron,
                lp_precision="definitely-not-a-format",
                hp_precision="Q16.16",
            )
