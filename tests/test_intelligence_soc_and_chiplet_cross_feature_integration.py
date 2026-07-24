# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossFeatureIntegration from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestCrossFeatureIntegration from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations

from tests.intelligence_soc_and_chiplet_support import *  # noqa: F403


class TestCrossFeatureIntegration:
    """End-to-end tests chaining multiple features together."""

    def test_tmr_plus_checksum(self):
        """TMR wrapper + model checksum embedding."""
        from sc_neurocore.compiler.intelligence import (
            generate_tmr_wrapper,
            embed_model_checksum,
        )

        tmr = generate_tmr_wrapper("sc_lif", data_width=16)
        result = embed_model_checksum(
            tmr,
            equations={"v": "a + b"},
            params={"tmr": True},
        )
        assert "sc_lif_tmr" in result
        assert "MODEL_HASH" in result

    def test_pipeline_plus_power_domain(self):
        """Pipeline wrapper output feeds power-domain wrapper input."""
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
            generate_power_domain_wrapper,
        )

        pipe = generate_pipeline_wrapper(
            "sc_lif",
            {"v": "a * b"},
            stages=2,
        )
        pg = generate_power_domain_wrapper("sc_lif_pipe")
        assert "sc_lif_pipe" in pipe
        assert "sc_lif_pipe_pg" in pg

    def test_mzi_then_noise(self):
        """Encode weights for photonic, then inject noise for robustness."""
        from sc_neurocore.compiler.intelligence import (
            encode_mzi_weights,
            inject_weight_noise,
        )

        weights = [[1.0, -0.5], [0.3, 0.8]]
        enc = encode_mzi_weights(weights)
        noisy = inject_weight_noise(weights, seed=42)
        enc_noisy = encode_mzi_weights(noisy)
        # Noisy encoding should differ from clean
        assert enc.phases_theta != enc_noisy.phases_theta

    def test_quant_sweep_then_compare(self):
        """Sweep quantisation, then compare top 2 widths on 2 targets."""
        from sc_neurocore.compiler.intelligence import (
            auto_quantisation_sweep,
            compare_targets,
        )

        sweep = auto_quantisation_sweep({"v": "a * b + c"}, widths=[8, 16])
        assert len(sweep) == 2
        cmp = compare_targets({"v": "a * b + c"}, ["artix7", "loihi2"])
        assert len(cmp) == 2
        # Both should have valid data
        assert sweep[0].data_width < sweep[1].data_width
        assert cmp[0].target != cmp[1].target

    def test_full_compilation_pipeline(self):
        """Full pipeline: compile → summary → checksum → encrypt."""
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
            embed_model_checksum,
            generate_bitstream_encryption,
        )

        eqs = {"v": "0.04 * v * v + 5 * v + 140 - u + I", "u": "a * (b * v - u)"}
        # Step 1: Summary
        summary = generate_compilation_summary("sc_izh", eqs, "artix7")
        assert "sc_izh" in summary
        # Step 2: Checksum
        verilog = "module sc_izh(...);\nendmodule"
        hashed = embed_model_checksum(verilog, equations=eqs)
        assert "MODEL_HASH" in hashed
        # Step 3: Encryption
        enc = generate_bitstream_encryption("sc_izh")
        assert "ENCRYPT" in enc
