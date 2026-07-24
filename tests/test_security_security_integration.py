# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSecurityIntegration from former test_security.py

"""Focused suite: TestSecurityIntegration from former test_security.py."""

from __future__ import annotations

from tests.security_support import *  # noqa: F403


class TestSecurityIntegration:
    """Integration tests combining multiple security modules."""

    def test_ethics_and_immune_combined(self):
        """Test ethical action filtering with immune system monitoring."""
        governor = AsimovGovernor()
        immune = DigitalImmuneSystem(tolerance=0.3)

        # Train immune system on "normal" decision patterns
        normal_pattern = np.array([1.0, 0.0, 0.0])  # PASS state
        immune.train_self(normal_pattern)

        # Safe action should pass both
        safe_action = ActionRequest(1, "HEAL", "HUMAN", "SAFE")
        ethics_ok = governor.check_laws(safe_action)
        immune_ok = immune.scan(normal_pattern)

        assert ethics_ok and immune_ok

    def test_watermark_survives_zkp_verification(self):
        """Test that watermarked weights can be ZKP-committed."""
        layer = MockLayer(n_neurons=5, n_inputs=8)
        trigger = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        # Inject watermark
        WatermarkInjector.inject_backdoor(layer, trigger, 2)

        # Commit the watermarked weights
        weights_flat = layer.weights.flatten().astype(np.float32)
        commitment = ZKPVerifier.commit(weights_flat.view(np.uint8))

        # Verify commitment is valid
        assert len(commitment) == 64

        # Verify watermark still works
        activation = WatermarkInjector.verify_watermark(layer, trigger, 2)
        assert activation >= 0.5
