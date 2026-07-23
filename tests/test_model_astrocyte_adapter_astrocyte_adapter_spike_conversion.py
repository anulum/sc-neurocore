# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocyteAdapterSpikeConversion from former test_model_astrocyte_adapter.py

"""Focused suite: TestAstrocyteAdapterSpikeConversion from former test_model_astrocyte_adapter.py."""

from __future__ import annotations

from tests.model_astrocyte_adapter_support import *  # noqa: F403

class TestAstrocyteAdapterSpikeConversion:
    """Tests for calcium-to-spike conversion semantics."""

    def test_fires_when_ca_above_threshold(self) -> None:
        """Spike = 1 when Ca > ca_threshold."""
        n = AstrocyteNeuron(ca_threshold=0.3)
        spikes_no_input = sum(n.step(0.0) for _ in range(10000))
        # Ca oscillates to 0.94 at I=0 → crosses 0.3 → spikes
        assert spikes_no_input > 0

    def test_ip3_input_drives_sustained_activity(self) -> None:
        """Sustained IP3 input keeps Ca high and fires almost every step."""
        n = AstrocyteNeuron()
        spikes = sum(n.step(0.5) for _ in range(10000))
        assert spikes > 9000, f"Only {spikes} spikes at I=0.5"

    def test_lower_threshold_more_spikes(self) -> None:
        """Lower thresholds produce more release events than high thresholds."""
        n_low = AstrocyteNeuron(ca_threshold=0.1)
        n_high = AstrocyteNeuron(ca_threshold=0.8)
        s_low = sum(n_low.step(0.0) for _ in range(10000))
        s_high = sum(n_high.step(0.0) for _ in range(10000))
        assert s_low > s_high

    def test_zero_input_oscillatory_spiking(self) -> None:
        """At I=0, Ca oscillates → intermittent spikes (not every step)."""
        n = AstrocyteNeuron(ca_threshold=0.3)
        outputs = [n.step(0.0) for _ in range(10000)]
        spikes = outputs.count(1)
        assert 100 < spikes < 9000, f"{spikes} — expected intermittent"
