# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware cross-component edge tests

"""Cross-component regression tests for boundary behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.bioware.bioware import (
    AEREvent,
    AERToSCConverter,
    BCMPlasticity,
    BiologicalSTDP,
    CultureHealth,
    MEAConfig,
    SCToOptoEncoder,
    SpikeDetector,
)


class TestEdgeCases:
    def test_single_sample_data(self) -> None:
        cfg = MEAConfig(num_channels=5)
        det = SpikeDetector(config=cfg)
        data = np.zeros((1, 5))
        spikes = det.detect(data)
        assert len(spikes) == 0

    def test_all_silent_neurons_no_pulses(self) -> None:
        bs = {0: np.zeros(100, dtype=np.uint8)}
        enc = SCToOptoEncoder()
        pulses = enc.encode(bs)
        assert len(pulses) == 0

    def test_stdp_symmetry(self) -> None:
        stdp = BiologicalSTDP(tau_plus_ms=20.0, tau_minus_ms=20.0, a_plus=0.01, a_minus=0.01)
        dw_pos = stdp.compute_dw(5.0)
        dw_neg = stdp.compute_dw(-5.0)
        assert abs(dw_pos + dw_neg) < 1e-10

    def test_bcm_zero_rate(self) -> None:
        bcm = BCMPlasticity()
        dw = bcm.compute_dw(0.0, 0.0)
        assert dw == 0.0

    def test_culture_health_zero_duration(self) -> None:
        ch = CultureHealth()
        counts = np.array([5, 10])
        with pytest.raises(ValueError, match="duration_s must be > 0"):
            ch.assess(counts, duration_s=0.0)

    def test_aer_to_sc_invalid_event(self) -> None:
        events = [AEREvent(neuron_id=0, timestamp=100, valid=False)]
        conv = AERToSCConverter()
        bs = conv.convert(events)
        assert len(bs) == 0


# ── Spike Sorter Tests (Gap 1) ─────────────────────────────────────────
