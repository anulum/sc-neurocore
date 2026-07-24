# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConfig from former test_adc_to_spike_kernel.py

"""Focused suite: TestConfig from former test_adc_to_spike_kernel.py."""

from __future__ import annotations

from tests.adc_to_spike_kernel_support import *  # noqa: F403


class TestConfig:
    """Config invariants and validation."""

    def test_q_bounds(self) -> None:
        """Q-format bounds derive from the configured integer and fractional width."""
        cfg = ADCSpikeWindowConfig(q_int=8, q_frac=8)
        assert cfg.q_total == 16
        assert cfg.q_min == -32768
        assert cfg.q_max == 32767

    @pytest.mark.parametrize(
        ("config_factory", "match"),
        [
            (lambda: ADCSpikeWindowConfig(adc_width=1), "adc_width"),
            (lambda: ADCSpikeWindowConfig(q_int=0), "Q-format"),
            (lambda: ADCSpikeWindowConfig(q_frac=-1), "Q-format"),
            (lambda: ADCSpikeWindowConfig(decimation=0), "decimation"),
            (lambda: ADCSpikeWindowConfig(threshold_q=0), "threshold_q"),
        ],
    )
    def test_validate_rejects(
        self, config_factory: Callable[[], ADCSpikeWindowConfig], match: str
    ) -> None:
        """Invalid scalar contracts fail before any sample stream is consumed."""
        with pytest.raises(ValueError, match=match):
            config_factory().validate()
