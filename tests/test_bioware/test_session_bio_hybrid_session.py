# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBioHybridSession from former test_session.py

"""Focused suite: TestBioHybridSession from former test_session.py."""

from __future__ import annotations

from sc_neurocore.bioware.bioware import (
    AERToSCConverter,
    ArtifactRejector,
    BioHybridSession,
    LatencyBudget,
    MEAConfig,
    MEAToAERTranscoder,
    PharmModel,
    SCToOptoEncoder,
    SpikeDetector,
    SpikeSorter,
)
from tests.test_bioware.session_support import (
    _synth_voltage,
)
from typing import (
    Any,
    cast,
)


class TestBioHybridSession:
    """Running a closed-loop bio-hybrid session."""

    def _make_session(self) -> BioHybridSession:
        cfg = MEAConfig(num_channels=10)
        det = SpikeDetector(config=cfg)
        tc = MEAToAERTranscoder(hw_clock_hz=1e6)
        sc = AERToSCConverter(bitstream_length=128, num_neurons=10)
        opto = SCToOptoEncoder()
        return BioHybridSession(
            mea_config=cfg,
            detector=det,
            transcoder=tc,
            sc_converter=sc,
            opto_encoder=opto,
        )

    def test_process_frame(self) -> None:
        """A frame gives round 1 with spikes."""
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert result["round"] == 1
        assert result["num_spikes"] > 0

    def test_full_pipeline(self) -> None:
        """A frame yields events, bitstreams and light pulses."""
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "aer_events" in result
        assert "bitstreams" in result
        assert "opto_pulses" in result
        assert "health" in result

    def test_multiple_rounds(self) -> None:
        """Three frames are three rounds."""
        session = self._make_session()
        for i in range(3):
            data = _synth_voltage(n_channels=10, seed=42 + i)
            session.process_frame(data)
        assert session.round_count == 3

    def test_health_in_result(self) -> None:
        """A frame reports the culture's health."""
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "health_score" in result["health"]
        assert "is_viable" in result["health"]

    def test_latency_measured(self) -> None:
        """A frame reports a positive latency."""
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "latency_us" in result
        assert result["latency_us"] > 0

    def test_process_frame_runs_all_optional_stages(self) -> None:
        """Every optional stage runs on a frame."""
        cfg = MEAConfig(num_channels=10)
        captured: dict[str, object] = {}

        class _ZenithStub:
            def step_from_bio_rates(self, rates: dict[int, float]) -> None:
                captured["rates"] = rates

        session = BioHybridSession(
            mea_config=cfg,
            detector=SpikeDetector(config=cfg),
            transcoder=MEAToAERTranscoder(hw_clock_hz=1e6),
            sc_converter=AERToSCConverter(bitstream_length=128, num_neurons=10),
            opto_encoder=SCToOptoEncoder(),
            artifact_rejector=ArtifactRejector(),
            sorter=SpikeSorter(num_units=3),
            pharm_model=PharmModel(),
            latency_budget=LatencyBudget(),
            zenith_core=cast(Any, _ZenithStub()),
        )
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data, stim_times_s=[0.001])
        assert result["round"] == 1
        assert "rates" in captured  # the zenith stage received decoded rates
