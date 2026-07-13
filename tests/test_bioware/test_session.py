# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware closed-loop session tests

"""Tests for closed-loop session orchestration and result contracts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.bioware.bioware import (
    AERToSCConverter,
    ArtifactRejector,
    BioHybridFrameResult,
    BioHybridSession,
    LatencyBudget,
    MEAConfig,
    MEAToAERTranscoder,
    PharmModel,
    SCToOptoEncoder,
    SpikeDetector,
    SpikeSorter,
)


FloatArray = npt.NDArray[np.float64]


def _synth_voltage(
    n_samples: int = 1000,
    n_channels: int = 10,
    seed: int = 42,
) -> FloatArray:
    """Generate synthetic MEA voltage data with embedded spikes."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 5, size=(n_samples, n_channels))
    for i in range(0, n_samples, 200):
        data[i, 0] = -80.0
        if i + 50 < n_samples:
            data[i + 50, 3] = -60.0
    return data


class TestBioHybridSession:
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
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert result["round"] == 1
        assert result["num_spikes"] > 0

    def test_full_pipeline(self) -> None:
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "aer_events" in result
        assert "bitstreams" in result
        assert "opto_pulses" in result
        assert "health" in result

    def test_multiple_rounds(self) -> None:
        session = self._make_session()
        for i in range(3):
            data = _synth_voltage(n_channels=10, seed=42 + i)
            session.process_frame(data)
        assert session.round_count == 3

    def test_health_in_result(self) -> None:
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "health_score" in result["health"]
        assert "is_viable" in result["health"]

    def test_latency_measured(self) -> None:
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "latency_us" in result
        assert result["latency_us"] > 0

    def test_process_frame_runs_all_optional_stages(self) -> None:
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


# ── Refractory Period Tests ──────────────────────────────────────────


class TestBioHybridFrameResult:
    """The packet returned by ``BioHybridSession.process_frame`` must be
    both a typed dataclass (new callers) and a read-only mapping view
    (legacy callers that did ``result["round"]``). Both surfaces carry
    identical data; the mapping wraps the dataclass, not a shadow dict.
    """

    def _make(self, **overrides: Any) -> BioHybridFrameResult:
        base: dict[str, Any] = dict(
            round=3,
            num_spikes=0,
            num_aer_events=0,
            num_bitstreams=0,
            num_opto_pulses=0,
            latency_us=1234.5,
            health={"score": 0.95},
            spikes=[],
            aer_events=[],
            bitstreams={},
            opto_pulses=[],
        )
        base.update(overrides)
        return BioHybridFrameResult(**base)

    def test_attribute_access(self) -> None:
        r = self._make()
        assert r.round == 3
        assert r.latency_us == pytest.approx(1234.5)
        assert r.health["score"] == pytest.approx(0.95)

    def test_dict_subscript_matches_attribute(self) -> None:
        r = self._make()
        assert r["round"] == r.round
        assert r["latency_us"] == r.latency_us
        assert r["health"] is r.health  # same object, not a copy

    def test_contains_reports_field_names(self) -> None:
        r = self._make()
        assert "round" in r
        assert "latency_us" in r
        assert "not_a_field" not in r
        assert 42 not in r  # non-string keys are not fields

    def test_unknown_key_raises_keyerror(self) -> None:
        r = self._make()
        with pytest.raises(KeyError, match="nope"):
            _ = r["nope"]

    def test_private_attribute_hidden_from_mapping(self) -> None:
        # Mapping view must not leak Python dunder / private names.
        r = self._make()
        with pytest.raises(KeyError):
            _ = r["__class__"]

    def test_keys_returns_declared_fields(self) -> None:
        r = self._make()
        assert set(r.keys()) == {
            "round",
            "num_spikes",
            "num_aer_events",
            "num_bitstreams",
            "num_opto_pulses",
            "latency_us",
            "health",
            "spikes",
            "aer_events",
            "bitstreams",
            "opto_pulses",
        }


# ── mea_fitness_hook — evo_substrate fitness adaptor ───────────────────
