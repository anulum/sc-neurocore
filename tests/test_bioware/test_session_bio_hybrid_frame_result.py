# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBioHybridFrameResult from former test_session.py

"""Focused suite: TestBioHybridFrameResult from former test_session.py."""

from __future__ import annotations

from tests.test_bioware.session_support import *  # noqa: F403


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
