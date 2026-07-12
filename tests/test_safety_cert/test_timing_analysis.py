# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

"""Focused tests for timing analysis."""

from typing import Any

import pytest

from sc_neurocore.safety_cert.safety_cert import (
    WCETAnalyzer,
    WCETPath,
)


def _unsafe(value: object) -> Any:
    """Return a deliberately invalid runtime value for boundary tests."""
    return value


class TestWCETAnalyzer:
    def test_basic_analysis(self) -> None:
        path = WCETAnalyzer.analyze(256, 8, 16)
        assert path.total_cycles > 0
        assert len(path.stages) == 4

    def test_wcet_ns(self) -> None:
        path = WCETAnalyzer.analyze(256, 8, 16)
        ns = path.wcet_ns(100.0)
        assert ns > 0

    def test_total_cycles_rejects_corrupted_internal_state(self) -> None:
        path = WCETPath("p1", "path", ["A"], [1])
        path.cycles_per_stage = _unsafe(["bad"])
        with pytest.raises(ValueError, match="cycles_per_stage"):
            _ = path.total_cycles

    def test_with_stp(self) -> None:
        path = WCETAnalyzer.analyze(256, 8, 16, has_stp=True)
        assert len(path.stages) == 5
        assert "STP_Update" in path.stages

    def test_scaling(self) -> None:
        small = WCETAnalyzer.analyze(128, 4, 8)
        large = WCETAnalyzer.analyze(1024, 64, 128)
        assert large.total_cycles > small.total_cycles

    def test_multistage(self) -> None:
        layers = [
            {"bitstream_length": 256, "num_inputs": 8, "num_neurons": 16},
            {"bitstream_length": 256, "num_inputs": 16, "num_neurons": 4},
        ]
        path = WCETAnalyzer.analyze_multistage(layers)
        assert len(path.stages) == 8
        assert path.total_cycles > 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"path_id": ""}, "path_id"),
            ({"description": ""}, "description"),
            ({"stages": [], "cycles_per_stage": []}, "stages must not be empty"),
            ({"stages": ["A", ""], "cycles_per_stage": [1, 2]}, "stages"),
            ({"stages": ["A"], "cycles_per_stage": [1, 2]}, "same length"),
            ({"cycles_per_stage": [1, -1]}, "cycles_per_stage"),
        ],
    )
    def test_wcet_path_rejects_invalid_contracts(self, kwargs: Any, match: Any) -> None:
        values = {
            "path_id": "p1",
            "description": "path",
            "stages": ["A", "B"],
            "cycles_per_stage": [1, 2],
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            WCETPath(**_unsafe(values))

    @pytest.mark.parametrize("clock_mhz", [0.0, -1.0, float("inf"), float("nan"), True])
    def test_wcet_ns_rejects_invalid_clock(self, clock_mhz: Any) -> None:
        path = WCETPath("p1", "path", ["A"], [1])
        with pytest.raises(ValueError, match="clock_mhz"):
            path.wcet_ns(clock_mhz)

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((0, 8, 16, False), "bitstream_length"),
            ((256, 0, 16, False), "num_inputs"),
            ((256, 8, 0, False), "num_neurons"),
            ((256, 8, 16, "yes"), "has_stp"),
        ],
    )
    def test_analyze_rejects_invalid_contracts(self, args: Any, match: Any) -> None:
        with pytest.raises(ValueError, match=match):
            WCETAnalyzer.analyze(*args)

    @pytest.mark.parametrize(
        ("layers", "match"),
        [
            ([], "non-empty list"),
            ([None], "dictionary"),
            ([{"bitstream_length": 0}], "bitstream_length"),
            ([{"num_inputs": 0}], "num_inputs"),
            ([{"num_neurons": 0}], "num_neurons"),
        ],
    )
    def test_analyze_multistage_rejects_invalid_contracts(self, layers: Any, match: Any) -> None:
        with pytest.raises(ValueError, match=match):
            WCETAnalyzer.analyze_multistage(layers)

    def test_wcet_path_rejects_non_list_stage_containers(self) -> None:
        with pytest.raises(ValueError, match="stages must be a list"):
            WCETPath("path", "description", _unsafe("stage"), [1])
        with pytest.raises(ValueError, match="cycles_per_stage must be a list"):
            WCETPath("path", "description", ["stage"], _unsafe((1,)))
