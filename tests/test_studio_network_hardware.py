# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the Studio network's hardware lowering and co-simulation

"""The hardware computes the network the Studio simulates, or the lowering says why not.

The co-simulation cases compile real RTL and run it in Icarus Verilog beside a
C model built by the compiler's bit-true generator and beside the Studio's own
double-precision run.
"""

from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sc_neurocore.studio.network_graph_spec import GraphRejected
from sc_neurocore.studio.network_hardware import (
    NETWORK_HARDWARE_SCHEMA_VERSION,
    HardwareLoweringRefused,
    lower_graph,
    lowering_public_dict,
)
from sc_neurocore.studio.network_hardware_cosim import (
    NETWORK_COSIM_SCHEMA_VERSION,
    HardwareCosimUnavailable,
    cosimulate,
)

_TOOLS = all(shutil.which(tool) for tool in ("iverilog", "vvp", "gcc"))
needs_tools = pytest.mark.skipif(not _TOOLS, reason="iverilog, vvp and gcc are required")


def _population(pop_id: str, model: str, count: int, params: dict[str, Any], **fields: Any) -> dict:
    population = {
        "id": pop_id,
        "type": "population",
        "label": pop_id,
        "model": model,
        "count": count,
        "neuron_type": "excitatory",
        "params": params,
        "drive": {"kind": "none"},
        "position": {"x": 0, "y": 0},
    }
    population.update(fields)
    return population


def _integrators(delay: float = 0.0) -> dict[str, Any]:
    return {
        "populations": [
            _population(
                "src",
                "PerfectIntegratorNeuron",
                3,
                {"c_m": 1.0, "v_threshold": 1.0},
                drive={"kind": "constant", "current": 0.375},
            ),
            _population("dst", "PerfectIntegratorNeuron", 2, {"c_m": 1.0, "v_threshold": 1.0}),
        ],
        "projections": [
            {
                "id": "p",
                "source": "src",
                "target": "dst",
                "weight": 0.5,
                "delay": delay,
                "rule": "all_to_all",
            }
        ],
        "dt": 1.0,
        "duration": 30.0,
        "seed": 5,
    }


def _leaky() -> dict[str, Any]:
    params = {"tau": 10.0, "resistance": 1.0, "v_threshold": 1.0}
    return {
        "populations": [
            _population(
                "a",
                "SCLapicqueLIFNeuron",
                4,
                params,
                drive={"kind": "constant", "current": 1.5},
            ),
            _population("b", "SCLapicqueLIFNeuron", 3, dict(params)),
        ],
        "projections": [
            {
                "id": "ab",
                "source": "a",
                "target": "b",
                "weight": 0.75,
                "delay": 1.0,
                "rule": "random",
                "probability": 0.6,
                "seed": 3,
            }
        ],
        "dt": 1.0,
        "duration": 40.0,
        "seed": 7,
    }


class TestLowering:
    def test_each_model_keeps_its_own_step_and_the_drive_is_an_input(self) -> None:
        graph = _leaky()
        graph["populations"].append(
            _population("acc", "PerfectIntegratorNeuron", 2, {"c_m": 2.0, "v_threshold": 1.5})
        )
        lowered = lower_graph(graph)

        templates = {pop.name: pop.neuron_type for pop in lowered.graph.populations}
        assert templates == {"a": "sc_lif", "b": "sc_lif", "acc": "sc_if"}
        a = lowered.graph.populations[0]
        decay = math.exp(-1.0 / 10.0)
        assert a.params["decay"][0] == decay
        assert a.params["gain"][0] == 1.0 - decay
        assert lowered.graph.populations[2].params["r"][0] == 0.5
        assert lowered.drive_lanes == (("drive_a", (1.5, 1.5, 1.5, 1.5)),)
        drive, projection = lowered.graph.connections
        assert (drive.src, drive.dst) == ("drive_a", "a")
        np.testing.assert_array_equal(drive.weights, np.eye(4))
        assert (projection.src, projection.dst, projection.delay_steps) == ("a", "b", 1)
        assert projection.weights.shape == (3, 4)
        assert set(np.unique(projection.weights)) <= {0.0, 0.75}
        assert lowered.graph.input_pop == "drive_a"

    def test_an_undriven_network_takes_its_first_population_as_input(self) -> None:
        graph = _integrators()
        graph["populations"][0]["drive"] = {"kind": "none"}
        lowered = lower_graph(graph)
        assert lowered.drive_lanes == ()
        assert lowered.graph.input_pop == "src"

    def test_the_input_digest_follows_the_graph_and_the_format(self) -> None:
        first = lower_graph(_integrators()).input_sha256()
        assert lower_graph(_integrators()).input_sha256() == first
        assert lower_graph(_integrators(delay=2.0)).input_sha256() != first
        assert lower_graph(_integrators(), data_width=32, fraction=16).input_sha256() != first

    def test_rounded_values_are_noted_and_exact_ones_are_not(self) -> None:
        assert lower_graph(_integrators()).notes == ()
        notes = lower_graph(_leaky()).notes
        assert any(
            note.startswith("population a: decay = ") and "held as 0.90625" in note
            for note in notes
        )

    def test_the_public_projection_names_what_was_compiled(self) -> None:
        lowered = lower_graph(_leaky(), data_width=32, fraction=16)
        public = lowering_public_dict(lowered)
        assert public["schema_version"] == NETWORK_HARDWARE_SCHEMA_VERSION
        assert public["input_sha256"] == lowered.input_sha256()
        assert public["q_format"] == "Q16.16"
        assert public["populations"] == [
            {"id": "a", "template": "sc_lif", "count": 4},
            {"id": "b", "template": "sc_lif", "count": 3},
        ]
        assert public["drive_lanes"] == [{"source": "drive_a", "currents": [1.5] * 4}]

    def test_everything_it_cannot_reproduce_is_refused_at_once(self) -> None:
        graph = _integrators()
        graph["populations"] += [
            _population("adex", "AdExNeuron", 1, {}),
            _population("qif", "QuadraticIFNeuron", 1, {}),
            _population(
                "noisy",
                "PerfectIntegratorNeuron",
                1,
                {"c_m": 1.0, "v_threshold": 1.0},
                drive={"kind": "poisson", "rate_hz": 20.0, "weight": 0.5},
            ),
            _population("charged", "SCLapicqueLIFNeuron", 1, {"v": 0.25}),
            _population("charged2", "SCLapicqueLIFNeuron", 1, {"tau": 5.0}),
            _population("big", "PerfectIntegratorNeuron", 1, {"c_m": 1.0, "v_threshold": 500.0}),
        ]
        graph["populations"][0]["drive"] = {"kind": "constant", "current": 1000.0}
        graph["projections"] += [
            {
                "id": "faint",
                "source": "src",
                "target": "dst",
                "weight": 0.0001,
                "rule": "all_to_all",
            },
            {
                "id": "late",
                "source": "src",
                "target": "dst",
                "weight": 0.5,
                "delay": 1100.0,
                "rule": "all_to_all",
            },
        ]
        graph["duration"] = 1200.0

        with pytest.raises(HardwareLoweringRefused) as refused:
            lower_graph(graph)

        reasons = "\n".join(refused.value.reasons)
        assert "population adex: model AdExNeuron has no hardware lowering" in reasons
        assert "model QuadraticIFNeuron (profile sc_symmetric) has no hardware lowering" in reasons
        assert "population noisy: a Poisson drive has no hardware source" in reasons
        assert "population charged: its membrane starts at 0.25, not at rest (0.0)" in reasons
        assert (
            "populations charged and charged2 are both sc_lif with different parameters" in reasons
        )
        assert "population big: v_threshold = 500.0 is outside the Q8.8 range" in reasons
        assert "population src drive: current = 1000.0 is outside the Q8.8 range" in reasons
        assert "projection faint: weight = 0.0001 quantises to zero in Q8.8" in reasons
        assert "projection late: 1100 delay steps exceed the synthesis guard of 1024" in reasons
        assert "populations src and big" in reasons  # the same template, different threshold

    def test_a_graph_that_does_not_validate_is_rejected_first(self) -> None:
        with pytest.raises(GraphRejected):
            lower_graph({"populations": []})


@needs_tools
class TestCosimulation:
    @pytest.mark.parametrize("delay", [0.0, 2.0])
    def test_integrators_spike_identically_in_rtl_model_and_studio(
        self, tmp_path: Path, delay: float
    ) -> None:
        """Exact values: the fixed-point hardware reproduces the Studio run itself."""
        graph = _integrators(delay)
        cosim = cosimulate(lower_graph(graph), graph, tmp_path)

        assert cosim.rtl_matches_model
        assert cosim.studio_first_divergence is None
        assert cosim.steps == 30
        assert len(cosim.rtl_raster) == 30
        assert sum(row.count("1") for row in cosim.rtl_raster) > 10
        receipt = cosim.to_public_dict()
        assert receipt["schema_version"] == NETWORK_COSIM_SCHEMA_VERSION
        assert receipt["rtl_matches_bit_true_model"] is True
        assert receipt["studio_agreement"] == {"identical": True, "first_divergent_step": None}

    def test_the_delay_moves_the_target_by_its_steps(self, tmp_path: Path) -> None:
        direct = cosimulate(
            lower_graph(_integrators(0.0)), _integrators(0.0), tmp_path / "d0", steps=12
        )
        delayed = cosimulate(
            lower_graph(_integrators(2.0)), _integrators(2.0), tmp_path / "d2", steps=12
        )
        first_direct = next(i for i, row in enumerate(direct.rtl_raster) if row[0] == "1")
        first_delayed = next(i for i, row in enumerate(delayed.rtl_raster) if row[0] == "1")
        assert first_delayed - first_direct == 2

    def test_rounding_shows_as_a_reported_divergence_not_a_model_mismatch(
        self, tmp_path: Path
    ) -> None:
        graph = _leaky()
        q88 = cosimulate(lower_graph(graph), graph, tmp_path / "q88")
        q1616 = cosimulate(
            lower_graph(graph, data_width=32, fraction=16), graph, tmp_path / "q1616"
        )

        assert q88.rtl_matches_model and q1616.rtl_matches_model
        assert q88.studio_first_divergence is not None
        assert q88.to_public_dict()["studio_agreement"]["identical"] is False
        assert q1616.studio_first_divergence is None

    def test_different_networks_give_different_rtl(self, tmp_path: Path) -> None:
        first = cosimulate(lower_graph(_integrators()), _integrators(), tmp_path / "a", steps=4)
        graph = copy.deepcopy(_integrators())
        graph["projections"][0]["weight"] = 0.25
        second = cosimulate(lower_graph(graph), graph, tmp_path / "b", steps=4)
        assert first.rtl_sha256 != second.rtl_sha256
        assert first.model_sha256 != second.model_sha256
        assert first.input_sha256 != second.input_sha256


def test_without_the_simulators_the_cosimulation_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty PATH is an installation without Icarus Verilog or a C compiler."""
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(HardwareCosimUnavailable, match="needs iverilog, vvp, gcc"):
        cosimulate(lower_graph(_integrators()), _integrators(), tmp_path)
