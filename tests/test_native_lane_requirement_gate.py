# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - The built-lane gate skips absence and never masks CI

"""An unbuilt lane skips; a lane CI requires fails.

A backend-parametrised test dispatches through a model's accelerator module. On
a checkout where the Go and Mojo shared libraries were never built, forty of
those cases failed with `<lane> backend is unavailable` — a statement about the
toolchain wearing the shape of a defect in the model. The Rust extension and the
Julia bridge already had gates for exactly this; the built lanes had none.

Every case here fails on that former behaviour, and the two that matter most are
the ones that keep the gate honest: with the lane present nothing skips, and
with `SC_NEUROCORE_REQUIRE_<LANE>` set an absent lane is a hard error. A gate
that skipped unconditionally would turn a broken CI build into a green wall.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from sc_neurocore.accel import brunel_wang
from tests.native_lane_requirement import (
    AVAILABILITY_PREDICATE,
    GATED_LANES,
    lane_is_built,
    require_native_lane,
)

CI_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _accel(available: Any) -> ModuleType:
    """Return a stand-in accelerator module with the given availability check."""
    return cast(ModuleType, SimpleNamespace(**{AVAILABILITY_PREDICATE: available}))


class TestLaneIsBuilt:
    def test_it_asks_the_module_s_own_predicate(self) -> None:
        """The gate must not answer differently from the dispatch under test."""
        asked: list[str] = []

        def predicate(lane: str) -> bool:
            asked.append(lane)
            return True

        assert lane_is_built(_accel(predicate), "go") is True
        assert asked == ["go"]

    def test_a_module_without_a_predicate_counts_as_built(self) -> None:
        """The gate has nothing to say there and must not skip on a guess."""
        assert lane_is_built(cast(ModuleType, SimpleNamespace()), "go") is True

    def test_a_predicate_that_raises_counts_as_absent(self) -> None:
        """A probe that cannot answer has not shown the lane is there."""

        def predicate(lane: str) -> bool:
            raise OSError("no library")

        assert lane_is_built(_accel(predicate), "go") is False

    def test_a_real_accelerator_module_answers_for_every_lane(self) -> None:
        """The predicate the gate relies on exists on the shipped modules."""
        for lane in ("python", "rust", "julia", "go", "mojo"):
            assert isinstance(lane_is_built(brunel_wang, lane), bool)


class TestRequireNativeLane:
    def test_an_absent_lane_skips_with_the_subject_named(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reader of the report must know which model went unexercised."""
        monkeypatch.delenv("SC_NEUROCORE_REQUIRE_GO", raising=False)
        with pytest.raises(pytest.skip.Exception, match=r"go lane is not built for probe"):
            require_native_lane(_accel(lambda lane: False), "go", subject="probe")

    def test_a_present_lane_does_not_skip(self) -> None:
        """The gate must cost nothing where the lane exists."""
        require_native_lane(_accel(lambda lane: True), "go", subject="probe")

    @pytest.mark.parametrize("lane", ["python", "rust", "julia"])
    def test_an_ungated_lane_is_left_alone(self, lane: str) -> None:
        """Rust and Julia have their own gates; python is always present."""
        require_native_lane(_accel(lambda name: False), lane, subject="probe")

    @pytest.mark.parametrize("lane", GATED_LANES)
    def test_a_required_lane_that_is_absent_is_a_hard_error(
        self, lane: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This is what stops the gate turning a broken CI build into skips."""
        monkeypatch.setenv(f"SC_NEUROCORE_REQUIRE_{lane.upper()}", "1")
        with pytest.raises(RuntimeError, match=f"the {lane} lane is required here"):
            require_native_lane(_accel(lambda name: False), lane, subject="probe")

    @pytest.mark.parametrize("lane", GATED_LANES)
    def test_a_required_lane_that_is_present_passes(
        self, lane: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requiring a lane that is there changes nothing."""
        monkeypatch.setenv(f"SC_NEUROCORE_REQUIRE_{lane.upper()}", "1")
        require_native_lane(_accel(lambda name: True), lane, subject="probe")

    def test_a_switch_set_to_anything_else_does_not_require(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the exact value CI sets turns absence into an error."""
        monkeypatch.setenv("SC_NEUROCORE_REQUIRE_GO", "0")
        with pytest.raises(pytest.skip.Exception):
            require_native_lane(_accel(lambda lane: False), "go", subject="probe")


class TestCIRequiresEveryGatedLane:
    def test_the_workflow_requires_each_gated_lane(self) -> None:
        """A gate CI does not require is a gate that can hide a broken build."""
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        for lane in GATED_LANES:
            assert f'SC_NEUROCORE_REQUIRE_{lane.upper()}: "1"' in workflow

    def test_the_workflow_still_requires_the_engine_and_the_bridge(self) -> None:
        """The two gates that came before must not be lost while adding these."""
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        assert 'SC_NEUROCORE_REQUIRE_ENGINE: "1"' in workflow
        assert 'SC_NEUROCORE_REQUIRE_JULIA: "1"' in workflow


class TestTheFixtureReachesTheSuites:
    def test_a_backend_parametrised_suite_exposes_its_accelerator_module(self) -> None:
        """The conftest fixture finds the module by that name; keep it findable."""
        suite = importlib.import_module("tests.test_brunel_wang_backends")

        accel = getattr(suite, "backends", None)
        assert isinstance(accel, ModuleType)
        assert hasattr(accel, AVAILABILITY_PREDICATE)
