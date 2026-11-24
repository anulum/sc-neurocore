# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio analysis codegen

"""Focused suite: the exported script states the experiment, not a guess."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sc_neurocore.studio.experiment_spec import ExperimentSpec

from sc_neurocore.studio.codegen import generate_experiment_script
from sc_neurocore.studio.experiment_spec import resolve_experiment
from sc_neurocore.studio.replay_pack import pinned_request

import pytest

pytest.importorskip("fastapi")


class TestCodegen:
    """Experiment scripts generated from a resolved request."""

    def _script(self, request: dict[str, Any]) -> tuple[ExperimentSpec, str]:
        spec = resolve_experiment(request)
        return spec, generate_experiment_script(spec, pinned_request(request, spec))

    def test_the_script_carries_the_parameter_override(self) -> None:
        """The script carries the model, the parameter override and the experiment hash."""
        spec, script = self._script(
            {"name": "COBALIFNeuron", "params": {"c_m": 200.0}, "duration": 100.0, "current": 10.0}
        )
        assert "'name': 'COBALIFNeuron'" in script
        assert "'c_m': 200.0" in script
        assert spec.experiment_sha256 in script

    def test_the_script_makes_no_assumption_about_the_step_signature(self) -> None:
        """The script resolves the request instead of calling the neuron directly."""
        _spec, script = self._script({"name": "COBALIFNeuron"})
        # These were the assumptions the previous export hard-coded.
        assert "step(current=" not in script
        assert "neuron.v" not in script
        assert "resolve_experiment(REQUEST)" in script

    def test_the_script_refuses_a_drifted_installation(self) -> None:
        """The script exits when the installation resolves a different experiment."""
        _spec, script = self._script({"name": "COBALIFNeuron"})
        assert "resolves a different experiment" in script
        assert "raise SystemExit" in script
