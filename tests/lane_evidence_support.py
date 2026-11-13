# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Test fixture that writes a run's lane evidence into the report

"""Write the lane a model ran, and the libraries it loaded, into the test's report.

A test module registers this module as a plugin, as the repository's other
shared fixtures are registered (``pytest_plugins = ("tests.lane_evidence_support",)``),
asks for :func:`lane_evidence` and calls it with the model after the run it
checks; importing the fixture into the module instead would make the linter
report an unused import and its redefinition by the test's parameter. The
fixture reads the model's lane record and appends one ``sc_neurocore.lane``
property and one ``sc_neurocore.native_binary`` property per library
(``<file name>=<sha256>``) to the running test. It appends to the test's user
properties directly, so that pytest writes them under its default report family
without the warning its ``record_property`` fixture gives there.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from sc_neurocore.neurons.evidence_references import sha256_file
from sc_neurocore.neurons.lane_runs import LaneRun, lane_run
from sc_neurocore.neurons.receipt_execution import LANE_PROPERTY, NATIVE_BINARY_PROPERTY

__all__ = ["lane_evidence"]


@pytest.fixture
def lane_evidence(request: pytest.FixtureRequest) -> Callable[[object], LaneRun]:
    """Return a function that writes a model's last lane run into the running test's report.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The running test, whose user properties receive the evidence.

    Returns
    -------
    callable
        Takes the model after its run and returns the record it wrote. It fails the
        test when the model has no recorded run.
    """

    def record(model: object) -> LaneRun:
        run = lane_run(model)
        assert run is not None, "the model has no recorded lane run"
        request.node.user_properties.append((LANE_PROPERTY, run.lane))
        for library in run.libraries:
            request.node.user_properties.append(
                (NATIVE_BINARY_PROPERTY, f"{library.name}={sha256_file(library)}")
            )
        return run

    return record
