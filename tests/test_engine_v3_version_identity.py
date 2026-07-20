# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — v3 engine version identity

"""Version-identity checks shared by the v3 engine release phases."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import sc_neurocore
import sc_neurocore_engine as v3


def _assert_engine_version_matches_core() -> None:
    """Validate source-tree and installed-wheel version surfaces."""

    assert v3.__version__ == sc_neurocore.__version__
    try:
        installed_version = version("sc-neurocore-engine")
    except PackageNotFoundError:
        return
    assert installed_version == sc_neurocore.__version__


class TestPhase10Version:
    def test_version(self) -> None:
        _assert_engine_version_matches_core()


class TestPhase11Version:
    def test_version(self) -> None:
        _assert_engine_version_matches_core()


class TestPhase12Version:
    def test_version(self) -> None:
        _assert_engine_version_matches_core()


class TestPhase8Version:
    def test_version_is_current(self) -> None:
        _assert_engine_version_matches_core()


class TestPhase9Version:
    def test_version_is_current(self) -> None:
        _assert_engine_version_matches_core()
