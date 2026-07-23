# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestJuliaWithoutJuliacallInstalled from former test_dispatcher_error_paths.py

"""Focused suite: TestJuliaWithoutJuliacallInstalled from former test_dispatcher_error_paths.py."""

from __future__ import annotations

from tests.dispatcher_error_paths_support import *  # noqa: F403

class TestJuliaWithoutJuliacallInstalled:
    """When juliacall is not installed, calling the dispatchers raises
    ImportError with the install-extras hint."""

    def test_jansen_rit_without_juliacall(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sc_neurocore.accel.julia.neurons as mod

        monkeypatch.setattr(mod, "_jl", None)
        monkeypatch.setattr(mod, "_JANSEN_RIT_LOADED", False)
        with pytest.raises(ImportError, match="juliacall not available"):
            mod._ensure_jansen_rit_loaded()

    def test_wong_wang_without_juliacall(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sc_neurocore.accel.julia.neurons as mod

        monkeypatch.setattr(mod, "_jl", None)
        monkeypatch.setattr(mod, "_WONG_WANG_LOADED", False)
        with pytest.raises(ImportError, match="juliacall not available"):
            mod._ensure_wong_wang_loaded()

    def test_wilson_cowan_without_juliacall(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sc_neurocore.accel.julia.neurons as mod

        monkeypatch.setattr(mod, "_jl", None)
        monkeypatch.setattr(mod, "_WILSON_COWAN_LOADED", False)
        with pytest.raises(ImportError, match="juliacall not available"):
            mod._ensure_wilson_cowan_loaded()
