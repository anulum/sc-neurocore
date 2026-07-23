# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestJuliaMissingKernelFile from former test_dispatcher_error_paths.py

"""Focused suite: TestJuliaMissingKernelFile from former test_dispatcher_error_paths.py."""

from __future__ import annotations

from tests.dispatcher_error_paths_support import *  # noqa: F403

class TestJuliaMissingKernelFile:
    """Julia loader helpers fail closed when a maintained kernel is absent."""

    def test_jansen_rit_missing_jl_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sc_neurocore.accel.julia.neurons as mod

        if not mod._HAS_JULIA_NEURONS:
            pytest.skip("juliacall not installed")
        monkeypatch.setattr(mod, "_JANSEN_RIT_LOADED", False)
        monkeypatch.setattr(mod, "_KERNEL_DIR", Path("/tmp/nonexistent_jansen_rit_dir"))
        with pytest.raises(FileNotFoundError, match="jansen_rit.jl missing"):
            mod._ensure_jansen_rit_loaded()

    def test_wong_wang_missing_jl_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sc_neurocore.accel.julia.neurons as mod

        if not mod._HAS_JULIA_NEURONS:
            pytest.skip("juliacall not installed")
        # Force a re-include path by resetting the loaded flag + pointing
        # the kernel dir at a location that does not contain the .jl.
        monkeypatch.setattr(mod, "_WONG_WANG_LOADED", False)
        monkeypatch.setattr(mod, "_KERNEL_DIR", Path("/tmp/nonexistent_wong_wang_dir"))
        with pytest.raises(FileNotFoundError, match="wong_wang.jl missing"):
            mod._ensure_wong_wang_loaded()

    def test_wilson_cowan_missing_jl_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sc_neurocore.accel.julia.neurons as mod

        if not mod._HAS_JULIA_NEURONS:
            pytest.skip("juliacall not installed")
        monkeypatch.setattr(mod, "_WILSON_COWAN_LOADED", False)
        monkeypatch.setattr(mod, "_KERNEL_DIR", Path("/tmp/nonexistent_wilson_cowan_dir"))
        with pytest.raises(FileNotFoundError, match="wilson_cowan.jl missing"):
            mod._ensure_wilson_cowan_loaded()
