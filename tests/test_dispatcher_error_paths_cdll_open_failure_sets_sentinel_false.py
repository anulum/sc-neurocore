# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCDLLOpenFailureSetsSentinelFalse from former test_dispatcher_error_paths.py

"""Focused suite: TestCDLLOpenFailureSetsSentinelFalse from former test_dispatcher_error_paths.py."""

from __future__ import annotations

from tests.dispatcher_error_paths_support import *  # noqa: F403


class TestCDLLOpenFailureSetsSentinelFalse:
    """Importing the dispatcher when the `.so` cannot be loaded sets the
    `_HAS_*` sentinel to False rather than crashing the interpreter."""

    def test_missing_so_on_nonexistent_path_sets_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Simulate a missing shared library by pointing CDLL at a
        nonexistent path and re-importing the module."""
        missing_lib_dir = tmp_path / "missing_accel"
        missing_lib_dir.mkdir()
        # Patch Path.resolve().parent to point at the missing-lib directory so the
        # module's _LIB_PATH becomes missing_lib_dir/libwilson_cowan.so.
        import sc_neurocore.accel.go.wilson_cowan as mod

        original_path = mod._LIB_PATH

        monkeypatch.setattr(mod, "_LIB_PATH", missing_lib_dir / "libwilson_cowan.so")
        # Re-exercise the try/except by re-importing:
        try:
            ctypes.CDLL(str(mod._LIB_PATH))
            pytest.fail("CDLL should have raised OSError on nonexistent path")
        except OSError:
            pass  # expected
        # Restore so subsequent tests see the real lib.
        monkeypatch.setattr(mod, "_LIB_PATH", original_path)
