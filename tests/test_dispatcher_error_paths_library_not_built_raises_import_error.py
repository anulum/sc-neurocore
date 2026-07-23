# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLibraryNotBuiltRaisesImportError from former test_dispatcher_error_paths.py

"""Focused suite: TestLibraryNotBuiltRaisesImportError from former test_dispatcher_error_paths.py."""

from __future__ import annotations

from tests.dispatcher_error_paths_support import *  # noqa: F403

class TestLibraryNotBuiltRaisesImportError:
    """When ``_lib`` is absent, calling the dispatcher raises ImportError
    with a helpful message telling the caller how to rebuild."""

    @pytest.mark.parametrize("module,fn_name,_label", CTYPES_DISPATCHERS)
    def test_missing_lib_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: ModuleType,
        fn_name: str,
        _label: str,
    ) -> None:
        monkeypatch.setattr(module, "_lib", None)
        fn = getattr(module, fn_name)
        if "wilson" in fn_name:
            kwargs = dict(ext_input=np.zeros(10))
        elif "jansen" in fn_name:
            kwargs = dict(p_ext=np.zeros(10))
        else:
            kwargs = dict(stim1=np.zeros(10), stim2=np.zeros(10), xi=np.zeros(20))
        with pytest.raises(ImportError, match="not built"):
            if "wilson" in fn_name:
                fn(0.1, 0.05, 10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.1, **kwargs)
            elif "jansen" in fn_name:
                fn(
                    0.1,
                    0.2,
                    0.3,
                    -0.4,
                    -0.1,
                    0.5,
                    3.25,
                    22.0,
                    100.0,
                    50.0,
                    135.0,
                    2.5,
                    6.0,
                    0.56,
                    0.0001,
                    **kwargs,
                )
            else:
                fn(
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                    0.1,
                    0.002,
                    0.641,
                    0.2609,
                    0.0497,
                    0.3255,
                    0.02,
                    0.0001,
                    **kwargs,
                )
