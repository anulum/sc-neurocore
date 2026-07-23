# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNonZeroReturnRaisesRuntimeError from former test_dispatcher_error_paths.py

"""Focused suite: TestNonZeroReturnRaisesRuntimeError from former test_dispatcher_error_paths.py."""

from __future__ import annotations

from tests.dispatcher_error_paths_support import *  # noqa: F403

class TestNonZeroReturnRaisesRuntimeError:
    """When the C shim returns non-zero, the dispatcher raises
    ``RuntimeError`` with the offending return code in the message."""

    @pytest.mark.parametrize("module,fn_name,_label", CTYPES_DISPATCHERS)
    def test_nonzero_return(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module: ModuleType,
        fn_name: str,
        _label: str,
    ) -> None:
        class FakeCShim:
            argtypes: list[object] = []
            restype: object = ctypes.c_int

            def __call__(self, *args: object, **kwargs: object) -> int:
                return 42  # arbitrary non-zero

        class FakeLib:
            pass

        # Real lib is loaded; inject a stub that returns 42.
        if "wilson" in fn_name:
            c_fn_attr = "wilson_cowan_simulate_c"
        elif "jansen" in fn_name:
            c_fn_attr = "jansen_rit_simulate_c"
        else:
            c_fn_attr = "wong_wang_simulate_c"
        lib = FakeLib()
        setattr(lib, c_fn_attr, FakeCShim())
        monkeypatch.setattr(module, "_lib", lib)

        fn = getattr(module, fn_name)
        args: tuple[object, ...]
        if "wilson" in fn_name:
            args = (0.1, 0.05, 10.0, 6.0, 10.0, 1.0, 1.0, 2.0, 1.2, 4.0, 0.1, np.zeros(10))
        elif "jansen" in fn_name:
            args = (
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
                np.zeros(10),
            )
        else:
            args = (
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
                np.zeros(10),
                np.zeros(10),
                np.zeros(20),
            )
        with pytest.raises(RuntimeError, match="42"):
            fn(*args)
