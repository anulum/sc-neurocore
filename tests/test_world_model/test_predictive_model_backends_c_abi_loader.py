# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (c_abi_loader) from former test_predictive_model_backends.py

from __future__ import annotations

from tests.test_world_model.predictive_model_backends_support import *  # noqa: F403


@pytest.mark.parametrize(
    ("backend_name", "relative_path"),
    [
        ("go", Path("accel/go/lgssm/liblgssm.so")),
        ("mojo", Path("accel/mojo/world_model/liblgssm.so")),
    ],
)
def test_c_abi_loader_handles_cache_missing_file_and_load_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_name: str,
    relative_path: Path,
) -> None:
    ensure = cast(Callable[[], bool], getattr(backends, f"_ensure_{backend_name}_loaded"))
    library_attr = f"_{backend_name}_lib"
    flag_attr = f"_HAS_{backend_name.upper()}_LGSSM"
    monkeypatch.setattr(backends, flag_attr, False)
    monkeypatch.setattr(backends, library_attr, object())
    assert ensure() is True
    assert getattr(backends, flag_attr) is True

    monkeypatch.setattr(backends, library_attr, None)
    monkeypatch.setattr(backends, flag_attr, True)
    monkeypatch.setattr(backends, "_PACKAGE_ROOT", tmp_path)
    assert ensure() is False
    assert getattr(backends, flag_attr) is False

    library_path = tmp_path / relative_path
    library_path.parent.mkdir(parents=True)
    library_path.touch()
    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda _path: (_ for _ in ()).throw(OSError("bad library")),
    )
    assert ensure() is False


@pytest.mark.parametrize(
    ("backend_name", "relative_path", "argument_count"),
    [
        ("go", Path("accel/go/lgssm/liblgssm.so"), 19),
        ("mojo", Path("accel/mojo/world_model/liblgssm.so"), 19),
    ],
)
def test_c_abi_loader_requires_symbol_and_configures_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_name: str,
    relative_path: Path,
    argument_count: int,
) -> None:
    ensure = cast(Callable[[], bool], getattr(backends, f"_ensure_{backend_name}_loaded"))
    library_attr = f"_{backend_name}_lib"
    flag_attr = f"_HAS_{backend_name.upper()}_LGSSM"
    library_path = tmp_path / relative_path
    library_path.parent.mkdir(parents=True)
    library_path.touch()
    monkeypatch.setattr(backends, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(backends, library_attr, None)
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())
    assert ensure() is False

    class FakeFunction:
        argtypes: list[object] | None = None
        restype: object | None = object()

        def __call__(self, *_args: object) -> None:
            return None

    function = FakeFunction()
    library = SimpleNamespace(kalman_filter_c=function)
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: library)
    assert ensure() is True
    assert len(function.argtypes or []) == argument_count
    assert function.restype is None
    assert getattr(backends, library_attr) is library
    assert getattr(backends, flag_attr) is True
