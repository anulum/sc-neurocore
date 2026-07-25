# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MPR native-runner and C-facade contracts

"""Native-runner and Go/Mojo C-facade boundary contracts for MPR dispatch."""

from .ermentrout_kopell_pop_dispatch_support import *


def test_native_runner_rechecks_rust_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backends, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="Rust MPR backend is unavailable"):
        backends._native_runner("rust")


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_input_shape_and_missing_library_fail_closed(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        module.simulate_ermentrout_kopell_pop(*_PARAMETERS, np.zeros((2, 2)))

    monkeypatch.setattr(module, "_lib", None)
    with pytest.raises(ImportError, match="libermentrout_kopell_pop.so not built"):
        module.simulate_ermentrout_kopell_pop(*_PARAMETERS, np.asarray([1.5]))


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_step_bound_precedes_contiguous_copy(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = np.broadcast_to(np.asarray([1.0]), ((1 << 31),))

    def unexpected_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized logical input reached contiguous allocation")

    monkeypatch.setattr(module.np, "ascontiguousarray", unexpected_copy)
    with pytest.raises(ValueError, match="signed-32-bit step limit"):
        module.simulate_ermentrout_kopell_pop(*_PARAMETERS, oversized)


@pytest.mark.parametrize("module", (go_backend, mojo_backend))
def test_c_facade_library_probe_handles_loader_failure(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_load(_path: str) -> None:
        raise OSError("shared library unavailable")

    monkeypatch.setattr(module.ctypes, "CDLL", fail_to_load)
    assert module._load_library() == (None, False)
