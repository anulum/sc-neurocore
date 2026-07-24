# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (auto_dispatch) from former test_chialvo_map_backends.py

from __future__ import annotations

from tests.chialvo_map_backends_support import *  # noqa: F403


def test_auto_uses_measured_first_available_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto dispatch must honour the data-driven order without hidden fallback."""
    visited: list[str] = []

    def order(_kernel: str, *, static: tuple[str, ...]) -> tuple[str, ...]:
        assert static == chialvo_map._AUTO_BACKENDS
        return ("go", "mojo", "rust", "julia", "python")

    def available(name: str) -> bool:
        visited.append(name)
        return name == "rust"

    def rust_simulate(_neuron: ChialvoMapNeuron, n_steps: int, current: float) -> RunResult:
        assert n_steps == 1
        assert current == 0.0
        return np.array([0.5], dtype=np.float64), 0, 0.5, 0.0

    monkeypatch.setattr(chialvo_map, "select_backend_order", order)
    monkeypatch.setattr(chialvo_map, "_backend_available", available)
    monkeypatch.setattr(ChialvoMapNeuron, "_simulate_rust", rust_simulate)
    trace, spikes = ChialvoMapNeuron().simulate(1, backend="auto")
    np.testing.assert_array_equal(trace, np.array([0.5], dtype=np.float64))
    assert spikes == 0
    assert visited == ["go", "mojo", "rust", "rust"]


def test_auto_python_floor_matches_explicit_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guaranteed auto floor must be the checked Python recurrence."""
    monkeypatch.setattr(chialvo_map, "_auto_backend", lambda: "python")
    explicit = _run("python", n_steps=100, current=0.01)
    automatic = _run("auto", n_steps=100, current=0.01)
    np.testing.assert_array_equal(automatic[0], explicit[0])
    assert automatic[1:] == explicit[1:]


def test_auto_empty_order_and_loaded_backend_loss_keep_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto retains its floor, while stale compiled handles remain explicit errors."""
    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "select_backend_order", lambda *_args, **_kwargs: ())
        trace, _spikes = ChialvoMapNeuron().simulate(2, backend="auto")
        assert trace.shape == (2,)

    with monkeypatch.context() as context:
        context.setattr(
            chialvo_map,
            "select_backend_order",
            lambda *_args, **_kwargs: ("unknown", "python"),
        )
        trace, _spikes = ChialvoMapNeuron().simulate(2, backend="auto")
        assert trace.shape == (2,)

    for backend, attribute in (
        ("rust", "_rust_simulate"),
        ("julia", "_julia_module"),
        ("go", "_go_lib"),
        ("mojo", "_mojo_lib"),
    ):
        with monkeypatch.context() as context:
            context.setattr(chialvo_map, "_backend_available", lambda _name: True)
            context.setattr(chialvo_map, attribute, None)
            with pytest.raises(RuntimeError, match="unavailable"):
                ChialvoMapNeuron().simulate(1, backend=backend)
