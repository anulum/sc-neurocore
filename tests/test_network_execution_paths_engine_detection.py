# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (engine_detection) from former test_network_execution_paths.py

"""Focused suite: engine-detection tests from former test_network_execution_paths.py."""

from __future__ import annotations

from tests.network_execution_paths_support import *  # noqa: F403


def test_engine_detection_helpers_after_cache_reset() -> None:
    """After a cache reset the engine loads and model support is matched by bare and suffixed names."""
    # Reset the cached engine so the loader runs under the tracer.
    network_module._RUST_ENGINE = None
    engine = network_module._get_rust_engine()
    assert engine is not False
    # A bare model name present directly in the supported set is matched.
    assert network_module._rust_supports_model("AdEx") is True
    # A "...Neuron"-suffixed name matches via the suffix-stripped lookup.
    assert network_module._rust_supports_model("AdExNeuron") is True
    assert network_module._rust_supports_model("DefinitelyNotARealModelNeuron") is False


def test_engine_loader_falls_back_to_top_level_network_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the bridge module the loader falls back to the top-level NetworkRunner."""
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sc_neurocore_engine.network":
            raise ImportError("bridge helper unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert network_module._load_network_runner_class().__name__ == "NetworkRunner"


def test_engine_loader_reports_missing_network_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without any engine module the loader raises ImportError naming NetworkRunner."""
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"sc_neurocore_engine.network", "sc_neurocore_engine"}:
            raise ImportError("engine unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="NetworkRunner is unavailable"):
        network_module._load_network_runner_class()


def test_get_rust_engine_caches_false_when_loader_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing loader makes the engine lookup return False."""

    def fake_loader() -> Any:
        raise ImportError("engine unavailable")

    monkeypatch.setattr(network_module, "_RUST_ENGINE", None)
    monkeypatch.setattr(network_module, "_load_network_runner_class", fake_loader)

    assert network_module._get_rust_engine() is False


def test_rust_supports_model_returns_false_without_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No model is reported as supported when the engine is absent."""
    monkeypatch.setattr(network_module, "_RUST_ENGINE", False)
    assert network_module._rust_supports_model("AdEx") is False


def test_can_use_rust_false_for_unsupported_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A network whose model the engine does not support cannot use Rust."""
    pop = Population(_MODEL, 2)
    net = Network(pop)
    monkeypatch.setattr(network_module, "_rust_supports_model", lambda _name: False)
    assert net._can_use_rust() is False
