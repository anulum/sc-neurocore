# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNativeBackendDispatch from former test_gamma_oscillation.py

"""Focused suite: TestNativeBackendDispatch from former test_gamma_oscillation.py."""

from __future__ import annotations

from tests.gamma_oscillation_support import *  # noqa: F403

class TestNativeBackendDispatch:
    """Explicit native backends expose the same public step contract."""

    @pytest.mark.parametrize(
        "backend,availability_flag,match",
        [
            (
                "rust",
                "_HAS_RUST_PING_STEP",
                "sc_neurocore_engine.py_ping_step",
            ),
            ("julia", "_HAS_JULIA_PING_STEP", "julia kernel"),
            ("go", "_HAS_GO_PING_STEP", "go kernel"),
            ("mojo", "_HAS_MOJO_PING_STEP", "mojo kernel"),
        ],
    )
    def test_explicit_backend_fails_closed_when_kernel_unavailable(
        self,
        monkeypatch,
        backend,
        availability_flag,
        match,
    ):
        monkeypatch.setattr(gamma_oscillation_module, availability_flag, False)
        with pytest.raises(RuntimeError, match=match):
            PINGCircuit(backend=backend)

    @pytest.mark.parametrize(
        "backend,is_available",
        [
            ("julia", _HAS_JULIA_PING_STEP),
            ("go", _HAS_GO_PING_STEP),
            ("mojo", _HAS_MOJO_PING_STEP),
        ],
    )
    def test_explicit_backend_produces_boolean_spike_trains(self, backend, is_available):
        if not is_available:
            pytest.skip(f"{backend} PING kernel is not built")

        ping = PINGCircuit(
            n_excitatory=12,
            n_inhibitory=6,
            i_drive_e_mean=4.0,
            i_drive_e_sigma=0.0,
            i_drive_i_mean=4.0,
            i_drive_i_sigma=0.0,
            sigma_e=0.0,
            sigma_i=0.0,
            seed=17,
            backend=backend,
        )

        total_e = 0
        total_i = 0
        for _ in range(250):
            spikes_e, spikes_i = ping.step(dt=0.1)
            assert spikes_e.dtype == np.bool_
            assert spikes_i.dtype == np.bool_
            assert spikes_e.shape == (12,)
            assert spikes_i.shape == (6,)
            total_e += int(np.count_nonzero(spikes_e))
            total_i += int(np.count_nonzero(spikes_i))

        assert total_e > 0
        assert total_i > 0
        assert np.all(np.isfinite(ping.v_e))
        assert np.all(np.isfinite(ping.v_i))
        assert np.all(ping.g_ampa_e >= 0.0)
        assert np.all(ping.g_gaba_i >= 0.0)

    def test_rust_kernel_discovery_falls_back_without_import_side_effects(self, monkeypatch):
        real_import_module = gamma_oscillation_module._importlib.import_module

        def reject_rust_engine(name):
            if name in {"sc_neurocore_engine.sc_neurocore_engine", "sc_neurocore_engine"}:
                raise ImportError(name)
            return real_import_module(name)

        monkeypatch.setattr(
            gamma_oscillation_module._importlib, "import_module", reject_rust_engine
        )
        _saved_ns = snapshot_module_namespace(gamma_oscillation_module)
        reloaded = importlib.reload(gamma_oscillation_module)
        try:
            assert reloaded._HAS_RUST_PING_STEP is False
            assert reloaded._rust_ping_step is None
            with pytest.raises(RuntimeError, match="sc_neurocore_engine.py_ping_step"):
                reloaded.PINGCircuit(backend="rust")
        finally:
            monkeypatch.undo()
            restore_module_namespace(gamma_oscillation_module, _saved_ns)

    def test_rust_kernel_discovery_uses_root_package_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        importlib_module = cast(Any, gamma_oscillation_module)._importlib
        real_import_module = importlib_module.import_module

        def fallback_root_engine(name: str) -> object:
            if name == "sc_neurocore_engine.sc_neurocore_engine":
                raise ImportError(name)
            if name == "sc_neurocore_engine":
                return SimpleNamespace(py_ping_step=lambda *args: (0, 0))
            return real_import_module(name)

        monkeypatch.setattr(importlib_module, "import_module", fallback_root_engine)
        _saved_ns = snapshot_module_namespace(gamma_oscillation_module)
        reloaded = importlib.reload(gamma_oscillation_module)
        try:
            assert reloaded._HAS_RUST_PING_STEP is True
            assert reloaded._rust_ping_step is not None
        finally:
            monkeypatch.undo()
            restore_module_namespace(gamma_oscillation_module, _saved_ns)

    def test_julia_discovery_failure_remains_optional(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "juliacall", None)
        _saved_ns = snapshot_module_namespace(gamma_oscillation_module)
        reloaded = importlib.reload(gamma_oscillation_module)
        try:
            assert reloaded._HAS_JULIA_PING_STEP is False
            assert reloaded._julia_ping_step is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(gamma_oscillation_module, _saved_ns)

    def test_ctypes_backend_discovery_failures_remain_optional(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_exists(path: str) -> bool:
            return path.endswith("libgamma_oscillation.so")

        def reject_cdll(path: str) -> object:
            raise OSError(path)

        monkeypatch.setattr(os.path, "exists", fake_exists)
        monkeypatch.setattr(ctypes, "CDLL", reject_cdll)
        _saved_ns = snapshot_module_namespace(gamma_oscillation_module)
        reloaded = importlib.reload(gamma_oscillation_module)
        try:
            assert reloaded._HAS_GO_PING_STEP is False
            assert reloaded._go_ping_step is None
            assert reloaded._HAS_MOJO_PING_STEP is False
            assert reloaded._mojo_ping_step is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(gamma_oscillation_module, _saved_ns)
