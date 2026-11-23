# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBackendSelector from former test_sc_forward.py

"""Focused suite: TestBackendSelector from former test_sc_forward.py."""

from __future__ import annotations

import numpy as np
import numpy.typing as nptyp
import pytest
from sc_neurocore.accel import (
    available_backends,
    backend as backend_mod,
    get_backend,
    sc_forward,
)
from sc_neurocore.accel.backend import (
    NumpyBackend,
    RustBackend,
)
from tests.sc_forward_support import (
    _pack_weights,
)


class TestBackendSelector:
    """NEU-SCPN.1 — get_backend / available_backends."""

    def test_auto_returns_known_backend(self) -> None:
        """The automatic choice is a known backend."""
        assert get_backend().name in backend_mod.PRIORITY

    def test_explicit_numpy(self) -> None:
        """Asking for NumPy gives the NumPy backend."""
        assert get_backend("numpy").name == "numpy"

    def test_unknown_name_rejected(self) -> None:
        """An unknown backend name is rejected."""
        with pytest.raises(ValueError, match="unknown backend"):
            get_backend("cuda")

    def test_unavailable_backend_raises(self) -> None:
        """A backend that is not available is reported."""
        # mojo is never implemented for this surface -> probe returns None.
        with pytest.raises(RuntimeError, match="not available"):
            get_backend("mojo")

    def test_available_backends_reports_numpy(self) -> None:
        """The availability report covers every backend and has NumPy."""
        status = available_backends()
        assert status["numpy"] is True
        assert set(status) == set(backend_mod.PRIORITY)

    def test_backend_instance_passed_through(self) -> None:
        """A backend instance is used as given."""
        packed = _pack_weights(np.array([[0.5]]), 256, seed=1)
        out = sc_forward(packed, np.array([0.5]), length=256, backend=NumpyBackend())
        assert out.shape == (1,)

    def test_auto_falls_back_to_numpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without faster backends the automatic choice is NumPy."""
        monkeypatch.setattr(backend_mod, "_probe", lambda name: None)
        assert get_backend("auto").name == "numpy"

    def test_probe_rust_handles_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failing Rust probe reports no backend."""

        def _boom(*_args: object, **_kwargs: object) -> nptyp.NDArray[np.float64]:
            raise RuntimeError("engine missing")

        monkeypatch.setattr(RustBackend, "sc_forward", _boom)
        assert backend_mod._probe("rust") is None
