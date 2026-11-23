# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (native_runners) from former test_predictive_model_backends.py

"""Native runners of the predictive-model backends fail closed."""

from __future__ import annotations

import pytest
from sc_neurocore.world_model import (
    _lgssm_backends as backends,
)
from sc_neurocore.world_model._lgssm_backends import (
    ExplicitBackendName,
)
from tests.test_world_model.predictive_model_backends_support import (
    _inputs,
    _model,
)


@pytest.mark.parametrize(
    ("backend", "attribute", "message"),
    [
        ("rust", "_rust_kalman_filter", "Rust backend was selected"),
        ("julia", "_julia_module", "Julia backend was selected"),
        ("go", "_go_lib", "Go backend was selected"),
        ("mojo", "_mojo_lib", "Mojo backend was selected"),
    ],
)
def test_native_runners_fail_closed_without_loaded_runtime(
    monkeypatch: pytest.MonkeyPatch,
    backend: ExplicitBackendName,
    attribute: str,
    message: str,
) -> None:
    """Each native backend fails closed when its runtime is not loaded."""
    monkeypatch.setattr(backends, attribute, None)
    observations, controls = _inputs()
    with pytest.raises(RuntimeError, match=message):
        backends.filter_native(backend, _model(), observations, controls)


def test_filter_native_rejects_python_backend() -> None:
    """The native filter refuses the Python backend."""
    observations, controls = _inputs()
    with pytest.raises(ValueError, match="cannot execute the Python backend"):
        backends.filter_native("python", _model(), observations, controls)
