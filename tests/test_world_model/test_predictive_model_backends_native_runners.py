# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (native_runners) from former test_predictive_model_backends.py

from __future__ import annotations

from tests.test_world_model.predictive_model_backends_support import *  # noqa: F403

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
    monkeypatch.setattr(backends, attribute, None)
    observations, controls = _inputs()
    with pytest.raises(RuntimeError, match=message):
        backends.filter_native(backend, _model(), observations, controls)


def test_filter_native_rejects_python_backend() -> None:
    observations, controls = _inputs()
    with pytest.raises(ValueError, match="cannot execute the Python backend"):
        backends.filter_native("python", _model(), observations, controls)


