# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (availability_rejects) from former test_topology_backends.py

from __future__ import annotations

from tests.topology_backends_support import *  # noqa: F403

def test_invalid_backend_name_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="cuda")


def test_backend_param_preserves_index_validation() -> None:
    with np.testing.assert_raises_regex(ValueError, "integer"):
        ollivier_ricci_curvature(_complete(3), True, 1, backend="python")


@pytest.mark.skipif(_go_available(), reason="Go backend is built; cannot test the unavailable path")
def test_go_unavailable_raises_when_requested() -> None:
    with pytest.raises(RuntimeError, match="Go topology backend"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="go")


@pytest.mark.skipif(
    _mojo_available(), reason="Mojo backend is built; cannot test the unavailable path"
)
def test_mojo_unavailable_raises_when_requested() -> None:
    with pytest.raises(RuntimeError, match="Mojo topology backend"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="mojo")


def test_rust_dispatch_without_handle_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_rust_ollivier", None)
    with pytest.raises(RuntimeError, match="Rust topology backend probed False"):
        topology._ollivier_ricci_rust(_complete(3), 0, 1)


def test_julia_dispatch_without_module_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_julia_module", None)
    with pytest.raises(RuntimeError, match="Julia topology module not loaded"):
        topology._ollivier_ricci_julia(_complete(3), 0, 1)


def test_go_dispatch_without_lib_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_go_lib", None)
    with pytest.raises(RuntimeError, match="Go topology library not loaded"):
        topology._ollivier_ricci_go(_complete(3), 0, 1)


def test_mojo_dispatch_without_lib_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_mojo_lib", None)
    with pytest.raises(RuntimeError, match="Mojo topology library not loaded"):
        topology._ollivier_ricci_mojo(_complete(3), 0, 1)


def test_rust_backend_requested_but_unavailable_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_HAS_RUST_TOPOLOGY", False)
    with pytest.raises(RuntimeError, match="Rust topology backend requested"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="rust")


def test_julia_backend_requested_but_unavailable_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_ensure_julia_loaded", lambda: False)
    with pytest.raises(RuntimeError, match="Julia topology backend requested"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="julia")


def test_go_backend_requested_but_unavailable_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_ensure_go_loaded", lambda: False)
    with pytest.raises(RuntimeError, match="Go topology backend requested"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="go")


def test_mojo_backend_requested_but_unavailable_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_ensure_mojo_loaded", lambda: False)
    with pytest.raises(RuntimeError, match="Mojo topology backend requested"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="mojo")
