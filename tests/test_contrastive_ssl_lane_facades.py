# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — refusal paths of the contrastive SSL lane facades

"""The Julia and Mojo facades refuse an absent runtime, a missing source and oversize input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.accel import contrastive_ssl as dispatch
from sc_neurocore.accel.julia.contrastive import ssl as julia_ssl
from sc_neurocore.accel.mojo import contrastive_ssl as mojo_ssl
from tests.native_lane_requirement import require_native_lane


def test_the_julia_facade_refuses_a_missing_runtime_or_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without juliacall, or without ``ssl.jl``, the facade raises instead of running."""
    monkeypatch.setattr(julia_ssl, "_LOADED", False)
    monkeypatch.setattr(julia_ssl, "_jl", None)
    with pytest.raises(ImportError, match="juliacall not available"):
        julia_ssl.ensure_loaded()

    class FakeMain:
        """Stand-in for ``juliacall.Main`` that must never include anything."""

        def include(self, path: str) -> None:
            """Fail the test if a missing source were included."""
            raise AssertionError(path)

    monkeypatch.setattr(julia_ssl, "_jl", FakeMain())
    monkeypatch.setattr(julia_ssl, "_SOURCE", tmp_path / "ssl.jl")
    with pytest.raises(FileNotFoundError, match="ssl.jl missing"):
        julia_ssl.ensure_loaded()


def test_the_mojo_facade_refuses_an_unbuilt_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the shared library every kernel raises the build instruction."""
    monkeypatch.setattr(mojo_ssl, "_lib", None)
    view = np.ones((2, 2))
    for call in (
        lambda: mojo_ssl.infonce_loss(view, view, 0.5),
        lambda: mojo_ssl.csdp_update(view, np.ones(2), np.ones(2), 0.1, 0.0, positive=True),
        lambda: mojo_ssl.goodness(view),
    ):
        with pytest.raises(ImportError, match="libcontrastive_ssl.so not built"):
            call()


def test_the_mojo_facade_refuses_shapes_beyond_the_signed_32_bit_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dimension the C ABI cannot carry is refused before the call."""
    monkeypatch.setattr(mojo_ssl, "_MAX_COUNT", 1)
    view = np.ones((2, 2))
    with pytest.raises(ValueError, match="view shape exceeds"):
        mojo_ssl.infonce_loss(view, view, 0.5)
    with pytest.raises(ValueError, match="weight shape exceeds"):
        mojo_ssl.csdp_update(view, np.ones(2), np.ones(2), 0.1, 0.0, positive=False)


def test_a_kernel_status_is_an_internal_contract_breach() -> None:
    """A non-zero status after validation raises, naming the kernel and the status."""
    mojo_ssl._check(0, "goodness")
    with pytest.raises(RuntimeError, match="Mojo CSDP refused validated input with status 2"):
        mojo_ssl._check(2, "CSDP")


def test_the_built_mojo_kernels_refuse_invalid_arguments_themselves() -> None:
    """Called directly, the library returns its documented refusal statuses."""
    require_native_lane(dispatch, "mojo", subject="contrastive_ssl")
    library = mojo_ssl._library()
    out = np.zeros(4)
    values = np.ones(4)
    assert (
        library.contrastive_ssl_goodness_c(-1, mojo_ssl._address(values), mojo_ssl._address(out))
        == 1
    )
    assert (
        library.contrastive_ssl_goodness_c(4, mojo_ssl._address(values), mojo_ssl._address(values))
        == 1
    )
    bad = np.array([1.0, np.nan, 0.0, 0.0])
    assert (
        library.contrastive_ssl_goodness_c(4, mojo_ssl._address(bad), mojo_ssl._address(out)) == 2
    )
    view = np.ones((2, 2))
    assert (
        library.contrastive_ssl_infonce_c(
            2, 2, -1.0, mojo_ssl._address(view), mojo_ssl._address(view), mojo_ssl._address(out)
        )
        == 2
    )
    assert (
        library.contrastive_ssl_infonce_c(
            2, 0, 0.5, mojo_ssl._address(view), mojo_ssl._address(view), mojo_ssl._address(out)
        )
        == 1
    )
    weights = np.ones((2, 2))
    assert (
        library.contrastive_ssl_csdp_c(
            2,
            2,
            0.1,
            0.0,
            2,
            mojo_ssl._address(weights),
            mojo_ssl._address(values),
            mojo_ssl._address(values),
            mojo_ssl._address(out),
        )
        == 1
    )
    assert (
        library.contrastive_ssl_csdp_c(
            2,
            2,
            -0.1,
            0.0,
            1,
            mojo_ssl._address(weights),
            mojo_ssl._address(values),
            mojo_ssl._address(values),
            mojo_ssl._address(out),
        )
        == 2
    )


def test_an_unloadable_library_marks_the_mojo_lane_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A library the loader cannot open leaves the lane absent instead of raising at import."""

    def unloadable(path: str) -> None:
        raise OSError(f"cannot open {path}")

    monkeypatch.setattr(mojo_ssl, "load_mojo_library", unloadable)
    assert mojo_ssl._load_library() == (None, False)
