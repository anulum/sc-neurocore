# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — dispatch contract of the contrastive SSL lanes

"""The contrastive SSL dispatcher validates like the original and selects lanes honestly."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.accel import contrastive_ssl as dispatch
from sc_neurocore.contrastive.ssl import CSDPRule, SpikeContrastiveLoss

RNG = np.random.default_rng(20261123)
VIEW_A = RNG.normal(size=(6, 4))
VIEW_B = RNG.normal(size=(6, 4))
WEIGHTS = RNG.normal(size=(3, 5))
PRE = RNG.random(5)
POST = RNG.random(3)


def test_the_python_lane_is_the_original() -> None:
    """With ``backend="python"`` every kernel returns exactly what the original returns."""
    assert dispatch.infonce_loss(VIEW_A, VIEW_B, temperature=0.3, backend="python") == (
        SpikeContrastiveLoss(0.3).compute(VIEW_A, VIEW_B)
    )
    rule = CSDPRule(lr=0.02, decay=0.004)
    for positive, phase in ((True, rule.positive_update), (False, rule.negative_update)):
        updated = dispatch.csdp_update(
            WEIGHTS, PRE, POST, positive=positive, lr=0.02, decay=0.004, backend="python"
        )
        np.testing.assert_array_equal(updated, phase(WEIGHTS, PRE, POST))
    assert dispatch.goodness(VIEW_A, backend="python") == CSDPRule().goodness(VIEW_A)
    assert dispatch.infonce_loss(VIEW_A[:1], VIEW_B[:1], backend="python") == 0.0


def test_auto_runs_python_without_a_measured_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a committed measurement for this host, ``auto`` is the Python original."""
    monkeypatch.setattr(dispatch, "select_backend_order", lambda kernel, static: static)
    assert dispatch.auto_backend() == "python"
    assert dispatch.goodness([1.0, -2.0, 0.5]) == 5.25


def test_auto_follows_a_measured_order_to_an_available_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measured order leads with a lane only when that lane is available."""
    monkeypatch.setattr(
        dispatch, "select_backend_order", lambda kernel, static: ("mojo", "julia", "python")
    )
    monkeypatch.setattr(dispatch, "backend_available", lambda backend: backend == "julia")
    assert dispatch.auto_backend() == "julia"
    monkeypatch.setattr(dispatch, "backend_available", lambda backend: False)
    assert dispatch.auto_backend() == "python"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: dispatch.infonce_loss(VIEW_A, VIEW_B, temperature=0.0),
            "temperature must be positive",
        ),
        (lambda: dispatch.infonce_loss(VIEW_A, VIEW_B[:, :3]), "same shape"),
        (lambda: dispatch.infonce_loss(VIEW_A[0], VIEW_B[0]), "view_a must be a 2-D array"),
        (lambda: dispatch.infonce_loss(np.empty((3, 0)), np.empty((3, 0))), "at least one feature"),
        (
            lambda: dispatch.infonce_loss([[np.nan]], [[1.0]]),
            "view_a must contain only finite values",
        ),
        (
            lambda: dispatch.csdp_update(WEIGHTS, PRE, POST, positive=True, lr=-1.0),
            "lr must be non-negative",
        ),
        (
            lambda: dispatch.csdp_update(WEIGHTS, PRE, POST, positive=True, decay=np.inf),
            "decay must be finite",
        ),
        (
            lambda: dispatch.csdp_update(WEIGHTS[:, :4], PRE, POST, positive=True),
            "weights must have shape",
        ),
        (
            lambda: dispatch.csdp_update(WEIGHTS[0], PRE, POST, positive=False),
            "weights must be a 2-D array",
        ),
        (
            lambda: dispatch.csdp_update(WEIGHTS, PRE[None, :], POST, positive=False),
            "pre_spikes must be a 1-D",
        ),
        (lambda: dispatch.goodness([1.0, np.nan]), "activations must contain only finite values"),
        (lambda: dispatch.goodness([1.0], backend=cast(Any, "cuda")), "backend must be one of"),
    ],
)
def test_invalid_inputs_are_refused_before_any_lane_runs(call: Any, message: str) -> None:
    """Every refusal of the original is raised by the dispatcher, whatever lane is chosen."""
    with pytest.raises(ValueError, match=message):
        call()


def test_a_named_unavailable_lane_is_an_error_not_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for a lane this host lacks raises; it never silently runs another lane."""
    monkeypatch.setattr(dispatch, "backend_available", lambda backend: backend == "python")
    with pytest.raises(RuntimeError, match="the mojo lane of contrastive_ssl is not available"):
        dispatch.goodness([1.0], backend="mojo")


def test_lane_availability_reflects_the_runtime_and_the_built_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lane is available only when its runtime loads; a broken runtime is reported absent."""
    assert dispatch.backend_available("python") is True
    assert dispatch.backend_available("rust") is False

    def lane(module: Any) -> Any:
        return lambda backend: module

    def raising(error: BaseException) -> Any:
        def ensure_loaded() -> None:
            raise error

        return SimpleNamespace(ensure_loaded=ensure_loaded)

    julia_error = type("JuliaError", (Exception,), {})
    for error, available in (
        (ImportError("no juliacall"), False),
        (FileNotFoundError("no ssl.jl"), False),
        (julia_error("broken project"), False),
    ):
        monkeypatch.setattr(dispatch, "_lane", lane(raising(error)))
        assert dispatch.backend_available("julia") is available
    monkeypatch.setattr(dispatch, "_lane", lane(SimpleNamespace(ensure_loaded=lambda: None)))
    assert dispatch.backend_available("julia") is True
    monkeypatch.setattr(dispatch, "_lane", lane(raising(KeyError("unexpected"))))
    with pytest.raises(KeyError):
        dispatch.backend_available("julia")

    for built in (True, False):
        monkeypatch.setattr(
            dispatch, "_lane", lane(SimpleNamespace(_HAS_MOJO_CONTRASTIVE_SSL=built))
        )
        assert dispatch.backend_available("mojo") is built

    def missing(backend: str) -> Any:
        raise ImportError(backend)

    monkeypatch.setattr(dispatch, "_lane", missing)
    assert dispatch.backend_available("mojo") is False


def test_the_dispatcher_hands_validated_contiguous_inputs_to_a_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lane receives C-contiguous float64 arrays and the validated rule parameters."""
    seen: dict[str, Any] = {}

    def infonce_loss(a: Any, b: Any, temperature: float) -> float:
        seen["loss"] = (a.flags.c_contiguous, b.dtype, temperature)
        return 1.5

    def csdp_update(w: Any, pre: Any, post: Any, lr: float, decay: float, *, positive: bool) -> Any:
        seen["csdp"] = (w.flags.c_contiguous, lr, decay, positive)
        return w + 1.0

    def goodness(values: Any) -> float:
        seen["goodness"] = values.flags.c_contiguous
        return 2.5

    fake = SimpleNamespace(infonce_loss=infonce_loss, csdp_update=csdp_update, goodness=goodness)
    monkeypatch.setattr(dispatch, "backend_available", lambda backend: True)
    monkeypatch.setattr(dispatch, "_lane", lambda backend: fake)
    assert (
        dispatch.infonce_loss(VIEW_A.T.T[:, ::-1], VIEW_B, temperature=0.7, backend="mojo") == 1.5
    )
    assert seen["loss"] == (True, np.dtype(np.float64), 0.7)
    np.testing.assert_array_equal(
        dispatch.csdp_update(
            WEIGHTS.T.T, PRE, POST, positive=False, lr=0.5, decay=0.25, backend="julia"
        ),
        WEIGHTS + 1.0,
    )
    assert seen["csdp"] == (True, 0.5, 0.25, False)
    assert dispatch.goodness(VIEW_A.T, backend="mojo") == 2.5
    assert seen["goodness"] is True
