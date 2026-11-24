# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The dispatch order from the packaged measured order

"""How a measured order reorders the static dispatch order, by host model and instruction-set level."""

from sc_neurocore.accel.backend_order import FASTEST_FIRST_BACKENDS
from sc_neurocore.accel import backend_selection as bs
import pytest
from .backend_selection_support import (
    _ADC,
    _DCLS,
    _HOST,
    _MIXED,
)
from .backend_selection_support import fixture_table as fixture_table


def test_select_uses_the_measured_order_for_the_measuring_host(fixture_table: None) -> None:
    """On the measuring host the measured lanes lead, fastest first, and the floor stays last."""
    assert bs.select_backend_order(_DCLS, cpu=_HOST, isa_level="x86-64-v3") == (
        "rust",
        "julia",
        "mojo",
        "go",
        "python",
    )


def test_the_measured_winner_differs_per_kernel(fixture_table: None) -> None:
    """The fastest lane is kernel-specific, not a fixed first choice."""
    assert bs.select_backend_order(_ADC, cpu=_HOST, isa_level="x86-64-v3")[0] == "mojo"
    assert bs.select_backend_order(_MIXED, cpu=_HOST, isa_level="x86-64-v3")[0] == "julia"


def test_an_unknown_kernel_or_host_gets_the_static_order(fixture_table: None) -> None:
    """With no entry for the kernel on this host model, the static order is returned verbatim."""
    assert (
        bs.select_backend_order("no_such_kernel", cpu=_HOST, isa_level="x86-64-v3")
        == FASTEST_FIRST_BACKENDS
    )
    assert (
        bs.select_backend_order(_DCLS, cpu="Imaginary CPU 1", isa_level="x86-64-v3")
        == FASTEST_FIRST_BACKENDS
    )


def test_an_unrecorded_entry_applies_on_the_exact_model_whatever_the_level(
    fixture_table: None,
) -> None:
    """An entry whose record states no flags is bound to its host model only."""
    for level in ("unknown", "x86-64-v1", "x86-64-v4"):
        assert bs.select_backend_order(_DCLS, cpu=_HOST, isa_level=level)[0] == "rust"


def test_a_recorded_level_applies_only_at_that_level_or_above(fixture_table: None) -> None:
    """A v4 measurement orders a v4 host; a v3 host gets the v2 measurement; v1 and unknown hosts get neither."""
    assert bs.select_backend_order("levelled", cpu=_HOST, isa_level="x86-64-v4")[0] == "mojo"
    assert bs.select_backend_order("levelled", cpu=_HOST, isa_level="x86-64-v3")[0] == "go"
    assert bs.select_backend_order("levelled", cpu=_HOST, isa_level="x86-64-v2")[0] == "go"
    for level in ("x86-64-v1", "unknown"):
        assert (
            bs.select_backend_order("levelled", cpu=_HOST, isa_level=level)
            == FASTEST_FIRST_BACKENDS
        )


def test_the_floor_is_last_and_the_backend_set_is_kept(fixture_table: None) -> None:
    """Reordering never adds or drops a lane, and the floor is always the final tier."""
    for kernel in (_DCLS, _ADC, _MIXED, "levelled"):
        order = bs.select_backend_order(kernel, cpu=_HOST, isa_level="x86-64-v4")
        assert order[-1] == "python"
        assert set(order) == set(FASTEST_FIRST_BACKENDS)


def test_a_custom_numpy_floor_is_respected(fixture_table: None) -> None:
    """A measured lane outside the static order is dropped, and numpy stays the floor."""
    static = ("rust", "mojo", "julia", "go", "numpy")
    assert bs.select_backend_order(_DCLS, static=static, cpu=_HOST, isa_level="x86-64-v4") == (
        "rust",
        "julia",
        "mojo",
        "go",
        "numpy",
    )


def test_an_empty_static_order_returns_empty(fixture_table: None) -> None:
    """There is nothing to reorder."""
    assert bs.select_backend_order(_DCLS, static=(), cpu=_HOST, isa_level="x86-64-v4") == ()


def test_the_running_host_is_used_when_no_host_is_given(
    fixture_table: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without arguments the running host's model and level decide."""
    monkeypatch.setattr(bs, "current_cpu", lambda: _HOST)
    monkeypatch.setattr(bs, "_running_level", lambda: "x86-64-v4")
    assert bs.select_backend_order("levelled")[0] == "mojo"


def test_a_level_that_is_not_a_level_is_refused(fixture_table: None) -> None:
    """A caller's misspelt level is an error, not a silent static order."""
    with pytest.raises(ValueError, match="not a level"):
        bs.select_backend_order("levelled", cpu=_HOST, isa_level="v4")
