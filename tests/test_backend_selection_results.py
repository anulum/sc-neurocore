# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reading the packaged measured backend order

"""The selector reads the packaged order table, and nothing else."""

import json

from sc_neurocore.accel.backend_order import FASTEST_FIRST_BACKENDS
from pathlib import Path
from sc_neurocore.accel import backend_selection as bs
import pytest
from .backend_selection_support import (
    _ADC,
    _DCLS,
    _HOST,
    _MIXED,
    _clear_measured_orders_cache,
)
from .backend_selection_support import fixture_table as fixture_table


def test_the_packaged_table_is_read_through_the_package_resources() -> None:
    """The committed table loads from the installed module's own directory and is well formed."""
    _clear_measured_orders_cache()
    text = bs._read_table()
    assert (
        text.startswith("{")
        and '"schema_version": "sc-neurocore.measured-backend-orders.v1"' in text
    )
    entries = bs.measured_orders()
    assert all(isinstance(entry, bs.MeasuredOrder) and entry.order for entry in entries)


def test_the_fixture_table_gives_its_entries(fixture_table: None) -> None:
    """Each well-formed entry becomes one measured order."""
    kernels = [(entry.kernel, entry.isa_level) for entry in bs.measured_orders()]
    assert kernels == [
        (_DCLS, "unrecorded"),
        (_ADC, "unrecorded"),
        (_MIXED, "unrecorded"),
        ("levelled", "x86-64-v2"),
        ("levelled", "x86-64-v4"),
    ]


def test_an_absent_or_malformed_table_gives_no_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch keeps the static order when the table cannot be read; it never raises."""

    def absent() -> str:
        """Report the measured table as missing."""
        raise FileNotFoundError("measured_backend_orders.json")

    for source in (absent, lambda: "{not-json", lambda: "[]", lambda: '{"entries": {}}'):
        monkeypatch.setattr(bs, "_read_table", source)
        _clear_measured_orders_cache()
        assert bs.measured_orders() == ()
        assert (
            bs.select_backend_order(_DCLS, cpu=_HOST, isa_level="x86-64-v4")
            == FASTEST_FIRST_BACKENDS
        )
    _clear_measured_orders_cache()


def test_malformed_entries_are_skipped_and_wellformed_ones_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry with a wrong type, an unknown level or an empty order is not an order."""
    good = {"kernel": "k", "cpu_model": _HOST, "isa_level": "unrecorded", "order": ["go"]}
    bad = [
        "not-a-mapping",
        {**good, "kernel": 1},
        {**good, "isa_level": "v3"},
        {**good, "order": []},
        {**good, "order": ["go", 2]},
    ]
    monkeypatch.setattr(bs, "_read_table", lambda: json.dumps({"entries": [*bad, good]}))
    _clear_measured_orders_cache()
    assert bs.measured_orders() == (bs.MeasuredOrder("k", _HOST, "unrecorded", ("go",)),)
    _clear_measured_orders_cache()


def test_the_selector_reads_no_benchmark_directory() -> None:
    """The directory scan is gone: the module names no results directory and no repository root."""
    source = Path(bs.__file__).read_text(encoding="utf-8")
    assert '"benchmarks"' not in source and "benchmarks/results" not in source
    assert not hasattr(bs, "_RESULTS_DIR") and not hasattr(bs, "_REPO_ROOT")
