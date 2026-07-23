# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_criterion_to_bencher.py

"""Module-level tests from former test_criterion_to_bencher.py."""

from __future__ import annotations

from criterion_to_bencher_support import *  # noqa: F403

def test_main_reads_stdin(monkeypatch, capsys) -> None:
    """The CLI entry point converts stdin to stdout."""
    import io

    monkeypatch.setattr(_CONVERTER.sys, "stdin", io.StringIO("b  time:   [1.0 µs 2.0 µs 3.0 µs]"))
    _CONVERTER.main()
    assert capsys.readouterr().out.strip() == "test b ... bench: 2000 ns/iter (+/- 0)"
