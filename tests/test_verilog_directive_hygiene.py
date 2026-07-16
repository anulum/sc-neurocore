# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog preprocessor-directive hygiene

"""Reject malformed Verilog preprocessor directives in RTL and testbench sources.

One responsibility: guard every tracked ``.v`` source against a *dangling backtick* — a
``\\``` that is not immediately followed by a directive or macro identifier. Verilog's
backtick always introduces a compiler directive (```define``, ```timescale```, ```ifdef``)
or a macro reference, so a lone or whitespace-trailing backtick is a hard syntax error that
Icarus Verilog rejects only during a full elaboration — which the standalone ``tb/`` sources
never receive in CI, so such a typo can survive silently. This hermetic lint catches the
class with no toolchain dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_DIRS = ("hdl", "tb")

# A backtick that does not begin an identifier (directive keyword or macro name).
_DANGLING_BACKTICK = re.compile(r"`(?![A-Za-z_])")
# Drop double-quoted string literals so an in-string backtick is not misread as a directive.
_STRING_LITERAL = re.compile(r'"[^"]*"')


def _verilog_sources() -> list[Path]:
    """Return every tracked Verilog source under the RTL and testbench trees."""
    sources: list[Path] = []
    for directory in _SOURCE_DIRS:
        sources.extend(sorted((_REPO_ROOT / directory).rglob("*.v")))
    return sources


def _dangling_backtick_lines(source: Path) -> list[str]:
    """Return ``path:line: text`` for each dangling backtick in ``source``."""
    violations: list[str] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        code = line.split("//", 1)[0]  # strip line comments
        code = _STRING_LITERAL.sub("", code)  # strip string literals
        if _DANGLING_BACKTICK.search(code):
            violations.append(f"{source.relative_to(_REPO_ROOT)}:{number}: {line.rstrip()}")
    return violations


@pytest.mark.parametrize("source", _verilog_sources(), ids=lambda path: str(path.name))
def test_verilog_source_has_no_dangling_backtick(source: Path) -> None:
    """Every RTL/testbench ``.v`` source is free of malformed backtick directives."""
    violations = _dangling_backtick_lines(source)
    assert not violations, "dangling backtick(s):\n" + "\n".join(violations)


def test_verilog_source_discovery_is_not_vacuous() -> None:
    """Guard: both source trees resolve to real ``.v`` files so the lint is not empty."""
    for directory in _SOURCE_DIRS:
        assert list((_REPO_ROOT / directory).rglob("*.v")), f"no .v sources under {directory}/"
