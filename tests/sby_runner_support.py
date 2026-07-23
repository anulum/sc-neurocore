# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sby_runner.py

from __future__ import annotations

"""Tests for the shared ``sby`` task runner.

The pure tests drive the parser, the tool probe, the verdict-completeness guard,
and the subprocess boundary with a crafted fake process, so they run everywhere.
One end-to-end test runs a real ``sby`` task and self-skips when the formal
toolchain (``sby`` / ``yosys`` / a solver) is absent, as on CI.
"""
import subprocess
from pathlib import Path
import pytest
from sc_neurocore.compiler import _sby_runner
from sc_neurocore.compiler._sby_runner import (
    SbyRun,
    formal_tools_available,
    is_inconclusive,
    parse_verdict,
    raise_for_incomplete,
    run_sby_task,
)
_HAS_FORMAL = formal_tools_available()
_needs_formal = pytest.mark.skipif(
    not _HAS_FORMAL, reason="SymbiYosys / Yosys / solver not available"
)
class _FakeProc:
    """Stand-in for a finished ``subprocess.run`` result."""

    def __init__(self, stdout: str, returncode: int) -> None:
        self.stdout = stdout
        self.returncode = returncode

__all__ = ['subprocess', 'Path', 'pytest', '_sby_runner', 'SbyRun', 'formal_tools_available', 'is_inconclusive', 'parse_verdict', 'raise_for_incomplete', 'run_sby_task', '_HAS_FORMAL', '_needs_formal', '_FakeProc']
