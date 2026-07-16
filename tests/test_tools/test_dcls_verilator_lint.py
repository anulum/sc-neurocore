# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilator --Wall lint gate for the DCLS Q8.8 datapath

"""Keep the DCLS Q8.8 datapath free of inferred latches and Verilator lint regressions.

The combinational tent kernel (`sc_dcls_tent_kernel`) previously left six per-tap scratch
registers unassigned on the invalid-sigma and non-spiking control paths, so Verilator inferred
a latch for each — reported twice (module standalone and as an instance of `sc_dcls_layer_core`),
the "12 inferred latches" flagged in the KIMI audit. Defaulting the scratch registers closed all
of them. This gate lints the DCLS datapath with `-Wall` and asserts:

* zero ``LATCH`` warnings (the correctness class stays closed — a latch on a datapath scratch
  register is always a coding defect here, never intentional); and
* the total warning count stays within a ratchet ceiling, so no new lint warning creeps in.

The ceiling is a ratchet: lower it as modules are cleaned, never raise it. Skipped when
Verilator is unavailable (CI installs it in the HDL job).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HDL_DIR = _REPO_ROOT / "hdl"

# DCLS Q8.8 datapath tops: the tent kernel (the inferred-latch site, linted standalone so a latch
# is caught even when the layer is not instantiated) and the layer core that instantiates it and
# pulls in the axonal delay line.
_DCLS_TOP_MODULES = ("sc_dcls_tent_kernel.v", "sc_dcls_layer_core.v")

# Verilator -Wall warning ceiling per top after the KR-8 clean-up. RATCHET: only ever lower it.
_WARNING_CEILING = 0

_LATCH_RE = re.compile(r"^%Warning-LATCH", re.MULTILINE)
_WARNING_RE = re.compile(r"^%Warning-", re.MULTILINE)

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator is not installed"
)


def _lint(top: str) -> subprocess.CompletedProcess[str]:
    """Run ``verilator --lint-only -Wall`` on ``top`` within the HDL include path."""
    return subprocess.run(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            f"-I{_HDL_DIR}",
            str(_HDL_DIR / top),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.parametrize("top", _DCLS_TOP_MODULES)
def test_dcls_datapath_has_no_inferred_latches(top: str) -> None:
    """No combinational latch may be inferred anywhere in the DCLS datapath."""
    result = _lint(top)
    output = result.stdout + result.stderr
    latches = _LATCH_RE.findall(output)
    assert not latches, f"{top}: {len(latches)} inferred-latch warning(s):\n{output}"


@pytest.mark.parametrize("top", _DCLS_TOP_MODULES)
def test_dcls_datapath_stays_within_verilator_warning_ratchet(top: str) -> None:
    """The DCLS datapath must not accumulate new Verilator -Wall warnings."""
    result = _lint(top)
    output = result.stdout + result.stderr
    warnings = _WARNING_RE.findall(output)
    assert len(warnings) <= _WARNING_CEILING, (
        f"{top}: {len(warnings)} -Wall warning(s) exceed the ratchet ceiling "
        f"{_WARNING_CEILING}; fix them or, only if genuinely intentional, lower the "
        f"ceiling deliberately:\n{output}"
    )
    # A clean top (ceiling 0) must exit 0; a non-zero exit with no counted warning means the
    # lint failed to run (e.g. a parse error), which must not pass silently.
    if _WARNING_CEILING == 0:
        assert result.returncode == 0, f"{top}: verilator exited {result.returncode}:\n{output}"
