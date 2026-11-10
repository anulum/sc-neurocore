# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public pages claim no board or silicon run

"""Keep public pages and the safety-monitor sources from claiming a silicon run.

No board run has validated any SC-NeuroCore RTL yet (``ARCHITECTURE.md`` §SHD,
``hdl/README.md``), and the safety monitor has no committed synthesis report.
Each phrase below was once published although the repository's own evidence
contradicted it. The match ignores case, line breaks, Markdown emphasis and
``//`` comment prefixes, so a capitalised, emphasised or re-wrapped copy of a
claim fails as well; a case-sensitive match once let "Proven on silicon" pass.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CHECKED_FILES = (
    "ARCHITECTURE.md",
    "ROADMAP.md",
    "README.md",
    "docs/index.md",
    "docs/architecture/SYSTEM_MAP.md",
    "docs/benchmarks/BENCHMARKS.md",
    "docs/COMPETITIVE_LANDSCAPE.md",
    "docs/guides/FOR_RESEARCH_LABS.md",
    "docs/api/formal.md",
    "tools/architecture_map.toml",
    "docs/_generated/capability_manifest.json",
    "docs/_generated/studio_manifest.json",
    "src/sc_neurocore/formal/proofs/safety_bounds.lean",
    "src/sc_neurocore/hdl_gen/safety/safety_monitor.sv",
    "src/sc_neurocore/hdl_gen/safety/tb_safety_monitor.sv",
)
_RUN_CLAIMS = (
    "proven on silicon",
    "proven end-to-end",
    "silicon-proven",
    "real xilinx silicon",
    "fits in under 2%",
    "fires identically on silicon",
    "enforced in silicon",
    "enforces the bound in silicon",
    "every clock edge in silicon",
    "all 6 formally proven",
)


def run_claims_in(text: str) -> list[str]:
    """Return the run claims ``text`` makes, in ``_RUN_CLAIMS`` order.

    ``//`` comment prefixes and Markdown emphasis are removed, every run of
    whitespace becomes one space and the text is case-folded before matching.
    """
    plain = re.sub(r"(?m)^\s*//", " ", text)
    plain = re.sub(r"[*_`]", "", plain)
    folded = " ".join(plain.split()).casefold()
    return [claim for claim in _RUN_CLAIMS if claim in folded]


@pytest.mark.parametrize("relative", _CHECKED_FILES)
def test_file_claims_no_board_or_silicon_run(relative: str) -> None:
    """No checked page or monitor source claims a silicon run or a proof it lacks."""
    text = (_ROOT / relative).read_text(encoding="utf-8")
    assert run_claims_in(text) == [], (
        f"{relative} claims {run_claims_in(text)} while no board run or monitor synthesis exists"
    )


@pytest.mark.parametrize(
    ("published", "claim"),
    [
        (
            "PYNQ-Z2 deployment (`hdl/pynq/`). Proven on silicon\nvia the SHD pipeline.",
            "proven on silicon",
        ),
        ("What fires in simulation fires\nidentically on silicon.", "fires identically on silicon"),
        (
            "in software are enforced in silicon at ≥100 MHz on the PYNQ-Z2",
            "enforced in silicon",
        ),
        ("▼  runs every clock edge in silicon", "every clock edge in silicon"),
        (
            "the hardware monitor `neuro_safe_monitor.sv` enforces the\nbound in silicon, and",
            "enforces the bound in silicon",
        ),
        (
            "// Enforces all 6 formally proven safety boundaries at the nanosecond scale.",
            "all 6 formally proven",
        ),
        ("// Verifies all 6 formally\n// proven safety properties.", "all 6 formally proven"),
        ("**Proven** on silicon via the SHD pipeline.", "proven on silicon"),
    ],
)
def test_each_published_wording_is_caught(published: str, claim: str) -> None:
    """Negative control: the wordings this guard replaced, as published, are each caught."""
    assert run_claims_in(published) == [claim]


def test_corrected_wording_is_not_a_claim() -> None:
    """The replacement wording states the boundary and is not mistaken for a claim."""
    corrected = (
        "The SHD pipeline is synthesised for a Zynq XC7Z020 and matches its Q8.8 reference in"
        " co-simulation; no board run has validated it yet. The monitor checks the 6 safety"
        " bounds stated in safety_bounds.lean every clock cycle (4 proved there, 2 axiomatised)."
    )
    assert run_claims_in(corrected) == []
