# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (smoke) from former test_catalogue_formal.py

from __future__ import annotations

from tests.catalogue_formal_support import *  # noqa: F403


@pytest.mark.parametrize(
    "sby_name",
    [
        "sc_lapicque.sby",
        "sc_perfect_integrator.sby",
        "sc_perfect_integrator_naud_gerstner_2012.sby",
        "sc_quadratic_if.sby",
        "sc_quadratic_if_latham_2000.sby",
        "sc_dpineuron.sby",
        "sc_integerqifneuron.sby",
        "sc_mccullochpittsneuron.sby",
        "sc_poissonneuron.sby",
    ],
)
def test_catalogue_formal_smoke_pass(sby_name: str) -> None:
    """Run a compact subset of catalogue SymbiYosys jobs end-to-end."""
    if shutil.which("sby") is None:
        pytest.skip("sby not on PATH")
    sby_path = CATALOGUE / sby_name
    assert sby_path.is_file()
    sby_text = sby_path.read_text(encoding="utf-8")
    solver = "cvc5" if "smtbmc cvc5" in sby_text else "z3"
    if shutil.which(solver) is None:
        pytest.skip(f"{solver} not on PATH")
    timeout_seconds = 900 if sby_name == "sc_dpineuron.sby" else 120
    proc = subprocess.run(
        ["sby", "-f", sby_name],
        cwd=CATALOGUE,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "DONE (PASS" in combined, combined[-2000:]
