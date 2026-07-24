# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (manual_template) from former test_package_materialisation.py

from __future__ import annotations

from tests.test_safety_cert.package_materialisation_support import *  # noqa: F403


def test_manual_template_is_deterministic_and_non_certifying() -> None:
    """Manual output must state its limits and use a non-normative crosswalk."""
    report = SafetyManualGenerator.generate(
        "Example Controller",
        SILLevel.SIL_2,
        ["neuron"],
        42.5,
        generated_on="2026-07-12",
    )
    assert "2026-07-12" in report
    assert "Draft evidence template only" in report
    assert "does not establish equivalence" in report
    with pytest.raises(ValueError, match="generated_on"):
        SafetyManualGenerator.generate(
            "Example Controller",
            SILLevel.SIL_2,
            ["neuron"],
            42.5,
            generated_on="12 July 2026",
        )
