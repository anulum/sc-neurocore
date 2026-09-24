# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio calls a model perfect only on verified tiers

"""The declared and the verified perfect judgements are served side by side."""

from __future__ import annotations

import pytest

from sc_neurocore.studio.model_catalogue import _verified_perfect, get_model_detail


def test_a_declared_perfect_model_without_fresh_receipts_is_not_verified_perfect() -> None:
    detail = get_model_detail("AdExNeuron")
    assert detail is not None
    readiness = detail["readiness"]
    assert readiness["is_perfect"] is True
    assert readiness["is_perfect_verified"] is False
    verified = readiness["verified"]
    assert readiness["is_perfect_verified"] == _verified_perfect(
        verified["science_tier"], verified["silicon_tier"], readiness["terminal_silicon_tier"]
    )


@pytest.mark.parametrize(
    ("science", "silicon", "target", "perfect"),
    [
        (5, 2, "H2", True),
        (5, 3, "H2", True),
        (5, 1, "H2", False),
        (4, 5, "H2", False),
        (5, None, "H2", False),
        (5, 5, None, False),
        (5, 5, "H9", False),
    ],
)
def test_verified_perfect_needs_s5_and_the_terminal_tier(
    science: int, silicon: int | None, target: str | None, perfect: bool
) -> None:
    assert _verified_perfect(science, silicon, target) is perfect
