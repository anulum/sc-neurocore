# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests that the online_learning package docstring matches its exports

"""Lock the online_learning package docstring to the methods it actually ships.

The module previously advertised "e-prop, RTRL, and forward-gradient methods"
while only e-prop and an eligibility-based online trainer are implemented. These
tests pin the corrected description so the overclaim cannot reappear.
"""

from __future__ import annotations

import sc_neurocore.online_learning as online_learning


def test_public_api_is_exactly_the_implemented_trainers() -> None:
    """The package exports only the e-prop and online-trainer building blocks."""
    assert sorted(online_learning.__all__) == [
        "EpropTrainer",
        "OnlineLIFLayer",
        "OnlineTrainer",
    ]
    for name in online_learning.__all__:
        assert hasattr(online_learning, name)


def test_docstring_does_not_advertise_unimplemented_methods() -> None:
    """The docstring must not claim RTRL or forward-gradient (neither is implemented)."""
    doc = (online_learning.__doc__ or "").lower()

    assert "rtrl" not in doc
    assert "forward-gradient" not in doc
    assert "forward gradient" not in doc


def test_docstring_names_the_implemented_methods() -> None:
    """The docstring names e-prop and the online trainer it actually provides."""
    doc = online_learning.__doc__ or ""

    assert "e-prop" in doc.lower()
    assert "EpropTrainer" in doc
    assert "OnlineTrainer" in doc
