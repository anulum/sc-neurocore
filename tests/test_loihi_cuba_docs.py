# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path


def _loihi_cuba_doc() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "docs" / "api" / "models" / "loihi_cuba.md").read_text(encoding="utf-8")


def test_loihi_cuba_doc_rejects_stale_stub_claims() -> None:
    text = _loihi_cuba_doc()

    assert "STUB" not in text
    assert "Spikes (10K steps, I=5.0) | 1999" not in text
    assert "I = 5" in text
    assert "0 spikes" in text


def test_loihi_cuba_doc_declares_hardware_validation_boundary() -> None:
    text = _loihi_cuba_doc().lower()

    assert "hardware validation boundary" in text
    assert "not a loihi 1 board execution claim" in text
    assert "lava" in text
    assert "loihi 1 hardware access" in text
    assert "board logs" in text
