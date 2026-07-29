# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — compatibility custody guard

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_identity_has_no_descriptor_or_scientific_claim() -> None:
    descriptor = _ROOT / "src/sc_neurocore/neurons/model_descriptors/KilincBhattMapNeuron.toml"
    assert not descriptor.exists()
    page = (_ROOT / "docs/api/models/kilinc_bhatt_map.md").read_text(encoding="utf-8")
    assert "compatibility alias" in page
    assert "not a scientific model identity" in page
