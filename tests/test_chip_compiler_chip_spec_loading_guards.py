# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChipSpecLoadingGuards from former test_chip_compiler.py

"""Focused suite: TestChipSpecLoadingGuards from former test_chip_compiler.py."""

from __future__ import annotations

from tests.chip_compiler_support import *  # noqa: F403

class TestChipSpecLoadingGuards:
    def test_load_chip_spec_rejects_invalid_json(self, tmp_path):
        from sc_neurocore.chip_compiler.chip_spec import load_chip_spec

        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid chip spec JSON"):
            load_chip_spec(bad)

    def test_validate_core_payload_rejects_non_object(self):
        from sc_neurocore.chip_compiler.chip_spec import _validate_core_payload

        with pytest.raises(ValueError, match="core must be an object"):
            _validate_core_payload([1, 2], source="spec")

    def test_required_float_rejects_non_numeric(self):
        from sc_neurocore.chip_compiler.chip_spec import _required_float

        with pytest.raises(ValueError, match="must be numeric"):
            _required_float({"freq": "fast"}, "freq", "spec")
