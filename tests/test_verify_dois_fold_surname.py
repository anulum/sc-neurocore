# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFoldSurname from former test_verify_dois.py

"""Focused suite: TestFoldSurname from former test_verify_dois.py."""

from __future__ import annotations

import pytest
from tests.verify_dois_support import (
    verify_dois,
)


class TestFoldSurname:
    """Folding author strings to an ASCII family-name token."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Jahr, C. E.", "jahr"),
            ("Llinás, R.", "llinas"),  # á -> a
            ("Mihalaş, Ş.", "mihalas"),  # ş -> s
            ("Fourcaud-Trocmé, N.", "fourcaudtrocme"),  # hyphen dropped, é -> e
            ("Connor, J. A. & Stevens, C. F.", "connor"),  # only the leading family
            ("Chay, T. R.", "chay"),
            ("", ""),
            ("Şahin", "sahin"),
        ],
    )
    def test_folds_to_ascii_family_token(self, raw: str, expected: str) -> None:
        """Each author string folds to its lower-case ASCII family token."""
        assert verify_dois.fold_surname(raw) == expected

    def test_diacritic_and_plain_fold_identically(self) -> None:
        """A surname with and without its diacritic folds to the same token."""
        assert verify_dois.fold_surname("Llinás, R.") == verify_dois.fold_surname("Llinas, R.")
