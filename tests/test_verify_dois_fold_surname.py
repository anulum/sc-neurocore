# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFoldSurname from former test_verify_dois.py

"""Focused suite: TestFoldSurname from former test_verify_dois.py."""

from __future__ import annotations

from tests.verify_dois_support import *  # noqa: F403


class TestFoldSurname:
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
        assert verify_dois.fold_surname(raw) == expected

    def test_diacritic_and_plain_fold_identically(self) -> None:
        assert verify_dois.fold_surname("Llinás, R.") == verify_dois.fold_surname("Llinas, R.")
