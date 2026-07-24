# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestClassifyMatch from former test_verify_dois.py

"""Focused suite: TestClassifyMatch from former test_verify_dois.py."""

from __future__ import annotations

from tests.verify_dois_support import *  # noqa: F403


class TestClassifyMatch:
    @staticmethod
    def _outcome(**over: Any) -> dict[str, Any]:
        base = {"registry": "crossref", "resolves": True, "first_author": "jahr", "year": 1990}
        base.update(over)
        return base

    def test_exact_match_is_verified(self) -> None:
        result = verify_dois.classify_match("jahr", 1990, self._outcome())
        assert result == {
            "author_match": True,
            "year_match": True,
            "verified": True,
            "translation": False,
        }

    def test_translation_verifies_on_resolution_despite_author_year_mismatch(self) -> None:
        # A declared translation deliberately keeps the original work's author and
        # year (Lapicque 1907) while the DOI points to the translator's paper
        # (Brunel 2007); resolution alone verifies it, and the literal mismatch is
        # recorded rather than treated as fabrication.
        result = verify_dois.classify_match(
            "lapicque", 1907, self._outcome(first_author="brunel", year=2007), translation=True
        )
        assert result["author_match"] is False
        assert result["year_match"] is False
        assert result["translation"] is True
        assert result["verified"] is True

    def test_translation_still_requires_the_doi_to_resolve(self) -> None:
        # The translation flag relaxes author/year, never the fabrication catcher.
        result = verify_dois.classify_match(
            "lapicque", 1907, {"registry": "crossref", "resolves": False}, translation=True
        )
        assert result["verified"] is False

    def test_year_within_one_still_verifies(self) -> None:
        # 2007 claimed, 2006 registered (a book's print-vs-online date) -> still verified.
        result = verify_dois.classify_match(
            "izhikevich", 2007, self._outcome(first_author="izhikevich", year=2006)
        )
        assert result["year_match"] is True
        assert result["verified"] is True

    def test_year_off_by_two_fails(self) -> None:
        result = verify_dois.classify_match("jahr", 1990, self._outcome(year=1992))
        assert result["year_match"] is False
        assert result["verified"] is False

    def test_wrong_paper_fails_on_author_even_if_it_resolves(self) -> None:
        # ConnorStevens class: the DOI resolved, but to an unrelated author -> not verified.
        result = verify_dois.classify_match(
            "connor", 1971, self._outcome(first_author="hall", year=1971)
        )
        assert result["author_match"] is False
        assert result["verified"] is False

    def test_non_resolving_doi_is_never_verified(self) -> None:
        result = verify_dois.classify_match(
            "jahr", 1990, {"registry": "crossref", "resolves": False}
        )
        assert result["verified"] is False

    def test_missing_claimed_year_fails_closed(self) -> None:
        result = verify_dois.classify_match("jahr", None, self._outcome())
        assert result["year_match"] is False
        assert result["verified"] is False

    def test_missing_claimed_author_fails_closed(self) -> None:
        result = verify_dois.classify_match("", 1990, self._outcome())
        assert result["author_match"] is False
        assert result["verified"] is False
