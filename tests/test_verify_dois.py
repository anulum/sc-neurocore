# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — unit tests for the provenance DOI verifier's pure logic

"""Unit tests for the network-free logic of ``tools/provenance/verify_dois.py``.

The diacritic folding and the author/year/verified classification are what keep
the DOI gate both strict (a fabricated or wrong DOI cannot pass) and free of
false positives (Llinás vs Llinas, a 2006/2007 book date). Registry routing is
checked with the HTTP layer monkeypatched, so no test here touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "provenance"))

import verify_dois  # noqa: E402


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


class TestResolveDoiRouting:
    def test_arxiv_doi_routes_to_datacite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, str] = {}

        def fake_datacite(doi: str) -> dict[str, Any]:
            seen["registry"] = "datacite"
            return {
                "first_author": "higuchi",
                "year": 2024,
                "title": "Balanced Resonate-and-Fire Neurons",
            }

        def fail_crossref(doi: str) -> dict[str, Any]:  # pragma: no cover - must not be called
            raise AssertionError("arXiv DOI must not be sent to Crossref")

        monkeypatch.setattr(verify_dois, "_resolve_datacite", fake_datacite)
        monkeypatch.setattr(verify_dois, "_resolve_crossref", fail_crossref)
        outcome = verify_dois.resolve_doi("10.48550/arXiv.2402.14603")
        assert outcome["registry"] == "datacite"
        assert outcome["resolves"] is True
        assert outcome["first_author"] == "higuchi"
        assert seen["registry"] == "datacite"

    def test_plain_doi_routes_to_crossref(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            verify_dois,
            "_resolve_crossref",
            lambda doi: {"first_author": "jahr", "year": 1990, "title": "t"},
        )
        outcome = verify_dois.resolve_doi("10.1523/jneurosci.10-09-03178.1990")
        assert outcome["registry"] == "crossref"
        assert outcome["resolves"] is True

    def test_not_found_marks_unresolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(verify_dois, "_resolve_crossref", lambda doi: {"__not_found__": True})
        outcome = verify_dois.resolve_doi("10.9999/does.not.exist")
        assert outcome["resolves"] is False

    def test_network_error_marks_unresolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(verify_dois, "_resolve_crossref", lambda doi: None)
        outcome = verify_dois.resolve_doi("10.1000/x")
        assert outcome["resolves"] is False
        assert outcome["error"] == "network-or-parse"
