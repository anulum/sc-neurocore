# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResolveDoiRouting from former test_verify_dois.py

"""Focused suite: TestResolveDoiRouting from former test_verify_dois.py."""

from __future__ import annotations

from tests.verify_dois_support import *  # noqa: F403


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
