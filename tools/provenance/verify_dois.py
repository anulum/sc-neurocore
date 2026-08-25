# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — provenance DOI verifier: resolve every descriptor DOI at its registry

"""Online verifier that resolves every neuron-descriptor provenance DOI at its
registry of record and records the outcome in a committed ledger.

Two registries are consulted by DOI prefix: DataCite for arXiv preprint DOIs
(``10.48550/arXiv.*``, which Crossref does not index) and Crossref for
everything else. For each DOI the verifier records whether it resolves, whether
the resolved first-author surname and publication year agree with the values the
descriptor claims (diacritic-insensitively), and a ``verified`` flag that is true
only when all three hold. The ledger is the offline source of truth checked by
``tests/test_provenance_doi_integrity.py``; a fabricated or mistyped DOI cannot
earn ``verified: true`` because the registry lookup that writes the ledger fails
on it.

Run ``python tools/provenance/verify_dois.py`` to rebuild the ledger.
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.neurons.model_catalogue import load_descriptor  # noqa: E402

MAILTO = "protoscience@anulum.li"
DESCRIPTOR_DIR = REPO_ROOT / "src" / "sc_neurocore" / "neurons" / "model_descriptors"
LEDGER_PATH = Path(__file__).resolve().parent / "doi_ledger.json"
_ARXIV_PREFIX = "10.48550/arxiv."
_REQUEST_TIMEOUT_S = 25
_CROSSREF_POLITE_DELAY_S = 0.3


def fold_surname(name: str) -> str:
    """Reduce an author string to a diacritic-free lowercase family-name token.

    Descriptor authors are stored ``"Family, I. I."`` (and occasionally
    ``"Family, I. & Other, J."``); only the leading family name is taken. NFKD
    decomposition drops combining marks so ``Llinás`` and ``Llinas``,
    ``Mihalaş`` and ``Mihalas``, ``Fourcaud-Trocmé`` and ``Fourcaud-Trocme`` all
    fold to the same token — the diacritic false positives that a naïve string
    compare would raise.
    """
    family = name.split(",")[0].split("&")[0].strip()
    decomposed = unicodedata.normalize("NFKD", family)
    return "".join(
        ch for ch in decomposed if ch.isalpha() and not unicodedata.combining(ch)
    ).lower()


def _http_json(url: str) -> dict[str, Any] | None:
    # Only ever open the hardcoded https Crossref/DataCite API endpoints; reject any other
    # scheme so a crafted DOI can never turn this into a file:// or custom-scheme read (B310).
    if not url.startswith("https://"):
        raise ValueError(f"refusing to open a non-HTTPS URL: {url!r}")
    request = urllib.request.Request(
        url, headers={"User-Agent": f"sc-neurocore-doi-verify ({MAILTO})"}
    )
    try:
        with urllib.request.urlopen(  # nosec B310 - scheme guarded to https above
            request, timeout=_REQUEST_TIMEOUT_S
        ) as response:
            if response.status != 200:
                return None
            parsed = json.loads(response.read().decode())
            return parsed if isinstance(parsed, dict) else None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"__not_found__": True}
        return None
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def _resolve_crossref(doi: str) -> dict[str, Any] | None:
    payload = _http_json(
        f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}"
    )
    if payload is None or payload.get("__not_found__"):
        return payload
    message = payload.get("message", {})
    authors = message.get("author", []) or []
    first_author = fold_surname(authors[0].get("family", "")) if authors else ""
    year = None
    for key in ("published-print", "published-online", "issued"):
        parts = message.get(key, {}).get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break
    return {"first_author": first_author, "year": year, "title": (message.get("title") or [""])[0]}


def _resolve_datacite(doi: str) -> dict[str, Any] | None:
    payload = _http_json(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
    if payload is None or payload.get("__not_found__"):
        return payload
    attributes = payload.get("data", {}).get("attributes", {})
    creators = attributes.get("creators", []) or []
    first_author = fold_surname(creators[0].get("name", "")) if creators else ""
    titles = attributes.get("titles", []) or [{}]
    return {
        "first_author": first_author,
        "year": attributes.get("publicationYear"),
        "title": titles[0].get("title", ""),
    }


def resolve_doi(doi: str) -> dict[str, Any]:
    """Resolve ``doi`` at the correct registry and report the outcome.

    Returns a dict with ``registry`` and ``resolves``; when it resolves, also the
    registry's ``first_author`` (folded surname), ``year`` and ``title``.
    """
    registry = "datacite" if doi.lower().startswith(_ARXIV_PREFIX) else "crossref"
    resolved = _resolve_datacite(doi) if registry == "datacite" else _resolve_crossref(doi)
    if resolved is None:
        return {"registry": registry, "resolves": False, "error": "network-or-parse"}
    if resolved.get("__not_found__"):
        return {"registry": registry, "resolves": False}
    return {"registry": registry, "resolves": True, **resolved}


def classify_match(
    claimed_first: str,
    claimed_year: int | None,
    outcome: dict[str, Any],
    translation: bool = False,
) -> dict[str, bool]:
    """Decide author/year/verified verdicts from a claimed pair and a registry outcome.

    Successful resolution and an exact first-author surname are the fabrication
    catchers and stay exact; the publication year is a weak, ambiguous signal (a
    registry's online date routinely trails a book's or preprint's cited print year
    by a year), so it is confirmed within +/-1. A wrong DOI that resolves to a
    different paper still fails on the author check.

    When ``translation`` is set the descriptor deliberately keeps the original
    work's author and year while the DOI points to a later translation/reprint, so
    the literal author/year comparison is expected to differ. The DOI still has to
    *resolve* (the fabrication catcher), and that resolution alone verifies it; the
    literal ``author_match``/``year_match`` are recorded unchanged for the record.
    """
    author_match = bool(claimed_first) and claimed_first == outcome.get("first_author")
    registry_year = outcome.get("year")
    year_match = (
        claimed_year is not None
        and registry_year is not None
        and abs(int(claimed_year) - int(registry_year)) <= 1
    )
    resolves = bool(outcome.get("resolves"))
    verified = resolves and (translation or (author_match and year_match))
    return {
        "author_match": author_match,
        "year_match": year_match,
        "verified": verified,
        "translation": translation,
    }


def build_ledger() -> dict[str, dict[str, Any]]:
    """Resolve and cross-check every descriptor provenance DOI, keyed by DOI."""
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    ledger: dict[str, dict[str, Any]] = {}
    for toml_path in sorted(DESCRIPTOR_DIR.glob("*.toml")):
        descriptor = load_descriptor(toml_path.stem)
        provenance = getattr(descriptor, "provenance", None)
        if provenance is None:
            continue
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8")).get("provenance", {})
        if not isinstance(raw, dict):
            continue
        primary_authors = list(getattr(provenance, "authors", []) or [])
        primary_year = getattr(provenance, "year", None)
        primary_translation = bool(getattr(provenance, "doi_is_translation", False))
        claims: list[tuple[str, str, list[str], int | None, bool]] = []
        for key, value in sorted(raw.items()):
            if key != "doi" and not key.endswith("_doi"):
                continue
            if not isinstance(value, str) or not value:
                continue
            prefix = key.removesuffix("_doi")
            authors_value = raw.get(f"{prefix}_authors", primary_authors)
            authors = (
                [str(author) for author in authors_value]
                if isinstance(authors_value, list)
                else primary_authors
            )
            year_value = raw.get(f"{prefix}_year", primary_year)
            year = year_value if isinstance(year_value, int) else primary_year
            translation = bool(raw.get(f"{prefix}_doi_is_translation", primary_translation))
            label = toml_path.stem if key == "doi" else f"{toml_path.stem}:{key}"
            claims.append((label, value, authors, year, translation))

        for label, doi, claimed_authors, claimed_year, translation in claims:
            if doi in ledger:
                continue
            claimed_first = fold_surname(claimed_authors[0]) if claimed_authors else ""
            outcome = resolve_doi(doi)
            if outcome["registry"] == "crossref":
                time.sleep(_CROSSREF_POLITE_DELAY_S)
            match = classify_match(claimed_first, claimed_year, outcome, translation)
            ledger[doi] = {
                "model": label,
                "registry": outcome["registry"],
                "resolves": outcome["resolves"],
                "author_match": match["author_match"],
                "year_match": match["year_match"],
                "translation": match["translation"],
                "verified": match["verified"],
                "registry_first_author": outcome.get("first_author"),
                "registry_year": outcome.get("year"),
                "title": outcome.get("title", "")[:120],
                "verified_utc": stamp,
            }
    return ledger


def main() -> int:
    ledger = build_ledger()
    LEDGER_PATH.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    unverified = {doi: row for doi, row in ledger.items() if not row["verified"]}
    print(f"resolved {len(ledger)} provenance DOIs -> {LEDGER_PATH.relative_to(REPO_ROOT)}")
    print(f"verified: {len(ledger) - len(unverified)}  unverified: {len(unverified)}")
    for doi, row in sorted(unverified.items()):
        reason = (
            "does-not-resolve"
            if not row["resolves"]
            else ("author-mismatch" if not row["author_match"] else "year-mismatch")
        )
        print(f"  UNVERIFIED {row['model']} {doi} [{row['registry']}] {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
