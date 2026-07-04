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
    request = urllib.request.Request(
        url, headers={"User-Agent": f"sc-neurocore-doi-verify ({MAILTO})"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as response:
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


def build_ledger() -> dict[str, dict[str, Any]]:
    """Resolve and cross-check every descriptor provenance DOI, keyed by DOI."""
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    ledger: dict[str, dict[str, Any]] = {}
    for toml_path in sorted(DESCRIPTOR_DIR.glob("*.toml")):
        descriptor = load_descriptor(toml_path.stem)
        provenance = getattr(descriptor, "provenance", None)
        doi = getattr(provenance, "doi", None) if provenance else None
        if not doi or doi in ledger:
            continue
        claimed_authors = list(getattr(provenance, "authors", []) or [])
        claimed_first = fold_surname(claimed_authors[0]) if claimed_authors else ""
        claimed_year = getattr(provenance, "year", None)
        outcome = resolve_doi(doi)
        if outcome["registry"] == "crossref":
            time.sleep(_CROSSREF_POLITE_DELAY_S)
        author_match = bool(claimed_first) and claimed_first == outcome.get("first_author")
        # The first author and successful resolution are the fabrication catchers and stay
        # exact; the publication year is a weak, ambiguous signal (a registry's online date
        # routinely trails a book's or preprint's cited print year by a year), so it is
        # confirmed within +/-1 rather than exactly. A wrong DOI still fails on author_match.
        registry_year = outcome.get("year")
        year_match = (
            claimed_year is not None
            and registry_year is not None
            and abs(int(claimed_year) - int(registry_year)) <= 1
        )
        ledger[doi] = {
            "model": toml_path.stem,
            "registry": outcome["registry"],
            "resolves": outcome["resolves"],
            "author_match": author_match,
            "year_match": year_match,
            "verified": outcome["resolves"] and author_match and year_match,
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
