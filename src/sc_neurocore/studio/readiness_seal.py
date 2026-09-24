# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio verified-readiness seal for installations without a checkout

"""Carry the Studio's verified readiness into an installation that cannot re-verify it.

Verified readiness is re-derived on every read in a checkout: each facet
receipt is bound only while every subject it recorded still has the digest it
had. Several of those subjects — validator tests, benchmark scripts — are not
part of an installed distribution, so an installation that verified on its own
would find them missing and show every receipt-bound tier as lost.

The seal is that derivation, written in a checkout by
``tools/studio_readiness_seal.py --write`` and shipped as package data. A test
compares it with a fresh derivation, so a seal that no longer matches the
repository fails the suite. An installation serves the sealed record and says
so: the ``source`` of its verified block is ``sealed``, not ``receipts``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

SEAL_SCHEMA = "sc-neurocore.studio.verified-readiness-seal.v1"
SEAL_PATH = Path(__file__).resolve().parent / "verified_readiness.json"
SOURCE_RECEIPTS = "receipts"
SOURCE_SEALED = "sealed"
SOURCE_UNSEALED = "unsealed"


def checkout_available(repo_root: Path) -> bool:
    """Return whether ``repo_root`` is a checkout the receipts can be re-verified in.

    Receipt subjects include the validator tests and the descriptor sources, so
    both trees must be present; an installation or an unpacked source
    distribution has neither the tests nor this layout.
    """
    return (repo_root / "tests").is_dir() and (
        repo_root / "src" / "sc_neurocore" / "neurons" / "model_descriptors"
    ).is_dir()


def build_seal(
    names: list[str], verified_detail: Callable[[str], dict[str, Any]]
) -> dict[str, Any]:
    """Return the seal of ``names``, each derived by ``verified_detail``."""
    return {
        "schema": SEAL_SCHEMA,
        "derivation": (
            "the Studio's per-model verified readiness, derived in a checkout from "
            "facet receipts whose subjects all still matched"
        ),
        "models": {name: verified_detail(name) for name in sorted(names)},
    }


def render_seal(seal: Mapping[str, Any]) -> str:
    """Return the seal exactly as it is written: sorted keys, no timestamps."""
    return json.dumps(seal, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


@cache
def _load_seal(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SEAL_SCHEMA:
        raise ValueError(f"{path.name} is not a {SEAL_SCHEMA} document")
    return payload


def sealed_detail(class_name: str, path: Path = SEAL_PATH) -> dict[str, Any]:
    """Return the sealed verified block of one model.

    A model the seal does not hold was not verified when the distribution was
    built; it is reported unverified with that reason, never given tiers.
    """
    if not path.is_file():
        return _unsealed(class_name, "this installation carries no readiness seal")
    models = _load_seal(path)["models"]
    if class_name not in models:
        return _unsealed(class_name, "the readiness seal holds no record of this model")
    return {**models[class_name], "source": SOURCE_SEALED}


def _unsealed(class_name: str, reason: str) -> dict[str, Any]:
    return {
        "profile": None,
        "science_tier": 0,
        "science_label": "S0",
        "silicon_tier": None,
        "silicon_label": "none",
        "facets": [],
        "source": SOURCE_UNSEALED,
        "unsealed_reason": f"{class_name}: {reason}",
    }


__all__ = [
    "SEAL_PATH",
    "SEAL_SCHEMA",
    "SOURCE_RECEIPTS",
    "SOURCE_SEALED",
    "SOURCE_UNSEALED",
    "build_seal",
    "checkout_available",
    "render_seal",
    "sealed_detail",
]
