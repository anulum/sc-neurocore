#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Random-double corpus for the cross-runtime evidence seal

"""Write the random-double corpus both seal implementations are checked against.

The hand-written vectors carry the cases a person would think of. They cannot
carry the ones nobody thinks of: a double is 64 bits, its shortest
round-tripping decimal form is chosen by an algorithm, and the two runtimes
choose it separately. The divergences that matter — Python's ``1e-07`` against
JavaScript's ``1e-7``, an integral double written with or without a fractional
part — were found by drawing bit patterns at random, not by enumeration.

That experiment used to be run once and quoted. Quoting it proves nothing
about the implementations as they stand today, so it is a committed corpus
instead: this tool draws the bit patterns from a fixed seed, records each
double as a JSON number beside the canonical text this build produces for it,
and both test suites read the file. The Python suite proves the text has not
drifted; the TypeScript suite proves the other runtime reaches the same text
from the same JSON — which is the parity claim, checked on every run rather
than remembered.

A value whose canonical text changes makes the file stale and fails the
freshness case. That is deliberate: the canonical form is a contract, and
changing it changes every digest already issued under
``studio.evidence-seal.v1``, so it takes a new schema version rather than a
regenerated file.

Run it with no arguments to rewrite the corpus in place.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path
from typing import Any

from sc_neurocore.studio.evidence_seal import (
    EVIDENCE_SEAL_SCHEMA_VERSION,
    canonical_seal_text,
)

#: Where both test suites read the corpus from.
CORPUS_PATH = (
    Path(__file__).resolve().parents[1] / "studio/frontend/src/evidenceSealRandomVectors.json"
)
#: Seed of the bit-pattern draw. Fixed, so the corpus is reproducible.
CORPUS_SEED = 4409
#: How many doubles the corpus carries.
CORPUS_SIZE = 1024


def random_doubles(*, seed: int = CORPUS_SEED, count: int = CORPUS_SIZE) -> list[float]:
    """Return doubles drawn as random bit patterns, not as random magnitudes.

    Parameters
    ----------
    seed : int, optional
        Seed of the draw, so the corpus is reproducible.
    count : int, optional
        How many finite doubles to return.

    Returns
    -------
    list of float
        Finite doubles across the whole exponent range, including subnormals
        and negative zero, which a draw over magnitudes would rarely reach.
    """
    rng = random.Random(seed)
    values: list[float] = []
    while len(values) < count:
        candidate: float = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        # A NaN is not equal to itself; neither infinity is sealable. Both are
        # refused by the seal, and the refusal has its own cases.
        if candidate == candidate and abs(candidate) != float("inf"):
            values.append(candidate)
    return values


def build_corpus(*, seed: int = CORPUS_SEED, count: int = CORPUS_SIZE) -> dict[str, Any]:
    """Return the corpus document: each double beside its canonical text."""
    # A finite double is always sealable, so a refusal here is a defect in the
    # draw or in the seal and must surface rather than be turned into a note.
    vectors = [
        {"canonical": canonical_seal_text(value), "value": value}
        for value in random_doubles(seed=seed, count=count)
    ]
    return {
        "count": len(vectors),
        "note": (
            "Random-double corpus for studio.evidence-seal.v1, drawn as bit patterns from "
            f"seed {seed}. tests/test_studio_evidence_seal.py and evidenceSeal.test.ts both "
            "read this file; the TypeScript half reaches these canonical strings from the "
            "same JSON through its own parser, which is the cross-runtime parity claim. "
            "Regenerate with tools/generate_evidence_seal_vectors.py."
        ),
        "schema_version": EVIDENCE_SEAL_SCHEMA_VERSION,
        "seed": seed,
        "vectors": vectors,
    }


def render_corpus(corpus: dict[str, Any]) -> str:
    """Return the corpus as the text the repository stores."""
    return json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the corpus, or report whether the stored one is current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CORPUS_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the stored corpus matches this build, writing nothing",
    )
    args = parser.parse_args(argv)
    text = render_corpus(build_corpus())
    if args.check:
        stored = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if stored == text:
            return 0
        print(f"stale evidence seal corpus: {args.output}")
        return 1
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
