#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — model descriptor corpus generator

"""Generate and maintain the on-disk model descriptor corpus.

For every registered neuron model this writes a curatable v2 descriptor TOML to
``neurons/model_descriptors/<ClassName>.toml``. Structural fields (parameters,
defaults, state, timestep) are read from the model code; any curation already on
disk (units, ranges, meaning, taxonomy, provenance, backends, reproducibility,
notes) is preserved by merging. Re-running is therefore idempotent and safe over
curated descriptors.

Usage::

    python tools/generate_model_descriptors.py            # write/refresh corpus
    python tools/generate_model_descriptors.py --check    # CI: fail on drift

``--check`` regenerates each descriptor in memory, merges committed curation, and
fails when the serialized result differs from the committed file — so the corpus
can never silently drift from the model code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tomli_w

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sc_neurocore.neurons.descriptor_generator import (  # noqa: E402
    generate_descriptor_payload,
    merge_descriptor_payloads,
)
from sc_neurocore.neurons.model_catalogue import (  # noqa: E402
    DESCRIPTOR_DIR,
    descriptor_path,
    load_descriptor_payload,
)
from sc_neurocore.neurons.model_taxonomy import canonical_model_name  # noqa: E402
from sc_neurocore.neurons.models import _CLASS_TO_MODULE  # noqa: E402


# The SPDX provenance header every committed descriptor carries (uniform across the
# corpus). Emitting it keeps regenerated files SPDX-compliant and makes ``--check``
# meaningful — without it every committed descriptor read as "out of sync" purely because
# the generator dropped the header, masking genuine body drift.
_DESCRIPTOR_HEADER = (
    "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
    "# Commercial license available\n"
    "# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.\n"
    "# © Code 2020–2026 Miroslav Šotek. All rights reserved.\n"
    "# ORCID: 0009-0009-3560-0851\n"
    "# Contact: www.anulum.li | protoscience@anulum.li\n"
    "# SC-NeuroCore — Source/config provenance header\n\n"
)


def _rendered_descriptor(class_name: str) -> str:
    """Return the serialized merged descriptor TOML for a model, with the SPDX header."""

    regenerated = generate_descriptor_payload(class_name)
    committed = load_descriptor_payload(class_name)
    payload = (
        merge_descriptor_payloads(committed, regenerated) if committed is not None else regenerated
    )
    return _DESCRIPTOR_HEADER + tomli_w.dumps(payload)


def _descriptor_identities() -> tuple[str, ...]:
    """Return unique catalogue identities, excluding compatibility aliases.

    Alias classes resolve to their canonical descriptor in ``descriptor_path``.
    Rendering an alias after its canonical class would overwrite that shared file
    with alias metadata and make the generator permanently non-idempotent.
    """

    return tuple(
        class_name
        for class_name in sorted(_CLASS_TO_MODULE)
        if canonical_model_name(class_name) == class_name
    )


def write_corpus() -> int:
    """Write or refresh every descriptor; return the number written."""

    DESCRIPTOR_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for class_name in _descriptor_identities():
        rendered = _rendered_descriptor(class_name)
        path = descriptor_path(class_name)
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            path.write_text(rendered, encoding="utf-8")
            written += 1
    return written


def check_corpus() -> list[str]:
    """Return descriptors that are missing or out of sync with the code."""

    problems: list[str] = []
    for class_name in _descriptor_identities():
        path = descriptor_path(class_name)
        if not path.is_file():
            problems.append(f"missing descriptor: {class_name}")
            continue
        if path.read_text(encoding="utf-8") != _rendered_descriptor(class_name):
            problems.append(f"out-of-sync descriptor: {class_name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the model descriptor corpus.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed corpus matches the model code instead of writing",
    )
    args = parser.parse_args(argv)
    if args.check:
        problems = check_corpus()
        if problems:
            print(f"descriptor corpus is out of sync ({len(problems)}):")
            for problem in problems[:50]:
                print(f"  {problem}")
            return 1
        print(f"descriptor corpus is in sync ({len(_descriptor_identities())} identities)")
        return 0
    written = write_corpus()
    print(f"wrote {written} descriptor(s) to {DESCRIPTOR_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
