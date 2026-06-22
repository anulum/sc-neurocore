#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Populate the measured behaviour facet for every model descriptor.

Runs the behaviour probe across the whole catalogue, writes the recorded
evidence manifest, and injects each model's measured ``behavior_tags`` into its
committed descriptor. The descriptor corpus is then re-rendered so the drift
gate stays satisfied. This is the slow offline act (it simulates every model
twice per current); the committed manifest is what the fast gate checks.

Usage::

    python tools/populate_behavior_tags.py            # probe + write tags + manifest
    python tools/populate_behavior_tags.py --check     # verify, do not write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomli_w

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sc_neurocore.neurons.behavior_taxonomy import validate_behavior_tags  # noqa: E402
from sc_neurocore.neurons.model_catalogue import (  # noqa: E402
    descriptor_path,
    load_descriptor_payload,
)
from sc_neurocore.neurons.models import _CLASS_TO_MODULE  # noqa: E402
from sc_neurocore.studio.behavior_probe import (  # noqa: E402
    BEHAVIOR_EVIDENCE_PATH,
    behavior_tags_for,
    probe_all_models,
)

# tools/generate_model_descriptors.py renders the corpus identically to the gate.
sys.path.insert(0, str(_REPO_ROOT / "tools"))
from generate_model_descriptors import write_corpus  # noqa: E402


def _inject_tags(manifest: Mapping[str, Any]) -> int:
    """Write each model's measured tags into its committed descriptor.

    Returns the number of descriptors whose tag set changed.
    """

    changed = 0
    for class_name in sorted(_CLASS_TO_MODULE):
        payload = load_descriptor_payload(class_name)
        if payload is None:
            continue
        tags = list(validate_behavior_tags(behavior_tags_for(class_name, manifest)))
        metadata = payload.setdefault("metadata", {})
        if metadata.get("behavior_tags") == tags:
            continue
        metadata["behavior_tags"] = tags
        descriptor_path(class_name).write_text(tomli_w.dumps(payload), encoding="utf-8")
        changed += 1
    return changed


def populate() -> tuple[int, int]:
    """Probe the catalogue, write the manifest, and inject tags.

    Returns ``(tagged_models, rerendered_descriptors)``.
    """

    manifest = probe_all_models()
    BEHAVIOR_EVIDENCE_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    changed = _inject_tags(manifest)
    rerendered = write_corpus()
    return changed, rerendered


def check() -> list[str]:
    """Re-probe and report descriptors whose tags disagree with the measurement."""

    manifest = probe_all_models()
    problems: list[str] = []
    for class_name in sorted(_CLASS_TO_MODULE):
        payload = load_descriptor_payload(class_name)
        if payload is None:
            continue
        committed = tuple(payload.get("metadata", {}).get("behavior_tags", ()))
        measured = validate_behavior_tags(behavior_tags_for(class_name, manifest))
        if tuple(committed) != measured:
            problems.append(
                f"{class_name}: committed {list(committed)} != measured {list(measured)}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Populate the measured behaviour facet.")
    parser.add_argument(
        "--check", action="store_true", help="re-probe and verify instead of writing"
    )
    args = parser.parse_args(argv)
    if args.check:
        problems = check()
        if problems:
            print(f"behaviour tags out of sync ({len(problems)}):")
            for problem in problems[:50]:
                print(f"  {problem}")
            return 1
        print(f"behaviour tags in sync ({len(_CLASS_TO_MODULE)} models)")
        return 0
    changed, rerendered = populate()
    print(
        f"probed {len(_CLASS_TO_MODULE)} models; updated {changed} tag set(s); "
        f"re-rendered {rerendered} descriptor(s); manifest at {BEHAVIOR_EVIDENCE_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
