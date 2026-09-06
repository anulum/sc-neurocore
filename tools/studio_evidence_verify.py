#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Offline Studio evidence pack verifier

"""Recheck an exported Studio evidence pack without the Studio.

A bundle carries the verdict the exporter reached. Someone receiving the pack
has no reason to take that on trust: the exporter wrote both the evidence and
the verdict. This reads the pack from disk and reaches its own verdict —
recomputing every file digest against the manifest, re-sealing every subject
against its receipt, and re-resolving the dependency graph — then reports where
its finding and the recorded one differ.

The pack directory is the one containing ``evidence/manifest.json``.

Usage::

    PYTHONPATH=src:. python tools/studio_evidence_verify.py <pack directory>
    PYTHONPATH=src:. python tools/studio_evidence_verify.py <pack> --json out.json
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys

from sc_neurocore.studio.evidence_chain import (
    EVIDENCE_VERIFIER_VERSION,
    EvidenceChainReport,
    verify_evidence_chain,
)
from sc_neurocore.studio.evidence_receipt import (
    EvidenceReceiptError,
    read_evidence_receipt,
)

MANIFEST_RELATIVE_PATH = "evidence/manifest.json"
CHAIN_RELATIVE_PATH = "evidence/chain.json"


class EvidencePackError(ValueError):
    """Raised when a pack cannot be read as a Studio evidence bundle."""


def load_manifest(pack: Path) -> dict[str, object]:
    """Return the manifest of an exported pack.

    Parameters
    ----------
    pack : Path
        Directory containing ``evidence/manifest.json``.

    Returns
    -------
    dict
        The parsed manifest.

    Raises
    ------
    EvidencePackError
        If the manifest is absent or is not a JSON object.
    """
    manifest_path = pack / MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise EvidencePackError(f"{manifest_path} is not a file; this is not an evidence pack.")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise EvidencePackError(f"{manifest_path} does not hold a JSON object.")
    return document


def check_digests(pack: Path, manifest: Mapping[str, object]) -> list[str]:
    """Return one message per manifest entry whose file does not match it.

    Parameters
    ----------
    pack : Path
        Directory containing the pack.
    manifest : mapping
        The parsed manifest.

    Returns
    -------
    list of str
        Empty when every declared file is present at its declared digest and
        size.
    """
    problems: list[str] = []
    for entry in _entries(manifest):
        relative_path = entry.get("bundle_path")
        if not isinstance(relative_path, str):
            problems.append(f"A manifest entry of type {entry.get('type')!r} names no file.")
            continue
        path = pack / relative_path
        if not path.is_file():
            problems.append(f"{relative_path} is named by the manifest but absent from the pack.")
            continue
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry.get("sha256"):
            problems.append(
                f"{relative_path} hashes to {digest}, not to the recorded {entry.get('sha256')}."
            )
        size = entry.get("size_bytes")
        if isinstance(size, int) and size != len(payload):
            problems.append(f"{relative_path} is {len(payload)} bytes, not the recorded {size}.")
    return problems


def read_subjects(pack: Path, manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return every receipt-bearing subject the manifest names, read from disk.

    Parameters
    ----------
    pack : Path
        Directory containing the pack.
    manifest : mapping
        The parsed manifest.

    Returns
    -------
    dict of str to dict
        Pack-relative path to payload, for each file that parses as a JSON
        object carrying a receipt.
    """
    subjects: dict[str, dict[str, object]] = {}
    for entry in _entries(manifest):
        relative_path = entry.get("bundle_path")
        if not isinstance(relative_path, str) or relative_path == CHAIN_RELATIVE_PATH:
            continue
        path = pack / relative_path
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if read_evidence_receipt(payload) is None:
            continue
        subjects[relative_path] = payload
    return subjects


def compare_with_recorded(pack: Path, report: EvidenceChainReport) -> list[str]:
    """Return one message per subject on which the pack and this run disagree.

    Parameters
    ----------
    pack : Path
        Directory containing the pack.
    report : EvidenceChainReport
        What this run found.

    Returns
    -------
    list of str
        Empty when the recorded chain document agrees with this run, or when
        the pack carries no chain document to compare against.
    """
    chain_path = pack / CHAIN_RELATIVE_PATH
    if not chain_path.is_file():
        return [f"{CHAIN_RELATIVE_PATH} is absent; the pack records no verdict of its own."]
    document = json.loads(chain_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise EvidencePackError(f"{chain_path} does not hold a JSON object.")
    recorded_entries = document.get("entries")
    recorded: dict[str, str] = {}
    if isinstance(recorded_entries, list):
        for entry in recorded_entries:
            if isinstance(entry, dict):
                name = entry.get("name")
                verdict = entry.get("verdict")
                if isinstance(name, str) and isinstance(verdict, str):
                    recorded[name] = verdict
    problems: list[str] = []
    for entry in report.entries:
        if entry.name not in recorded:
            problems.append(f"{entry.name} carries a receipt the recorded chain does not list.")
        elif recorded[entry.name] != entry.verdict:
            problems.append(
                f"{entry.name} is recorded as {recorded[entry.name]} but verifies as "
                f"{entry.verdict}."
            )
    return problems


def verify_pack(pack: Path) -> dict[str, object]:
    """Verify one exported pack and return the finding.

    Parameters
    ----------
    pack : Path
        Directory containing ``evidence/manifest.json``.

    Returns
    -------
    dict
        ``verified``, the per-subject chain report, and the digest and
        agreement problems found.

    Raises
    ------
    EvidencePackError
        If the pack cannot be read at all.
    EvidenceReceiptError
        If a receipt in the pack is malformed.
    """
    manifest = load_manifest(pack)
    digest_problems = check_digests(pack, manifest)
    subjects = read_subjects(pack, manifest)
    report = verify_evidence_chain(subjects)
    agreement_problems = compare_with_recorded(pack, report)
    return {
        "agreement_problems": agreement_problems,
        "bundle_id": manifest.get("bundle_id", ""),
        "chain": report.to_public_dict(),
        "digest_problems": digest_problems,
        "subject_count": len(subjects),
        "verified": bool(
            not digest_problems and not agreement_problems and not report.contradicted()
        ),
        "verifier_version": EVIDENCE_VERIFIER_VERSION,
    }


def render_report(finding: Mapping[str, object]) -> str:
    """Return the operator-readable form of a verification finding.

    Parameters
    ----------
    finding : mapping
        The result of :func:`verify_pack`.

    Returns
    -------
    str
        A report naming the counts, every problem, and every subject that did
        not verify.
    """
    chain = finding["chain"]
    if not isinstance(chain, Mapping):
        raise TypeError("A verification finding expected a mapping for its chain.")
    lines = [
        f"Studio evidence pack {finding.get('bundle_id', '')}",
        f"  verifier: {finding.get('verifier_version')}",
        f"  subjects with receipts: {finding.get('subject_count')}",
        f"  verdicts: {_counts(chain.get('verdict_counts'))}",
    ]
    for label, key in (("digest", "digest_problems"), ("agreement", "agreement_problems")):
        problems = finding.get(key)
        if isinstance(problems, list) and problems:
            lines.append(f"  {label} problems:")
            lines.extend(f"    {problem}" for problem in problems)
    entries = chain.get("entries")
    unverified: list[Mapping[str, object]] = []
    if isinstance(entries, list):
        unverified = [
            entry
            for entry in entries
            if isinstance(entry, Mapping) and entry.get("verdict") != "verified"
        ]
    if unverified:
        lines.append("  subjects that did not verify:")
        lines.extend(
            f"    {entry.get('verdict')}: {entry.get('name')} — {entry.get('reason')}"
            for entry in unverified
        )
    lines.append(f"  result: {'verified' if finding.get('verified') else 'NOT VERIFIED'}")
    return "\n".join(lines)


def _counts(counts: object) -> str:
    if not isinstance(counts, Mapping) or not counts:
        return "none"
    return ", ".join(f"{name} {value}" for name, value in sorted(counts.items()))


def _entries(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise EvidencePackError("An evidence manifest must list its entries.")
    return [entry for entry in entries if isinstance(entry, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:
    """Verify a pack from the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        ``0`` when the pack verified, ``1`` when it did not or could not be
        read.
    """
    parser = argparse.ArgumentParser(
        description="Recheck an exported Studio evidence pack without the Studio."
    )
    parser.add_argument("pack", type=Path, help="directory containing evidence/manifest.json")
    parser.add_argument("--json", type=Path, default=None, help="write the finding to this file")
    arguments = parser.parse_args(argv)
    try:
        finding = verify_pack(arguments.pack)
    except (EvidencePackError, EvidenceReceiptError, json.JSONDecodeError) as exc:
        print(f"evidence pack cannot be verified: {exc}", file=sys.stderr)
        return 1
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(finding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(render_report(finding))
    if not finding["verified"]:
        print("evidence pack did not verify", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
